import pandas as pd
import numpy as np
import ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from config import *
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        self._futures_exchange_info = None
        self._futures_info_ts = None
        self._mtf_cache = {}
        self._deriv_cache = {}
        
    def get_klines(self, symbol, interval='30m', limit=100, exclude_last_open=True):
        """바이낸스에서 캔들스틱 데이터 가져오기"""
        try:
            # 선물(UM) 기준 K라인 사용
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 미종가 캔들 제외 옵션
            if exclude_last_open and len(df) > 0:
                df = df.iloc[:-1]
            
            return df
        except BinanceAPIException as e:
            logger.error(f"바이낸스 API 오류: {e}")
            return None

    def _get_futures_exchange_info(self):
        try:
            from datetime import datetime, timedelta
            if self._futures_exchange_info and self._futures_info_ts and datetime.utcnow() - self._futures_info_ts < timedelta(minutes=10):
                return self._futures_exchange_info
            info = self.client.futures_exchange_info()
            self._futures_exchange_info = info
            self._futures_info_ts = datetime.utcnow()
            return info
        except Exception as e:
            logger.error(f"선물 exchangeInfo 조회 오류: {e}")
            return None

    # ===== 파생 데이터 캐시 유틸 =====
    def _get_cached(self, key: str, ttl_sec: int):
        now = time.time()
        v = self._deriv_cache.get(key)
        if v and now - v[1] < ttl_sec:
            return v[0]
        return None

    def _set_cached(self, key: str, value, ttl_sec: int):
        self._deriv_cache[key] = (value, time.time(), ttl_sec)

    def get_funding_rate(self, symbol: str) -> float:
        try:
            key = f"funding:{symbol}"
            cached = self._get_cached(key, ttl_sec=60)
            if cached is not None:
                return cached
            fr = self.client.futures_funding_rate(symbol=symbol, limit=1)
            rate = float(fr[-1]['fundingRate']) * 100 if fr else float('nan')
            self._set_cached(key, rate, 60)
            return rate
        except Exception:
            return float('nan')

    def get_open_interest_usdt(self, symbol: str) -> float:
        try:
            key = f"oi:{symbol}"
            cached = self._get_cached(key, ttl_sec=60)
            if cached is not None:
                return cached
            oi = self.client.futures_open_interest(symbol=symbol)
            price = self.get_last_price(symbol)
            notional = float(oi['openInterest']) * float(price) if oi and price == price else float('nan')
            self._set_cached(key, notional, 60)
            return notional
        except Exception:
            return float('nan')

    def get_long_short_ratio(self, symbol: str, period: str = '5m') -> float:
        try:
            key = f"lsr:{symbol}:{period}"
            cached = self._get_cached(key, ttl_sec=180)
            if cached is not None:
                return cached
            data = self.client.futures_longshort_account_ratio(symbol=symbol, period=period, limit=1)
            ratio = float(data[-1]['longShortRatio']) if data else float('nan')
            self._set_cached(key, ratio, 180)
            return ratio
        except Exception:
            return float('nan')

    def _get_symbol_filters(self, symbol: str):
        info = self._get_futures_exchange_info()
        if not info:
            return None
        try:
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    filters = {f['filterType']: f for f in s.get('filters', [])}
                    return {
                        'lot_size': filters.get('LOT_SIZE', {}),
                        'market_lot_size': filters.get('MARKET_LOT_SIZE', {}),
                        'min_notional': filters.get('MIN_NOTIONAL', {}),
                        'price_filter': filters.get('PRICE_FILTER', {})
                    }
        except Exception:
            pass
        return None

    @staticmethod
    def _round_step(value: float, step: float, mode: str = 'down') -> float:
        if step is None or step == 0:
            return value
        import math
        q = value / step
        if mode == 'down':
            q = math.floor(q)
        elif mode == 'up':
            q = math.ceil(q)
        else:
            q = round(q)
        return q * step

    @staticmethod
    def _round_price_to_tick(price: float, tick_size: float, mode: str = 'nearest') -> float:
        if tick_size is None or tick_size == 0:
            return round(price, 6)
        mode_use = mode if mode in ('down', 'up', 'nearest') else 'nearest'
        if mode_use == 'nearest':
            import math
            steps = price / tick_size
            return round(steps) * tick_size
        return MarketAnalyzer._round_step(price, tick_size, mode_use)
            
    def calculate_technical_indicators(self, df):
        """기술적 지표 계산"""
        if df is None or len(df) < 50:
            return None
            
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # 볼린저 밴드
            bollinger = ta.volatility.BollingerBands(df['close'], window=BOLLINGER_PERIOD, window_dev=BOLLINGER_STD)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 이동평균 (NaN 유지)
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # 볼륨 지표
            # ta.volume.volume_sma는 버전에 따라 제공되지 않으므로 롤링 평균으로 대체
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            # OBV, MFI 추가
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'], fillna=True).on_balance_volume()
            df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=True).money_flow_index()
            
            # ADX (추세 강도)
            df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True).adx()
            
            # ATR (Average True Range)
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            return df
        except Exception as e:
            logger.error(f"기술적 지표 계산 오류: {e}")
            return None
            
    def analyze_market_strength(self, df):
        """시장 강도 분석"""
        if df is None or len(df) < 2:
            return 0
            
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 가격 모멘텀
            price_momentum = (current['close'] - prev['close']) / prev['close'] * 100
            
            # 볼륨 모멘텀
            volume_momentum = (current['volume'] - prev['volume']) / prev['volume'] * 100 if prev['volume'] > 0 else 0
            
            # RSI 모멘텀
            rsi_momentum = current['rsi'] - prev['rsi'] if not pd.isna(current['rsi']) and not pd.isna(prev['rsi']) else 0
            
            # MACD 신호
            macd_signal = 1 if current['macd'] > current['macd_signal'] else -1
            
            # 볼린저 밴드 위치
            bb_position = (current['close'] - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'])
            
            # 레짐 지표
            atr_pct = float(current['atr'] / current['close'] * 100) if current['close'] else 0
            bbw = float(current['bb_width']) if not pd.isna(current['bb_width']) else 0
            regime_ok = (ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX) and (BB_WIDTH_MIN <= bbw <= BB_WIDTH_MAX)

            # 종합 점수 계산 (추세/모멘텀 가중치 반영)
            sma200_val = current.get('sma_200')
            trend_bias = 0
            if pd.notna(sma200_val):
                trend_bias = 1 if current['close'] >= sma200_val else -1
            adx_val = current.get('adx')
            adx_strength = 1 if (pd.notna(adx_val) and adx_val >= ADX_MIN) else 0
            score = (
                price_momentum * 0.25 +
                volume_momentum * 0.2 +
                rsi_momentum * 0.15 +
                macd_signal * 0.2 +
                (bb_position - 0.5) * 0.1 +
                trend_bias * 0.05 +
                adx_strength * 0.05
            )
            
            return score if regime_ok else score * 0.3
        except Exception as e:
            logger.error(f"시장 강도 분석 오류: {e}")
            return 0
            
    def _mtf_confirm(self, symbol, desired_direction):
        """상위 타임프레임 컨펌: 1h/4h SMA200·MACD·ADX 기준"""
        if not ENABLE_MTF_CONFIRM:
            return True
        try:
            from datetime import datetime, timedelta
            for iv in MTF_CONFIRM_INTERVALS:
                cache_key = (symbol, iv, desired_direction)
                cached = self._mtf_cache.get(cache_key)
                if cached and datetime.utcnow() - cached[1] < timedelta(minutes=MTF_CONFIRM_CACHE_MINUTES):
                    if not cached[0]:
                        return False
                    else:
                        continue
                hdf = self.get_klines(symbol, interval=iv, limit=180, exclude_last_open=True)
                hdf = self.calculate_technical_indicators(hdf)
                if hdf is None or len(hdf) < 50:
                    self._mtf_cache[cache_key] = (False, datetime.utcnow())
                    return False
                cur = hdf.iloc[-1]
                # ADX 임계 적응: 변동성 높으면 상향
                atr_pct = float(cur['atr'] / cur['close'] * 100) if cur.get('atr') and cur['close'] else 0
                adx_th = ADX_MIN + (5 if atr_pct >= 2.0 else 0)
                sma200 = cur.get('sma_200')
                macd_val = cur.get('macd')
                macd_sig = cur.get('macd_signal')
                adx_val = cur.get('adx')
                up_ok = (pd.notna(sma200) and pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(adx_val) and (cur['close'] >= sma200) and (macd_val >= macd_sig) and (adx_val >= adx_th))
                down_ok = (pd.notna(sma200) and pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(adx_val) and (cur['close'] < sma200) and (macd_val < macd_sig) and (adx_val >= adx_th))
                if not (up_ok or down_ok):
                    self._mtf_cache[cache_key] = (False, datetime.utcnow())
                    return False
                if desired_direction == 'LONG' and not up_ok:
                    self._mtf_cache[cache_key] = (False, datetime.utcnow())
                    return False
                if desired_direction == 'SHORT' and not down_ok:
                    self._mtf_cache[cache_key] = (False, datetime.utcnow())
                    return False
                self._mtf_cache[cache_key] = (True, datetime.utcnow())
            return True
        except Exception:
            return False

    def _recent_swing_levels(self, df, lookback: int = 5):
        try:
            recent = df.iloc[-lookback:]
            swing_low = float(recent['low'].min())
            swing_high = float(recent['high'].max())
            return swing_low, swing_high
        except Exception:
            return None, None

    def generate_signal(self, symbol, df, interval='30m'):
        """시그널 생성"""
        if df is None or len(df) < 50:
            return None
            
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 시장 강도 분석
            market_strength = self.analyze_market_strength(df)
            
            # 진입 조건 확인
            entry_conditions = []
            
            # RSI 조건
            if current['rsi'] < RSI_OVERSOLD + 5:
                entry_conditions.append(('RSI 과매도', 1))
            elif current['rsi'] > RSI_OVERBOUGHT - 5:
                entry_conditions.append(('RSI 과매수', -1))
                
            # MACD 조건
            if current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                entry_conditions.append(('MACD 골든크로스', 1))
            elif current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                entry_conditions.append(('MACD 데드크로스', -1))
                
            # 볼린저 밴드 조건
            if current['close'] < current['bb_lower']:
                entry_conditions.append(('볼린저 밴드 하단 터치', 1))
            elif current['close'] > current['bb_upper']:
                entry_conditions.append(('볼린저 밴드 상단 터치', -1))
                
            # 이동평균 조건
            if (pd.notna(current.get('ema_12')) and pd.notna(current.get('ema_26')) and pd.notna(prev.get('ema_12')) and pd.notna(prev.get('ema_26')) and current['ema_12'] > current['ema_26'] and prev['ema_12'] <= prev['ema_26']):
                entry_conditions.append(('EMA 골든크로스', 1))
            elif (pd.notna(current.get('ema_12')) and pd.notna(current.get('ema_26')) and pd.notna(prev.get('ema_12')) and pd.notna(prev.get('ema_26')) and current['ema_12'] < current['ema_26'] and prev['ema_12'] >= prev['ema_26']):
                entry_conditions.append(('EMA 데드크로스', -1))
                
            # 볼륨 조건
            if current['volume_ratio'] > 1.5:
                entry_conditions.append(('높은 거래량', 1 if market_strength > 0 else -1))
            # 추세 필터: SMA200
            if pd.notna(current.get('sma_200')) and current['close'] >= current['sma_200']:
                entry_conditions.append(('장기상승 추세(SMA200 위)', 1))
            else:
                if pd.notna(current.get('sma_200')):
                    entry_conditions.append(('장기하락 추세(SMA200 아래)', -1))
            # ADX 필터: 추세 강도
            if pd.notna(current.get('adx')) and current['adx'] >= 20:
                entry_conditions.append(('추세 강함(ADX≥20)', 1 if current['macd'] >= current['macd_signal'] else -1))
            # MFI 과매수/과매도 보정
            if not pd.isna(current['mfi']):
                if current['mfi'] <= 20:
                    entry_conditions.append(('MFI 과매도', 1))
                elif current['mfi'] >= 80:
                    entry_conditions.append(('MFI 과매수', -1))
                
            # 종합 점수 계산
            total_score = sum(score for _, score in entry_conditions)
            # 롱/숏 바이어스 계산 (조건 투표 기반)
            votes = [score for _, score in entry_conditions]
            positive_votes = sum(1 for s in votes if s > 0)
            negative_votes = sum(1 for s in votes if s < 0)
            total_votes = positive_votes + negative_votes
            if total_votes > 0:
                long_bias_pct = round(positive_votes / total_votes * 100)
                short_bias_pct = 100 - long_bias_pct
            else:
                if market_strength > 0:
                    long_bias_pct, short_bias_pct = 60, 40
                elif market_strength < 0:
                    long_bias_pct, short_bias_pct = 40, 60
                else:
                    long_bias_pct, short_bias_pct = 50, 50
            
            # 파생 데이터 게이트(과열/저유동성 회피)
            try:
                fr = self.get_funding_rate(symbol)
                oi_usdt = self.get_open_interest_usdt(symbol)
                lsr = self.get_long_short_ratio(symbol)
                if fr == fr and abs(fr) > MAX_FUNDING_ABS:
                    return None
                if oi_usdt == oi_usdt and oi_usdt < MIN_OPEN_INTEREST_USDT:
                    return None
                if lsr == lsr and (lsr >= LONG_SHORT_EXTREME_HIGH or lsr <= LONG_SHORT_EXTREME_LOW):
                    return None
            except Exception:
                pass

            # 시그널 생성 여부 결정
            if abs(total_score) >= 2:  # 최소 2개 이상의 조건 충족
                signal_type = 'LONG' if total_score > 0 else 'SHORT'

                # 상위 타임프레임 컨펌 게이트(신호 방향 일치 요구)
                mtf_tailwind = self._mtf_confirm(symbol, signal_type)
                if not mtf_tailwind:
                    if MTF_CONFIRM_STRICT:
                        return None
                    else:
                        total_score -= 1
                
                # 진입가 계산
                current_price = float(current['close'])
                atr = float(current['atr'])
                
                swing_low, swing_high = self._recent_swing_levels(df, lookback=8)
                # 엔트리 개수 결정(1~3): 변동성/강도 기반
                atr_pct = (atr / current_price * 100) if current_price > 0 else 0
                num_entries = 3 if atr_pct >= 1.5 else (2 if atr_pct >= 0.5 else 1)
                # LONG/SHORT별 엔트리 가격 생성
                entries = [current_price]
                if signal_type == 'LONG' and num_entries >= 2:
                    entries.append(current_price - atr * 0.5)
                if signal_type == 'LONG' and num_entries >= 3:
                    entries.append(current_price - atr * 1.0)
                if signal_type == 'SHORT' and num_entries >= 2:
                    entries.append(current_price + atr * 0.5)
                if signal_type == 'SHORT' and num_entries >= 3:
                    entries.append(current_price + atr * 1.0)

                # 구조적 SL (ATR vs 최근 스윙) + 피벗(s1/r1) 근접 페널티
                # 간단 피벗 계산
                recent_high = float(df['high'].iloc[-1])
                recent_low = float(df['low'].iloc[-1])
                recent_close = float(df['close'].iloc[-1])
                pivot = (recent_high + recent_low + recent_close) / 3.0
                r1 = 2 * pivot - recent_low
                s1 = 2 * pivot - recent_high

                if signal_type == 'LONG':
                    atr_sl = current_price - atr * 2
                    base_sl = min(atr_sl, swing_low) if swing_low else atr_sl
                    if s1 and abs(current_price - s1) <= 0.5 * atr:
                        base_sl = min(base_sl, s1 - 0.2 * atr)
                    stop_loss = base_sl
                else:
                    atr_sl = current_price + atr * 2
                    base_sl = max(atr_sl, swing_high) if swing_high else atr_sl
                    if r1 and abs(current_price - r1) <= 0.5 * atr:
                        base_sl = max(base_sl, r1 + 0.2 * atr)
                    stop_loss = base_sl
                    
                # 수익 목표 계산 (ATR 기반 멀티플 우선, 없으면 퍼센트 백업)
                profit_targets = []
                primary_entry = entries[0]
                # HTF 순풍/역풍에 따라 TP 멀티플 동적 조정
                atr_multipliers = [0.6, 1.2, 1.8, 2.6] if mtf_tailwind else [0.3, 0.7, 1.0, 1.5]
                if not pd.isna(atr) and atr > 0:
                    for mul in atr_multipliers:
                        if signal_type == 'LONG':
                            target_price = primary_entry + atr * mul
                        else:
                            target_price = primary_entry - atr * mul
                        profit_targets.append(target_price)
                else:
                    for target in PROFIT_TARGETS:
                        if signal_type == 'LONG':
                            target_price = primary_entry * (1 + target / 100)
                        else:
                            target_price = primary_entry * (1 - target / 100)
                        profit_targets.append(target_price)
                    
                # 레버리지/포지션 사이징 계산 (리스크 기반 + 동적 보정)
                risk_amount_pct = abs(primary_entry - stop_loss) / primary_entry * 100
                calculated_leverage = int(RISK_PERCENTAGE / max(risk_amount_pct, 1e-6))
                optimal_leverage = max(1, min(MAX_LEVERAGE, calculated_leverage))
                position_size_usdt = None
                suggested_qty = None
                tick_size = None
                if globals().get('TRADING_EQUITY_USDT', 0) and TRADING_EQUITY_USDT > 0:
                    # per-trade 리스크 금액
                    risk_amount_usdt = TRADING_EQUITY_USDT * (RISK_PERCENTAGE / 100.0)
                    # 1코인 당 리스크(USDT)
                    per_unit_risk = abs(primary_entry - stop_loss)
                    qty = risk_amount_usdt / max(per_unit_risk, 1e-8)
                    # 레버리지 적용한 명목 포지션 크기(USDT)
                    notional = qty * primary_entry
                    position_size_usdt = min(notional * optimal_leverage, TRADING_EQUITY_USDT * MAX_LEVERAGE)

                    # 거래소 제약 반영(LOT_SIZE, MIN_NOTIONAL)
                    sf = self._get_symbol_filters(symbol)
                    if sf:
                        lot = sf.get('market_lot_size') or sf.get('lot_size') or {}
                        step_size = float(lot.get('stepSize', '0')) if lot.get('stepSize') else 0.0
                        min_qty = float(lot.get('minQty', '0')) if lot.get('minQty') else 0.0
                        min_notional_f = sf.get('min_notional', {})
                        min_notional = float(min_notional_f.get('notional', '0')) if min_notional_f.get('notional') else 0.0
                        price_f = sf.get('price_filter', {})
                        tick_size = float(price_f.get('tickSize', '0')) if price_f.get('tickSize') else None

                        if step_size > 0:
                            qty = self._round_step(qty, step_size, 'down')
                        if min_qty and qty < min_qty:
                            qty = 0.0
                        notional = qty * primary_entry
                        if min_notional and notional < min_notional:
                            qty = 0.0
                        suggested_qty = qty if qty > 0 else None

                # 동적 레버리지/사이징 보정: HTF 정렬 + 신뢰도/레짐 가중(최대 1.5x)
                dynamic_mul = 1.0
                if self._mtf_confirm(symbol, 'LONG' if total_score > 0 else 'SHORT'):
                    # 신뢰도에 비례 + 변동성/밴드폭이 적정이면 추가
                    conf_mul = min(DYNAMIC_SIZING_MULTIPLIER_MAX - 1.0, (abs(total_score) / 5.0))
                    current = df.iloc[-1]
                    atr_pct_now = float(current['atr'] / current['close'] * 100) if current['close'] else 0
                    bbw_now = float(current['bb_width']) if not pd.isna(current['bb_width']) else 0
                    regime_bonus = 0.1 if (ATR_PCT_MIN <= atr_pct_now <= ATR_PCT_MAX and BB_WIDTH_MIN <= bbw_now <= BB_WIDTH_MAX) else 0.0
                    dynamic_mul += conf_mul + regime_bonus
                optimal_leverage = int(min(MAX_LEVERAGE, max(1, optimal_leverage * dynamic_mul)))
                if position_size_usdt:
                    position_size_usdt = round(min(position_size_usdt * dynamic_mul, TRADING_EQUITY_USDT * MAX_LEVERAGE), 2)

                # 가격/손절/목표가 tickSize 라운딩
                sf = self._get_symbol_filters(symbol)
                tick_size = None
                if sf:
                    price_f = sf.get('price_filter', {})
                    tick_size = float(price_f.get('tickSize', '0')) if price_f.get('tickSize') else None
                if tick_size:
                    # 엔트리: 최근접 라운딩, TP/SL: 방향 보정 라운딩
                    entries = [self._round_price_to_tick(p, tick_size, 'nearest') for p in entries]
                    if signal_type == 'LONG':
                        profit_targets = [self._round_price_to_tick(p, tick_size, 'up') for p in profit_targets]
                        stop_loss = self._round_price_to_tick(stop_loss, tick_size, 'down')
                    else:
                        profit_targets = [self._round_price_to_tick(p, tick_size, 'down') for p in profit_targets]
                        stop_loss = self._round_price_to_tick(stop_loss, tick_size, 'up')

                # TP 운영 모드 결정: 고신뢰·강추세면 전량 익절(보수적=TP2), 아니면 분할 익절
                tp_mode = 'partial'
                tp_full_index = None
                try:
                    if current.get('adx') and current['adx'] >= (ADX_MIN + 5) and (min(95, abs(total_score) * 20) >= 85):
                        tp_mode = 'full'
                        tp_full_index = 1 if len(profit_targets) > 1 else 0
                except Exception:
                    pass
                
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'type': signal_type,
                    'entry_prices': [round(p, 6) for p in entries],
                    'profit_targets': [round(price, 6) for price in profit_targets],
                    'stop_loss': round(stop_loss, 6),
                    'leverage': optimal_leverage,
                    'position_size_usdt': round(position_size_usdt, 2) if position_size_usdt else None,
                    'partial_tp_weights': PARTIAL_TP_WEIGHTS,
                    'suggested_qty': suggested_qty,
                    'tick_size': tick_size,
                    'long_bias_pct': long_bias_pct,
                    'short_bias_pct': short_bias_pct,
                    'confidence': min(95, abs(total_score) * 20),
                    'reasons': [reason for reason, _ in entry_conditions],
                    'market_strength': market_strength,
                    'timestamp': current['timestamp'],
                    'tp_mode': tp_mode,
                    'tp_full_index': tp_full_index
                }
                
            return None
            
        except Exception as e:
            logger.error(f"시그널 생성 오류: {e}")
            return None
            
    def get_top_symbols(self, limit=50):
        """선물 거래량 기준 상위 심볼 가져오기 (USDT PERPETUAL)"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                if not symbol.endswith('USDT'):
                    continue
                if symbol_info['status'] != 'TRADING':
                    continue
                if symbol_info.get('contractType') not in (None, 'PERPETUAL'):
                    continue
                # 레버리지/인버스/스테이블-스테이블 제외
                excluded_exact = {
                    'USDCUSDT','BUSDUSDT','FDUSDUSDT','TUSDUSDT','USDPUSDT','DAIUSDT','EURUSDT','FDUSDUSDT'
                }
                if symbol in excluded_exact:
                    continue
                base = symbol.replace('USDT','')
                leveraged_suffixes = ('UP','DOWN','BULL','BEAR','3L','3S','4L','4S','5L','5S','2L','2S')
                if any(base.endswith(suf) for suf in leveraged_suffixes):
                    continue
                symbols.append(symbol)
                    
            # 24시간 통계 가져오기 (선물)
            tickers = self.client.futures_ticker()
            usdt_tickers = [t for t in tickers if t['symbol'] in symbols]
            
            # 유동성/스프레드 필터
            filtered = []
            try:
                ob_list = self.client.futures_orderbook_ticker()
                ob_map = {o['symbol']: o for o in ob_list}
            except Exception:
                ob_map = {}
            for t in usdt_tickers:
                try:
                    quote_volume = float(t.get('quoteVolume', 0))
                    if quote_volume < MIN_VOLUME_24H:
                        continue
                    ob = ob_map.get(t['symbol'])
                    if not ob:
                        continue
                    bid = float(ob['bidPrice']); ask = float(ob['askPrice'])
                    spread_pct = (ask - bid) / ask * 100 if ask > 0 else 100
                    if spread_pct > MAX_SPREAD:
                        continue
                    # 파생 데이터 필터(펀딩/OI/롱숏비)
                    fr = self.get_funding_rate(t['symbol'])
                    oi_usdt = self.get_open_interest_usdt(t['symbol'])
                    lsr = self.get_long_short_ratio(t['symbol'])
                    if fr == fr and abs(fr) > MAX_FUNDING_ABS:
                        continue
                    if oi_usdt == oi_usdt and oi_usdt < MIN_OPEN_INTEREST_USDT:
                        continue
                    if lsr == lsr and (lsr >= LONG_SHORT_EXTREME_HIGH or lsr <= LONG_SHORT_EXTREME_LOW):
                        continue
                    filtered.append((t['symbol'], quote_volume))
                except Exception:
                    continue
            
            # 거래량 기준 정렬
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            return [sym for sym, _ in filtered[:limit]]
            
        except Exception as e:
            logger.error(f"상위 심볼 가져오기 오류: {e}")
            return []

    def get_last_price(self, symbol: str) -> float:
        try:
            p = self.client.futures_symbol_ticker(symbol=symbol)
            return float(p['price'])
        except Exception:
            return float('nan')
