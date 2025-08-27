import asyncio
import logging
import json
import pandas as pd
from telegram import Bot
from telegram.error import TelegramError
from datetime import datetime
import schedule
import time
from market_analyzer import MarketAnalyzer
from config import *
from backtester import backtest as bt_simple
from signal_generator import AdvancedSignalGenerator
from persistence import init_db, save_signal, update_signal_event, load_open_signals
import websocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.analyzer = MarketAnalyzer()
        self.generator = AdvancedSignalGenerator()
        self.sent_signals = set()  # 중복 전송 방지
        self.cooldowns = {}  # {(symbol, interval): last_timestamp}
        self.open_signals = {}  # {(symbol, interval): signal_dict}
        self._last_update_id = None  # 텔레그램 명령 폴링 위치
        try:
            init_db()
            for s in load_open_signals():
                key = (s['symbol'], s['interval'])
                self.open_signals[key] = s
        except Exception:
            pass
        
    async def send_signal(self, signal):
        """시그널을 텔레그램으로 전송"""
        try:
            key = (signal['symbol'], signal.get('interval', '30m'))
            if key in self.sent_signals:
                return False
                
            # 시그널 메시지 생성
            message = self.format_signal_message(signal)
            
            # 텔레그램 전송
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            # 전송 완료 표시
            self.sent_signals.add(key)
            logger.info(f"시그널 전송 완료: {signal['symbol']}")
            
            return True
            
        except TelegramError as e:
            logger.error(f"텔레그램 전송 오류: {e}")
            return False
        except Exception as e:
            logger.error(f"시그널 전송 오류: {e}")
            return False
            
    def _cooldown_ok(self, symbol, interval):
        key = (symbol, interval)
        if key not in self.cooldowns:
            return True
        last_ts = self.cooldowns[key]
        # interval을 분으로 환산하여 COOLDOWN_CANDLES만큼 경과 필요
        try:
            mult = 1
            if interval.endswith('m'):
                mult = int(interval[:-1])
            elif interval.endswith('h'):
                mult = int(interval[:-1]) * 60
            required_minutes = mult * COOLDOWN_CANDLES
            from datetime import datetime, timedelta
            return datetime.utcnow() >= last_ts + timedelta(minutes=required_minutes)
        except Exception:
            return True

    def _set_cooldown(self, symbol, interval):
        from datetime import datetime
        self.cooldowns[(symbol, interval)] = datetime.utcnow()

    def format_signal_message(self, signal):
        """시그널 메시지 포맷팅"""
        signal_type_emoji = "🚀" if signal['type'] == 'LONG' else "📉"
        confidence_emoji = "🔥" if signal['confidence'] >= 80 else "⚡" if signal['confidence'] >= 60 else "💡"
        
        # 타임프레임 표기 정규화 (요청 포맷)
        time_period = "Mid-Term"
            
        # Entry 라인 동적 생성(1~3)
        entry_lines = []
        for idx, ep in enumerate(signal['entry_prices'], start=1):
            entry_lines.append(f"{idx}) {ep}")
        entry_block = "\n".join(entry_lines)

        # TP 표시: 전량/분할 모드
        tp_block = ""
        if signal.get('tp_mode') == 'full' and signal.get('tp_full_index') is not None:
            i = signal['tp_full_index']
            tp_block = f"<b>Take-Profit</b>\n전량: {signal['profit_targets'][i]}"
        else:
            tp_lines = []
            for i, tp in enumerate(signal['profit_targets'], start=1):
                tp_lines.append(f"{i}) {tp}")
            tp_block = "<b>Take-Profit</b>\n" + "\n".join(tp_lines)

        rr = None
        try:
            if signal.get('risk_reward') and signal['risk_reward'].get('avg_reward_ratio') is not None:
                rr = float(signal['risk_reward']['avg_reward_ratio'])
        except Exception:
            rr = None

        message = f"""
{signal_type_emoji} <b>#{signal['symbol']} | {signal['type']} | {signal.get('interval','30m')}</b>

<b>Entry</b>
{entry_block}

{tp_block}

<b>Stop-Loss</b>
{signal['stop_loss']}

<b>Leverage</b>
{signal['leverage']}x [Isolated]

<b>Quality</b>
{signal.get('quality','-')} | 신뢰도 {signal.get('confidence','-')}%{f" | R:R {rr:.2f}" if rr is not None else ''}
        """.strip()
        
        return message
        
    async def scan_and_send_signals(self):
        """시장 스캔 및 시그널 전송"""
        try:
            logger.info("시장 스캔 시작...")
            
            # 상위 거래량 심볼 가져오기
            symbols = self.analyzer.get_top_symbols(limit=30)
            
            if not symbols:
                logger.warning("심볼을 가져올 수 없습니다.")
                return
                
            signals_sent = 0
            logger.info(f"대상 심볼 {len(symbols)}개 | 타임프레임 {','.join(SCAN_INTERVALS)}")

            async def handle_pair(symbol: str, interval: str):
                try:
                    df = self.analyzer.get_klines(symbol, interval=interval, limit=240, exclude_last_open=True)
                    if df is None:
                        logger.debug(f"{symbol} {interval}: 데이터 없음")
                        return False
                    df = self.analyzer.calculate_technical_indicators(df)
                    if df is None:
                        logger.debug(f"{symbol} {interval}: 지표 계산 실패")
                        return False
                    # 실시간 완화 모드: 생성은 완화, 전송 기준은 강화
                    if REALTIME_RELAXED:
                        base = self.analyzer.generate_signal(
                            symbol, df, interval=interval,
                            relaxed=RELAXED_IGNORE_MTF,
                            ignore_derivatives=RELAXED_IGNORE_DERIVATIVES
                        )
                        adv = None
                        if base:
                            adv = self.generator.generate_advanced_signal(symbol, df)
                            if adv:
                                # 전송 임계 상향
                                rr = adv.get('risk_reward')
                                rr_ok = (rr and rr.get('avg_reward_ratio') is not None and float(rr['avg_reward_ratio']) >= RELAXED_MIN_RR_AVG)
                                conf_ok = adv.get('confidence', 0) >= max(QUALITY_MIN_CONFIDENCE, RELAXED_MIN_CONFIDENCE)
                                if not (rr_ok and conf_ok):
                                    adv = None
                    else:
                        adv = self.generator.generate_advanced_signal(symbol, df)
                        if adv:
                            rr = adv.get('risk_reward')
                            rr_ok = (rr and rr.get('avg_reward_ratio') is not None and float(rr['avg_reward_ratio']) >= MIN_RR_AVG)
                            conf_ok = adv.get('confidence', 0) >= QUALITY_MIN_CONFIDENCE
                            if not (rr_ok and conf_ok):
                                adv = None

                    if adv and self._cooldown_ok(symbol, interval):
                        adv['interval'] = interval
                        ok = await self.send_signal(adv)
                        if ok:
                            self._set_cooldown(symbol, interval)
                            key = (symbol, interval)
                            self.open_signals[key] = adv
                            try:
                                save_signal(adv)
                            except Exception:
                                pass
                            return True
                    # 세부 원인 로깅
                    try:
                        base = self.analyzer.generate_signal(
                            symbol, df, interval=interval,
                            relaxed=REALTIME_RELAXED and RELAXED_IGNORE_MTF,
                            ignore_derivatives=REALTIME_RELAXED and RELAXED_IGNORE_DERIVATIVES
                        )
                        if not base:
                            logger.info(f"필터 통과 실패: {symbol} {interval} | 베이식 신호 없음(MTF/레짐/펀딩/OI/롱숏비/조건 미충족)")
                        else:
                            rr = self.generator.calculate_risk_reward_ratio(base['entry_prices'][0], base['stop_loss'], base['profit_targets'])
                            rr_min = RELAXED_MIN_RR_AVG if REALTIME_RELAXED else MIN_RR_AVG
                            if rr and rr.get('avg_reward_ratio') is not None and float(rr['avg_reward_ratio']) < float(rr_min):
                                logger.info(f"필터 통과 실패: {symbol} {interval} | R:R {rr['avg_reward_ratio']:.2f} < {MIN_RR_AVG}")
                            elif base.get('confidence', 0) < (max(QUALITY_MIN_CONFIDENCE, RELAXED_MIN_CONFIDENCE) if REALTIME_RELAXED else QUALITY_MIN_CONFIDENCE):
                                logger.info(f"품질 미달: {symbol} {interval} | 신뢰도 {base.get('confidence', 0)}% < {QUALITY_MIN_CONFIDENCE}%")
                            else:
                                logger.info(f"고급 필터 미통과: {symbol} {interval}")
                    except Exception:
                        pass
                    return False
                except Exception as e:
                    logger.error(f"{symbol} {interval} 처리 오류: {e}")
                    return False

            tasks = [handle_pair(sym, iv) for sym in symbols for iv in SCAN_INTERVALS]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            signals_sent = sum(1 for r in results if r is True)

            logger.info(f"스캔 완료. {signals_sent}개 시그널 전송됨")
            
        except Exception as e:
            logger.error(f"시장 스캔 오류: {e}")

    async def poll_commands(self):
        """텔레그램 /명령어 폴링(짧게 1회)"""
        try:
            params = {}
            if self._last_update_id is not None:
                params['offset'] = self._last_update_id + 1
            updates = await self.bot.get_updates(timeout=5, **params)
            for u in updates or []:
                try:
                    self._last_update_id = u.update_id
                    msg = getattr(u, 'message', None)
                    if not msg or not getattr(msg, 'text', ''):
                        continue
                    text = msg.text.strip()
                    if not text.startswith('/'):
                        continue
                    # 허용된 채팅에서만 처리
                    if str(getattr(msg.chat, 'id', '')) != str(TELEGRAM_CHAT_ID):
                        continue
                    await self.handle_command(text)
                except Exception:
                    continue
        except Exception:
            pass

    async def process_symbol_interval(self, symbol: str, interval: str):
        """단일 심볼/타임프레임 처리 (WS 트리거용)"""
        try:
            df = self.analyzer.get_klines(symbol, interval=interval, limit=240, exclude_last_open=True)
            if df is None:
                return
            df = self.analyzer.calculate_technical_indicators(df)
            if df is None:
                return
            adv = self.generator.generate_advanced_signal(symbol, df)
            if adv and adv.get('confidence', 0) >= QUALITY_MIN_CONFIDENCE and self._cooldown_ok(symbol, interval):
                adv['interval'] = interval
                success = await self.send_signal(adv)
                if success:
                    self._set_cooldown(symbol, interval)
                    self.open_signals[(symbol, interval)] = adv
                    try:
                        save_signal(adv)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"WS 처리 오류 {symbol} {interval}: {e}")

    async def monitor_open_signals(self):
        """TP/SL 체결 모니터링 및 수익률 알림"""
        try:
            if not self.open_signals:
                return
            for key, sig in list(self.open_signals.items()):
                symbol, interval = key
                try:
                    last_price = self.analyzer.get_last_price(symbol)
                    if not last_price or last_price != last_price:
                        continue
                    entry = sig['entry_prices'][0]
                    sl = sig['stop_loss']
                    tps = sig['profit_targets']
                    direction = sig['type']
                    # 가상 포지션 상태
                    st = sig.setdefault('state', {'be_moved': False, 'trail_on': False, 'peak': entry})
                    # 체결 조건
                    hit_tp_idx = None
                    for i, tp in enumerate(tps):
                        if (direction == 'LONG' and last_price >= tp) or (direction == 'SHORT' and last_price <= tp):
                            hit_tp_idx = i
                            break
                    hit_sl = (direction == 'LONG' and last_price <= sl) or (direction == 'SHORT' and last_price >= sl)
                    # 알림
                    if hit_tp_idx is not None:
                        pnl_pct = (last_price - entry) / entry * 100 if direction == 'LONG' else (entry - last_price) / entry * 100
                        msg = f"✅ TP{hit_tp_idx+1} 체결 | #{symbol} {direction} | {interval}\n수익률: {pnl_pct:.2f}%"
                        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                        try:
                            update_signal_event(symbol, interval, f"TP{hit_tp_idx+1}", last_price, info={'pnl_pct': pnl_pct})
                        except Exception:
                            pass
                        # 분할익절 비중 및 누적 PnL
                        weights = sig.get('partial_tp_weights') or [0.25,0.25,0.25,0.25]
                        st.setdefault('filled_weight', 0.0)
                        fill_w = weights[hit_tp_idx] if hit_tp_idx < len(weights) else 0.0
                        st['filled_weight'] = min(1.0, st['filled_weight'] + fill_w)
                        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"📤 부분 익절 {fill_w*100:.0f}% | 누적 {st['filled_weight']*100:.0f}%")

                        if hit_tp_idx == 0 and not st['be_moved']:
                            st['be_moved'] = True
                            sig['stop_loss'] = entry  # BE 이동
                            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🔁 SL → BE 이동 | #{symbol}")
                        if hit_tp_idx >= TRAILING_ACTIVATE_AFTER_TP_INDEX and not st['trail_on']:
                            st['trail_on'] = True
                            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"📈 트레일링 활성화 | #{symbol}")
                        # 모든 비중 체결 시 종료
                        if st['filled_weight'] >= 0.999:
                            self.open_signals.pop(key, None)
                            continue
                    elif hit_sl:
                        pnl_pct = (last_price - entry) / entry * 100 if direction == 'LONG' else (entry - last_price) / entry * 100
                        msg = f"❌ SL 체결 | #{symbol} {direction} | {interval}\n수익률: {pnl_pct:.2f}%"
                        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                        try:
                            update_signal_event(symbol, interval, "SL", last_price, info={'pnl_pct': pnl_pct}, close=True)
                        except Exception:
                            pass
                        self.open_signals.pop(key, None)
                    else:
                        # 트레일링 동작 (ATR 기반)
                        if st['trail_on'] and TRAILING_ATR_MULTIPLIER > 0:
                            # 최신 ATR 조회
                            tdf = self.analyzer.get_klines(symbol, interval=interval, limit=60, exclude_last_open=True)
                            tdf = self.analyzer.calculate_technical_indicators(tdf)
                            if tdf is not None and len(tdf) >= 20 and not pd.isna(tdf['atr'].iloc[-1]):
                                atr_now = float(tdf['atr'].iloc[-1])
                                sf = self.analyzer._get_symbol_filters(symbol)
                                tick_size = None
                                if sf:
                                    price_f = sf.get('price_filter', {})
                                    tick_size = float(price_f.get('tickSize', '0')) if price_f.get('tickSize') else None
                                old_sl = sig['stop_loss']
                                if direction == 'LONG':
                                    st['peak'] = max(st['peak'], last_price)
                                    candidate = st['peak'] - atr_now * TRAILING_ATR_MULTIPLIER
                                    # 라운딩: LONG SL down
                                    if tick_size:
                                        candidate = self.analyzer._round_price_to_tick(candidate, tick_size, 'down')
                                    new_sl = max(old_sl, candidate)
                                else:
                                    st['peak'] = min(st['peak'], last_price)
                                    candidate = st['peak'] + atr_now * TRAILING_ATR_MULTIPLIER
                                    # 라운딩: SHORT SL up
                                    if tick_size:
                                        candidate = self.analyzer._round_price_to_tick(candidate, tick_size, 'up')
                                    new_sl = min(old_sl, candidate)
                                if new_sl != old_sl:
                                    sig['stop_loss'] = new_sl
                                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🔧 트레일링 SL 조정 → {new_sl} | #{symbol}")
                                    try:
                                        update_signal_event(symbol, interval, "TRAIL_SL", new_sl, info=None)
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.error(f"모니터링 오류 {symbol}: {e}")
        except Exception as e:
            logger.error(f"모니터링 루프 오류: {e}")

    def run_realtime(self):
        """실시간 스캔 + TP/SL 모니터링 루프"""
        logger.info("실시간 모니터링 시작...")
        try:
            import asyncio
            async def _loop():
                while True:
                    await self.scan_and_send_signals()
                    await self.monitor_open_signals()
                    await self.poll_commands()
                    await asyncio.sleep(REALTIME_SCAN_SECONDS)
            asyncio.run(_loop())
        except KeyboardInterrupt:
            logger.info("실시간 모니터링 중단")
        except Exception as e:
            logger.error(f"실시간 모니터링 오류: {e}")
            
    def run_scheduler(self):
        """스케줄러 실행"""
        # 30분마다 시그널 스캔
        schedule.every(SIGNAL_INTERVAL).minutes.do(
            lambda: asyncio.run(self.scan_and_send_signals())
        )
        
        # 매일 자정에 전송된 시그널 기록 초기화
        schedule.every().day.at("00:00").do(
            lambda: self.sent_signals.clear()
        )
        
        logger.info("스케줄러 시작됨")
        
        while True:
            try:
                schedule.run_pending()
                # 명령 폴링도 함께 수행
                try:
                    asyncio.run(self.poll_commands())
                except Exception:
                    pass
                time.sleep(60)  # 1분마다 체크
            except KeyboardInterrupt:
                logger.info("봇 종료됨")
                break
            except Exception as e:
                logger.error(f"스케줄러 오류: {e}")
                time.sleep(60)

    def run_ws(self):
        """바이낸스 선물 Kline WebSocket 구독(캔들 마감 시 신호 평가)"""
        logger.info("WebSocket 실시간 모드 시작...")
        try:
            symbols = self.analyzer.get_top_symbols(limit=30)
            if not symbols:
                logger.warning("WS: 구독할 심볼이 없습니다.")
                return

            streams = []
            for s in symbols:
                for iv in SCAN_INTERVALS:
                    streams.append(f"{s.lower()}@kline_{iv}")

            def on_open(ws):
                sub = {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 1
                }
                ws.send(json.dumps(sub))
                logger.info(f"WS 구독 {len(streams)}개 스트림")

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    k = None
                    if 'data' in data and 'k' in data['data']:
                        k = data['data']['k']
                    elif 'k' in data:
                        k = data['k']
                    if not k:
                        return
                    if k.get('x') is True:  # 캔들 마감
                        sym = k.get('s')
                        iv = k.get('i')
                        if sym and iv:
                            asyncio.run(self.process_symbol_interval(sym, iv))
                except Exception as e:
                    logger.error(f"WS 메시지 오류: {e}")

            def on_error(ws, error):
                logger.error(f"WS 오류: {error}")

            def on_close(ws, code, msg):
                logger.info(f"WS 종료: {code} {msg}")

            while True:
                try:
                    ws = websocket.WebSocketApp(
                        "wss://fstream.binance.com/ws",
                        on_open=on_open,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                    )
                    ws.run_forever(ping_interval=20, ping_timeout=10)
                    time.sleep(5)  # 재연결 대기
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"WS 연결 오류: {e}")
                    time.sleep(5)
        except Exception as e:
            logger.error(f"WS 모드 오류: {e}")
    async def test_message(self):
        """테스트 메시지 전송"""
        try:
            test_message = """
🧪 <b>시그널 봇 테스트</b>

봇이 정상적으로 작동하고 있습니다!

✅ 바이낸스 API 연결
✅ 기술적 분석 엔진
✅ 텔레그램 봇 연결

시그널 스캔을 시작합니다...
            """.strip()
            
            msg = await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=test_message,
                parse_mode='HTML'
            )
            # 명령 수신 시 처리(간단 폴링): 최신 메시지에 커맨드가 있으면 처리
            try:
                updates = await self.bot.get_updates(limit=1)
                if updates:
                    txt = updates[-1].message.text or ''
                    if txt.startswith('/'):
                        await self.handle_command(txt)
            except Exception:
                pass
            
            logger.info("테스트 메시지 전송 완료")
            
        except Exception as e:
            logger.error(f"테스트 메시지 전송 오류: {e}")
            
    def start(self):
        """봇 시작"""
        logger.info("시그널 봇 시작...")
        
        # 테스트 메시지 전송
        asyncio.run(self.test_message())
        
        # 스케줄러 시작
        self.run_scheduler()

    async def handle_command(self, text: str):
        """텔레그램 명령 처리 (/backtest 등)"""
        try:
            parts = text.strip().split()
            cmd = parts[0].lower()
            if cmd == '/backtest':
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                interval = parts[2] if len(parts) > 2 else '30m'
                lookback = int(parts[3]) if len(parts) > 3 else 400
                report = bt_simple(symbol, interval, lookback)
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🧪 백테스트 결과\n{report}")
            elif cmd == '/status':
                open_cnt = len(self.open_signals)
                msg = (
                    f"📡 상태\n"
                    f"스캔 타임프레임: {', '.join(SCAN_INTERVALS)}\n"
                    f"최소 신뢰도: {QUALITY_MIN_CONFIDENCE}% | 최소 R:R: {MIN_RR_AVG}\n"
                    f"대상 심볼 수(최근): 30\n"
                    f"오픈 시그널: {open_cnt}개"
                )
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            elif cmd == '/top':
                limit = int(parts[1]) if len(parts) > 1 else 15
                syms = self.analyzer.get_top_symbols(limit=limit)
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"💠 상위 심볼 {len(syms)}개\n" + ', '.join(syms))
            elif cmd == '/open':
                if not self.open_signals:
                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="열린 시그널 없음")
                else:
                    lines = []
                    for (sym, iv), s in list(self.open_signals.items()):
                        lines.append(f"#{sym} {s.get('type','-')} {iv} | EP {s['entry_prices'][0]} SL {s['stop_loss']}")
                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="📜 오픈 시그널\n" + "\n".join(lines))
            elif cmd in ('/metrics','/why','/debug'):
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                interval = parts[2] if len(parts) > 2 else '30m'
                df = self.analyzer.get_klines(symbol, interval=interval, limit=240, exclude_last_open=True)
                if df is None:
                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"데이터 없음: {symbol} {interval}")
                    return
                df = self.analyzer.calculate_technical_indicators(df)
                if df is None:
                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"지표 계산 실패: {symbol} {interval}")
                    return
                cur = df.iloc[-1]
                # 파생 데이터
                fr = self.analyzer.get_funding_rate(symbol)
                oi = self.analyzer.get_open_interest_usdt(symbol)
                lsr = self.analyzer.get_long_short_ratio(symbol)
                # 레짐 체크
                try:
                    atr_pct = float(cur['atr'] / cur['close'] * 100) if cur['close'] else float('nan')
                    bbw = float(cur['bb_width']) if pd.notna(cur['bb_width']) else float('nan')
                except Exception:
                    atr_pct, bbw = float('nan'), float('nan')
                regime_ok = (
                    (atr_pct == atr_pct and ATR_PCT_MIN <= atr_pct <= ATR_PCT_MAX) and
                    (bbw == bbw and BB_WIDTH_MIN <= bbw <= BB_WIDTH_MAX)
                )
                # MTF 컨펌(방향별)
                mtf_long = self.analyzer._mtf_confirm(symbol, 'LONG')
                mtf_short = self.analyzer._mtf_confirm(symbol, 'SHORT')
                # 시그널 프리뷰
                base = self.analyzer.generate_signal(
                    symbol, df, interval=interval,
                    relaxed=REALTIME_RELAXED and RELAXED_IGNORE_MTF,
                    ignore_derivatives=REALTIME_RELAXED and RELAXED_IGNORE_DERIVATIVES
                )
                preview = "없음"
                rr_txt = ""
                if base:
                    rr = self.generator.calculate_risk_reward_ratio(base['entry_prices'][0], base['stop_loss'], base['profit_targets'])
                    if rr and rr.get('avg_reward_ratio') is not None:
                        rr_txt = f" | R:R {float(rr['avg_reward_ratio']):.2f}"
                    preview = f"{base['type']} | 신뢰도 {base.get('confidence','-')}%{rr_txt}"
                # 메시지
                msg = (
                    f"🔎 {symbol} {interval} 지표\n"
                    f"가격 {float(cur['close']):.6f} | RSI {float(cur['rsi']):.2f} | ADX {float(cur['adx']):.2f}\n"
                    f"MACD {float(cur['macd']):.4f} / {float(cur['macd_signal']):.4f} (hist {float(cur['macd_histogram']):.4f})\n"
                    f"BB폭 {bbw:.4%} | ATR {float(cur['atr']):.6f} ({atr_pct:.2f}%)\n"
                    f"펀딩 {fr:.4f}% | OI ${oi:,.0f} | 롱숏비 {lsr:.2f}\n"
                    f"레짐 {'OK' if regime_ok else 'FAIL'} | MTF L:{'OK' if mtf_long else 'NO'} S:{'OK' if mtf_short else 'NO'}\n"
                    f"시그널 프리뷰: {preview}"
                )
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            logger.error(f"명령 처리 오류: {e}")

if __name__ == "__main__":
    bot = SignalBot()
    bot.start()
