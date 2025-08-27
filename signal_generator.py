import pandas as pd
import numpy as np
from market_analyzer import MarketAnalyzer
import logging
from config import *

logger = logging.getLogger(__name__)

class AdvancedSignalGenerator:
    def __init__(self):
        self.analyzer = MarketAnalyzer()
        
    def calculate_volume_profile(self, df):
        """볼륨 프로파일 분석"""
        try:
            if df is None or len(df) < 20:
                return None
                
            # 볼륨 가중 평균가
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # 볼륨 분포 분석
            volume_quantiles = df['volume'].quantile([0.25, 0.5, 0.75])
            
            # 고볼륨 구간 식별
            high_volume_periods = df[df['volume'] > volume_quantiles[0.75]]
            
            return {
                'vwap': df['vwap'].iloc[-1],
                'volume_trend': 'increasing' if df['volume'].iloc[-1] > df['volume'].iloc[-5:].mean() else 'decreasing',
                'high_volume_zones': high_volume_periods[['timestamp', 'close', 'volume']].to_dict('records')
            }
        except Exception as e:
            logger.error(f"볼륨 프로파일 분석 오류: {e}")
            return None
            
    def calculate_support_resistance(self, df):
        """지지/저항선 계산"""
        try:
            if df is None or len(df) < 50:
                return None
                
            # 피벗 포인트 계산
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            # 과거 고점/저점 분석
            recent_highs = df['high'].rolling(window=20).max()
            recent_lows = df['low'].rolling(window=20).min()
            
            # 지지/저항 레벨 식별
            resistance_levels = []
            support_levels = []
            
            for i in range(len(df) - 20, len(df)):
                if df['high'].iloc[i] == recent_highs.iloc[i]:
                    resistance_levels.append(df['high'].iloc[i])
                if df['low'].iloc[i] == recent_lows.iloc[i]:
                    support_levels.append(df['low'].iloc[i])
                    
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2,
                's1': s1, 's2': s2,
                'resistance_levels': sorted(list(set(resistance_levels)))[-3:],
                'support_levels': sorted(list(set(support_levels)))[:3]
            }
        except Exception as e:
            logger.error(f"지지/저항선 계산 오류: {e}")
            return None
            
    def calculate_momentum_divergence(self, df):
        """모멘텀 다이버전스 분석"""
        try:
            if df is None or len(df) < 30:
                return None
                
            # RSI 다이버전스
            rsi = df['rsi'].iloc[-10:]
            price = df['close'].iloc[-10:]
            
            # 가격과 RSI의 고점/저점 비교
            price_highs = []
            rsi_highs = []
            price_lows = []
            rsi_lows = []
            
            for i in range(1, len(price) - 1):
                if price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1]:
                    price_highs.append((i, price.iloc[i]))
                if rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i] > rsi.iloc[i+1]:
                    rsi_highs.append((i, rsi.iloc[i]))
                if price.iloc[i] < price.iloc[i-1] and price.iloc[i] < price.iloc[i+1]:
                    price_lows.append((i, price.iloc[i]))
                if rsi.iloc[i] < rsi.iloc[i-1] and rsi.iloc[i] < rsi.iloc[i+1]:
                    rsi_lows.append((i, rsi.iloc[i]))
                    
            # 다이버전스 확인
            bearish_divergence = False
            bullish_divergence = False
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                    bearish_divergence = True
                    
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                    bullish_divergence = True
                    
            return {
                'bearish_divergence': bearish_divergence,
                'bullish_divergence': bullish_divergence
            }
        except Exception as e:
            logger.error(f"모멘텀 다이버전스 분석 오류: {e}")
            return None
            
    def calculate_risk_reward_ratio(self, entry_price, stop_loss, profit_targets):
        """리스크 대비 수익 비율 계산"""
        try:
            risk = abs(entry_price - stop_loss)
            reward_ratios = []
            
            for target in profit_targets:
                reward = abs(target - entry_price)
                ratio = reward / risk if risk > 0 else 0
                reward_ratios.append(ratio)
                
            avg_reward_ratio = np.mean(reward_ratios)
            
            return {
                'risk': risk,
                'reward_ratios': reward_ratios,
                'avg_reward_ratio': avg_reward_ratio,
                'risk_level': 'low' if avg_reward_ratio >= 3 else 'medium' if avg_reward_ratio >= 2 else 'high'
            }
        except Exception as e:
            logger.error(f"리스크 대비 수익 비율 계산 오류: {e}")
            return None
            
    def generate_advanced_signal(self, symbol, df):
        """고급 시그널 생성"""
        try:
            if df is None or len(df) < 50:
                return None
                
            # 기본 시그널 생성
            basic_signal = self.analyzer.generate_signal(symbol, df)
            
            if not basic_signal:
                return None
                
            # 추가 분석 수행
            volume_profile = self.calculate_volume_profile(df)
            support_resistance = self.calculate_support_resistance(df)
            momentum_divergence = self.calculate_momentum_divergence(df)
            
            # 신뢰도 점수 계산
            confidence_score = basic_signal['confidence']
            
            # 볼륨 프로파일 점수
            if volume_profile:
                if volume_profile['volume_trend'] == 'increasing':
                    confidence_score += 10
                if len(volume_profile['high_volume_zones']) > 0:
                    confidence_score += 5
                    
            # 지지/저항선 점수
            if support_resistance:
                current_price = basic_signal['entry_prices'][0]
                
                # 지지선 근처에서 롱
                if basic_signal['type'] == 'LONG':
                    for support in support_resistance['support_levels']:
                        if abs(current_price - support) / support < 0.02:  # 2% 이내
                            confidence_score += 15
                            break
                            
                # 저항선 근처에서 숏
                elif basic_signal['type'] == 'SHORT':
                    for resistance in support_resistance['resistance_levels']:
                        if abs(current_price - resistance) / resistance < 0.02:  # 2% 이내
                            confidence_score += 15
                            break
                            
            # 다이버전스 점수
            if momentum_divergence:
                if basic_signal['type'] == 'LONG' and momentum_divergence['bullish_divergence']:
                    confidence_score += 20
                elif basic_signal['type'] == 'SHORT' and momentum_divergence['bearish_divergence']:
                    confidence_score += 20
                    
            # 리스크 대비 수익 비율 점수
            risk_reward = self.calculate_risk_reward_ratio(
                basic_signal['entry_prices'][0],
                basic_signal['stop_loss'],
                basic_signal['profit_targets']
            )
            
            if risk_reward:
                if risk_reward['risk_level'] == 'low':
                    confidence_score += 15
                elif risk_reward['risk_level'] == 'medium':
                    confidence_score += 10
                    
            # R:R 하한 미달 시 폐기
            if risk_reward and risk_reward.get('avg_reward_ratio') is not None:
                if float(risk_reward['avg_reward_ratio']) < float(MIN_RR_AVG):
                    return None

            # 최종 신뢰도 조정
            final_confidence = min(95, confidence_score)
            
            # 시그널 품질 평가
            signal_quality = 'Premium' if final_confidence >= 85 else 'High' if final_confidence >= 70 else 'Medium'
            
            # 고급 시그널 정보 추가
            advanced_signal = basic_signal.copy()
            advanced_signal['confidence'] = final_confidence
            advanced_signal['quality'] = signal_quality
            advanced_signal['volume_profile'] = volume_profile
            advanced_signal['support_resistance'] = support_resistance
            advanced_signal['momentum_divergence'] = momentum_divergence
            advanced_signal['risk_reward'] = risk_reward
            
            return advanced_signal
            
        except Exception as e:
            logger.error(f"고급 시그널 생성 오류: {e}")
            return None
            
    def filter_signals_by_quality(self, signals, min_confidence=70):
        """품질 기준으로 시그널 필터링"""
        try:
            filtered_signals = []
            
            for signal in signals:
                if signal and signal['confidence'] >= min_confidence:
                    # 추가 필터링 조건
                    if signal['quality'] in ['Premium', 'High']:
                        filtered_signals.append(signal)
                        
            # 신뢰도 기준으로 정렬
            filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"시그널 필터링 오류: {e}")
            return []
            
    def get_market_sentiment(self, df):
        """시장 심리 분석"""
        try:
            if df is None or len(df) < 20:
                return 'neutral'
                
            # 여러 지표를 종합하여 시장 심리 판단
            rsi = df['rsi'].iloc[-1]
            macd_hist = df['macd_histogram'].iloc[-1]
            bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            
            # 점수 계산
            score = 0
            
            if rsi < 30:
                score += 2  # 과매도
            elif rsi > 70:
                score -= 2  # 과매수
                
            if macd_hist > 0:
                score += 1  # 상승 모멘텀
            else:
                score -= 1  # 하락 모멘텀
                
            if bb_position < 0.2:
                score += 1  # 볼린저 밴드 하단
            elif bb_position > 0.8:
                score -= 1  # 볼린저 밴드 상단
                
            # 심리 판단
            if score >= 2:
                return 'bullish'
            elif score <= -2:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"시장 심리 분석 오류: {e}")
            return 'neutral'
