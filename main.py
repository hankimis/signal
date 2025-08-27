#!/usr/bin/env python3
"""
완벽한 텔레그램 코인 시그널 봇
승률 높은 자동 시그널 생성 및 전송
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from telegram_bot import SignalBot
from signal_generator import AdvancedSignalGenerator
from market_analyzer import MarketAnalyzer
from config import *

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PerfectSignalBot:
    def __init__(self):
        self.telegram_bot = SignalBot()
        self.advanced_generator = AdvancedSignalGenerator()
        self.market_analyzer = MarketAnalyzer()
        
    async def run_single_scan(self):
        """단일 스캔 실행 (테스트용)"""
        try:
            logger.info("단일 시장 스캔 시작...")
            
            # 상위 거래량 심볼 가져오기
            symbols = self.market_analyzer.get_top_symbols(limit=20)
            
            if not symbols:
                logger.warning("심볼을 가져올 수 없습니다.")
                return
                
            logger.info(f"분석 대상 심볼: {len(symbols)}개")
            
            all_signals = []
            
            for symbol in symbols[:10]:  # 상위 10개만 분석
                try:
                    logger.info(f"{symbol} 분석 중...")
                    
                    # 캔들스틱 데이터 가져오기
                    df = self.market_analyzer.get_klines(symbol, interval='30m', limit=240)
                    
                    if df is None:
                        continue
                        
                    # 기술적 지표 계산
                    df = self.market_analyzer.calculate_technical_indicators(df)
                    
                    if df is None:
                        continue
                        
                    # 고급 시그널 생성
                    signal = self.advanced_generator.generate_advanced_signal(symbol, df)
                    
                    if signal:
                        all_signals.append(signal)
                        logger.info(f"{symbol}: {signal['type']} 시그널 생성 (신뢰도: {signal['confidence']}%)")
                        
                except Exception as e:
                    logger.error(f"{symbol} 분석 오류: {e}")
                    continue
                    
            # 품질 기준으로 필터링
            filtered_signals = self.advanced_generator.filter_signals_by_quality(all_signals, min_confidence=70)
            
            logger.info(f"필터링된 시그널: {len(filtered_signals)}개")
            
            # 상위 시그널만 전송
            top_signals = filtered_signals[:5]  # 상위 5개만
            
            for signal in top_signals:
                try:
                    await self.telegram_bot.send_signal(signal)
                    await asyncio.sleep(3)  # 3초 딜레이
                except Exception as e:
                    logger.error(f"시그널 전송 오류: {e}")
                    
            logger.info("단일 스캔 완료")
            
        except Exception as e:
            logger.error(f"단일 스캔 오류: {e}")
            
    async def run_continuous_monitoring(self):
        """연속 모니터링 실행"""
        try:
            logger.info("연속 모니터링 시작...")
            
            while True:
                try:
                    await self.run_single_scan()
                    
                    # 다음 스캔까지 대기
                    logger.info(f"{SIGNAL_INTERVAL}분 후 다음 스캔...")
                    await asyncio.sleep(SIGNAL_INTERVAL * 60)
                    
                except Exception as e:
                    logger.error(f"모니터링 루프 오류: {e}")
                    await asyncio.sleep(300)  # 5분 대기 후 재시도
                    
        except KeyboardInterrupt:
            logger.info("모니터링 중단됨")
        except Exception as e:
            logger.error(f"연속 모니터링 오류: {e}")
            
    def check_environment(self):
        """환경 설정 확인"""
        required_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY'
        ]
        
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            logger.error(f"필수 환경 변수가 설정되지 않음: {missing_vars}")
            return False
            
        logger.info("환경 설정 확인 완료")
        return True
        
    def print_banner(self):
        """봇 배너 출력"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 완벽한 시그널 봇 🚀                    ║
║                                                              ║
║  • 실시간 시장 분석 및 자동 시그널 생성                      ║
║  • 고급 기술적 지표 분석 (RSI, MACD, 볼린저 밴드 등)        ║
║  • 승률 높은 진입/청산 포인트 자동 계산                     ║
║  • 텔레그램 자동 시그널 전송                                ║
║  • 리스크 관리 및 최적 레버리지 계산                         ║
║                                                              ║
║  시작 시간: {}                                    ║
╚══════════════════════════════════════════════════════════════╝
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        print(banner)
        
    async def start(self, mode='continuous'):
        """봇 시작"""
        try:
            self.print_banner()
            
            # 환경 설정 확인
            if not self.check_environment():
                logger.error("환경 설정이 올바르지 않습니다. .env 파일을 확인해주세요.")
                return
                
            logger.info("봇 초기화 완료")
            
            if mode == 'single':
                await self.run_single_scan()
            elif mode == 'continuous':
                await self.run_continuous_monitoring()
            else:
                logger.error(f"알 수 없는 모드: {mode}")
                
        except Exception as e:
            logger.error(f"봇 시작 오류: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='완벽한 텔레그램 코인 시그널 봇')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='continuous',
                       help='실행 모드: single (단일 스캔) 또는 continuous (연속 모니터링)')
    parser.add_argument('--realtime', action='store_true', help='실시간 스캔/모니터링 모드')
    
    args = parser.parse_args()
    
    # 봇 인스턴스 생성 및 시작
    bot = PerfectSignalBot()
    
    try:
        if args.realtime:
            # 실시간 모드 직접 진입
            if WS_ENABLED:
                bot.telegram_bot.run_ws()
            else:
                bot.telegram_bot.run_realtime()
        else:
            asyncio.run(bot.start(mode=args.mode))
    except KeyboardInterrupt:
        logger.info("사용자에 의해 봇이 중단됨")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()
