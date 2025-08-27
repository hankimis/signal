#!/usr/bin/env python3
"""
봇 테스트 스크립트
"""

import asyncio
import logging
from market_analyzer import MarketAnalyzer
from signal_generator import AdvancedSignalGenerator
from telegram_bot import SignalBot

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_market_analyzer():
    """시장 분석기 테스트"""
    print("🔍 시장 분석기 테스트 시작...")
    
    try:
        analyzer = MarketAnalyzer()
        
        # 상위 심볼 가져오기 테스트
        symbols = analyzer.get_top_symbols(limit=5)
        print(f"✅ 상위 심볼 가져오기: {symbols[:3]}")
        
        if symbols:
            # 첫 번째 심볼로 상세 분석 테스트
            symbol = symbols[0]
            print(f"📊 {symbol} 상세 분석 중...")
            
            # 캔들스틱 데이터 가져오기
            df = analyzer.get_klines(symbol, interval='30m', limit=100)
            if df is not None:
                print(f"✅ 캔들스틱 데이터: {len(df)}개")
                
                # 기술적 지표 계산
                df = analyzer.calculate_technical_indicators(df)
                if df is not None:
                    print(f"✅ 기술적 지표 계산 완료")
                    
                    # 시그널 생성 테스트
                    signal = analyzer.generate_signal(symbol, df)
                    if signal:
                        print(f"✅ 시그널 생성: {signal['type']} (신뢰도: {signal['confidence']}%)")
                    else:
                        print("⚠️ 시그널 생성 실패 (현재 시장 상황에 맞지 않음)")
                else:
                    print("❌ 기술적 지표 계산 실패")
            else:
                print("❌ 캔들스틱 데이터 가져오기 실패")
        else:
            print("❌ 상위 심볼 가져오기 실패")
            
    except Exception as e:
        print(f"❌ 시장 분석기 테스트 오류: {e}")

async def test_advanced_signal_generator():
    """고급 시그널 생성기 테스트"""
    print("\n🚀 고급 시그널 생성기 테스트 시작...")
    
    try:
        generator = AdvancedSignalGenerator()
        analyzer = MarketAnalyzer()
        
        # 테스트용 심볼
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in test_symbols:
            print(f"📊 {symbol} 고급 분석 중...")
            
            # 데이터 가져오기
            df = analyzer.get_klines(symbol, interval='30m', limit=100)
            if df is not None:
                df = analyzer.calculate_technical_indicators(df)
                if df is not None:
                    # 고급 시그널 생성
                    advanced_signal = generator.generate_advanced_signal(symbol, df)
                    if advanced_signal:
                        print(f"✅ 고급 시그널: {advanced_signal['type']} (품질: {advanced_signal['quality']}, 신뢰도: {advanced_signal['confidence']}%)")
                        
                        # 추가 분석 정보 출력
                        if advanced_signal.get('volume_profile'):
                            print(f"   📈 볼륨 트렌드: {advanced_signal['volume_profile']['volume_trend']}")
                        if advanced_signal.get('momentum_divergence'):
                            print(f"   🔄 다이버전스: {advanced_signal['momentum_divergence']}")
                        if advanced_signal.get('partial_tp_weights'):
                            print(f"   🎯 분할 TP 비중: {advanced_signal['partial_tp_weights']}")
                    else:
                        print(f"⚠️ {symbol}: 고급 시그널 생성 실패")
                else:
                    print(f"❌ {symbol}: 기술적 지표 계산 실패")
            else:
                print(f"❌ {symbol}: 데이터 가져오기 실패")
                
    except Exception as e:
        print(f"❌ 고급 시그널 생성기 테스트 오류: {e}")

async def test_telegram_bot():
    """텔레그램 봇 테스트"""
    print("\n📱 텔레그램 봇 테스트 시작...")
    
    try:
        bot = SignalBot()
        
        # 테스트 메시지 전송
        print("📤 테스트 메시지 전송 중...")
        await bot.test_message()
        print("✅ 테스트 메시지 전송 완료")
        
    except Exception as e:
        print(f"❌ 텔레그램 봇 테스트 오류: {e}")
        print("💡 환경 변수 설정을 확인해주세요 (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")

async def run_all_tests():
    """모든 테스트 실행"""
    print("🧪 완벽한 시그널 봇 테스트 시작\n")
    
    # 시장 분석기 테스트
    await test_market_analyzer()
    
    # 고급 시그널 생성기 테스트
    await test_advanced_signal_generator()
    
    # 텔레그램 봇 테스트
    await test_telegram_bot()
    
    print("\n🎉 모든 테스트 완료!")

def backtest(symbol: str = 'BTCUSDT', interval: str = '30m', lookback: int = 400):
    """간단 백테스트: 엔트리/TP/SL 규칙 동일 적용(가상 체결). 문자열 리포트를 반환."""
    from market_analyzer import MarketAnalyzer
    import numpy as np
    ma = MarketAnalyzer()
    df = ma.get_klines(symbol, interval=interval, limit=lookback+120, exclude_last_open=True)
    df = ma.calculate_technical_indicators(df)
    if df is None or len(df) < 60:
        return "데이터 부족"
    wins = 0; losses = 0; pnls = []
    for i in range(60, len(df)):
        sub = df.iloc[:i]
        sig = ma.generate_signal(symbol, sub, interval=interval)
        if not sig:
            continue
        entry = sig['entry_prices'][0]
        sl = sig['stop_loss']
        tps = sig['profit_targets']
        typ = sig['type']
        # 다음 캔들부터 체결 시퀀스 스캔
        future = df.iloc[i: i+50]
        hit = None
        for _, r in future.iterrows():
            high = r['high']; low = r['low']
            if typ == 'LONG':
                if any(high >= tp for tp in tps):
                    hit = 'TP'; break
                if low <= sl:
                    hit = 'SL'; break
            else:
                if any(low <= tp for tp in tps):
                    hit = 'TP'; break
                if high >= sl:
                    hit = 'SL'; break
        if hit == 'TP':
            pnl = (tps[0] - entry)/entry*100 if typ=='LONG' else (entry - tps[0])/entry*100
            wins += 1; pnls.append(pnl)
        elif hit == 'SL':
            pnl = (sl - entry)/entry*100 if typ=='LONG' else (entry - sl)/entry*100
            losses += 1; pnls.append(pnl)
    total = wins + losses
    if total == 0:
        return "시그널 없음"
    import statistics
    lines = []
    lines.append(f"심볼 {symbol} | {interval} | 샘플 {total}")
    lines.append(f"승률 {wins/total*100:.2f}% | 평균Pnl {statistics.mean(pnls):.2f}% | 합계Pnl {sum(pnls):.2f}%")
    report = "\n".join(lines)
    print(report)
    return report

def main():
    """메인 함수"""
    import argparse
    parser = argparse.ArgumentParser(description='시그널 봇 테스트/백테스트')
    parser.add_argument('--backtest', action='store_true', help='간단 백테스트 실행')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='30m')
    parser.add_argument('--lookback', type=int, default=400)
    args = parser.parse_args()
    try:
        if args.backtest:
            backtest(args.symbol, args.interval, args.lookback)
        else:
            asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⏹️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")

if __name__ == "__main__":
    main()
