from market_analyzer import MarketAnalyzer

def backtest(symbol: str = 'BTCUSDT', interval: str = '30m', lookback: int = 400) -> str:
    """간단 백테스트: 엔트리/TP/SL 규칙 동일 적용(가상 체결). 문자열 리포트 반환"""
    ma = MarketAnalyzer()
    df = ma.get_klines(symbol, interval=interval, limit=lookback+240, exclude_last_open=True)
    df = ma.calculate_technical_indicators(df)
    if df is None or len(df) < 60:
        return "데이터 부족"
    wins = 0; losses = 0; pnls = []
    for i in range(60, len(df)):
        sub = df.iloc[:i]
        # 백테스트는 필터 완화(상위TF/파생 게이트 무시)
        sig = ma.generate_signal(symbol, sub, interval=interval, relaxed=True, ignore_derivatives=True)
        if not sig:
            continue
        entry = sig['entry_prices'][0]
        sl = sig['stop_loss']
        tps = sig['profit_targets']
        typ = sig['type']
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
    return "\n".join(lines)


