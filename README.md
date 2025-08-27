# 🚀 완벽한 텔레그램 코인 시그널 봇

승률 높은 자동 코인 시그널 생성 및 텔레그램 전송 봇입니다. 미종가 캔들 제외, 멀티 타임프레임(HTF) 컨펌, 쿨다운, 스프레드/유동성 필터 등을 적용해 실전 친화적으로 구성했습니다.

## ✨ 주요 기능

- **실시간 시장 분석**: 바이낸스 API를 통한 실시간 데이터 수집 (미종가 캔들 제외)
- **고급 기술적 분석**: RSI, MACD, 볼린저 밴드, EMA/SMA, ADX, MFI, OBV
- **멀티 타임프레임 컨펌**: 30m/1h 스캔 + 1h/4h 상위 추세 컨펌(옵션)
- **자동 시그널 생성**: ATR 기반 SL/TP, 3계단 분할 진입 가격(Entry 1~3)
- **스프레드/유동성 필터**: 24h 거래량 하한, 호가 스프레드 상한 적용
- **텔레그램 자동 전송**: 설정된 방으로 간결 포맷 메시지 전송
- **리스크 관리**: per-trade 리스크%, 레버리지 상한, (옵션) 포지션 사이즈 산출
- **쿨다운 정책**: 심볼/타임프레임별 최소 대기 캔들 수 적용

## 🛠️ 설치 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`env_example.txt` 파일을 참고하여 `.env` 파일을 생성하세요:

```bash
# 텔레그램 봇 설정
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# 바이낸스 API 설정
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
```

### 3. 텔레그램 봇 설정

1. [@BotFather](https://t.me/botfather)에서 봇 생성
2. 봇 토큰을 `TELEGRAM_BOT_TOKEN`에 설정
3. 봇을 원하는 채팅방에 초대
4. 채팅방 ID를 `TELEGRAM_CHAT_ID`에 설정

### 4. 바이낸스 API 설정

1. [바이낸스](https://www.binance.com) 계정 생성
2. API 키 생성 (읽기 권한만 필요)
3. API 키와 시크릿을 환경 변수에 설정

## 🚀 사용 방법

### 연속 모니터링 (기본)

```bash
python main.py
```

### 단일 스캔 (테스트용)

```bash
python main.py --mode single
```

## 📊 시그널 예시(간결 포맷)

```
🚀 #BTCUSDT | SHORT | 30m

Entry
1) 109491.6766
2) 109257.3132
3) 109022.9498

Stop-Loss
109980.0000

Leverage
10x [Isolated]
```

추가로 바이어스/포지션 사이즈/신뢰도/생성시간 등을 함께 표기하도록 설정할 수도 있습니다.

## ⚙️ 설정 옵션

`config.py` 파일에서 다음 설정을 조정할 수 있습니다:

- **SIGNAL_INTERVAL**: 스케줄러 사용 시 스캔 간격 (분)
- **SCAN_INTERVALS**: 스캔할 타임프레임 목록 예) ['15m','30m','1h']
- **ENABLE_MTF_CONFIRM**: 상위 타임프레임 컨펌 사용 여부
- **MTF_CONFIRM_INTERVALS**: 컨펌용 HTF 목록 예) ['1h','4h']
- **ADX_MIN**: 추세 강도 임계치 (기본 15~20 권장)
- **COOLDOWN_CANDLES**: 동일 심볼/타임프레임 재진입 최소 대기 캔들 수
- **MIN_VOLUME_24H**: 최소 24시간 거래량(USDT)
- **MAX_SPREAD**: 호가 스프레드 상한(%)
- **LEVERAGE_DEFAULT / MAX_LEVERAGE**: 기본/최대 레버리지
- **RISK_PERCENTAGE**: per-trade 계정 리스크 %
- **TRADING_EQUITY_USDT**: 운용 자본(USDT, 포지션 사이징 계산에 사용, 선택)
- **ATR_PROFIT_MULTIPLIERS**: ATR 기반 TP 배수 예) [0.5,1.0,1.5,2.0]
- **PROFIT_TARGETS**: 백업용 % TP (ATR 불가 시)
- **PARTIAL_TP_WEIGHTS**: 분할 익절 비중 예) [0.3,0.3,0.2,0.2]
- **TRAILING_ATR_MULTIPLIER**: 트레일링 스탑 거리(ATR 배수)

### 실시간 완화 모드(신호 생성 완화 + 전송 기준 강화)

환경변수 또는 `.env` 에 아래 값을 추가하면, 실시간 스캔 시 상위TF/파생 게이트를 완화해 후보 신호를 더 넓게 잡고, 전송 기준은 더 높게 잡아 승률을 우선합니다.

```env
REALTIME_RELAXED=true                 # 실시간 완화 모드 ON
RELAXED_IGNORE_MTF=true               # 상위TF 컨펌을 생성 단계에서 무시
RELAXED_IGNORE_DERIVATIVES=true       # 펀딩/OI/롱숏비 게이트 생성 단계에서 무시
RELAXED_MIN_CONFIDENCE=80             # 전송 최소 신뢰도(완화 모드)
RELAXED_MIN_RR_AVG=2.5                # 전송 최소 평균 R:R(완화 모드)
```

실시간 실행:

```bash
python main.py --realtime
```

### 텔레그램 명령어

봇이 실행 중일 때 채팅방에서 다음 명령어로 실시간 상태/지표/백테스트를 조회할 수 있습니다. 실시간 모드에서는 명령 처리가 스캔과 병렬로 동작해 즉시 응답합니다.

- `/status`: 현재 스캔 타임프레임, 최소 신뢰도·R:R, 오픈 시그널 개수 요약
- `/top [개수]`: 거래량·유동성 필터를 통과한 상위 심볼 목록 (기본 15)
- `/open`: 현재 오픈(모니터링 중) 시그널 목록 요약
- `/metrics [심볼] [주기]`: 실시간 지표/필터 수치와 시그널 프리뷰
  - 예) `/metrics BTCUSDT 30m`
  - 제공: 가격, RSI, ADX, MACD, BB폭, ATR/ATR%, 펀딩%, OI(USDT), 롱숏비, 레짐 OK/FAIL, MTF 컨펌, 베이식 시그널 프리뷰(R:R 포함)
- `/backtest [심볼] [주기] [캔들수]`: 간단 백테스트 결과
  - 예) `/backtest BTCUSDT 30m 600`
  - 캔들수는 400~1200 권장. 신호가 없을 경우 “시그널 없음” 출력
- 별칭: `/metrics`와 동일 동작을 `/why`, `/debug`로도 호출 가능

## 🔧 고급 분석 기능

### 기술적 지표

- **RSI**: 상대강도지수 (과매수/과매도 판단)
- **MACD**: 이동평균수렴확산지수 (추세 전환 신호)
- **볼린저 밴드**: 변동성 및 가격 범위 분석
- **이동평균**: 단기/장기 추세 분석
- **ATR**: 평균진폭 (변동성 측정)

### 추가 분석

- **볼륨 프로파일**: 거래량 분포 분석
- **지지/저항선**: 피벗 포인트 및 주요 레벨
- **모멘텀 다이버전스**: 가격과 지표의 불일치 감지
- **리스크 대비 수익**: 최적 진입점 계산
- **멀티 타임프레임 컨펌**: HTF 추세/모멘텀/강도 정렬 확인
- **미종가 캔들 제외**: 리페인트/훼손 방지

## 📈 승률 향상 전략

1. **다중 지표 확인**: 최소 2개 이상의 지표가 일치할 때만 시그널 생성
2. **볼륨 확인**: 거래량 증가와 함께하는 움직임 우선
3. **지지/저항 활용**: 주요 레벨 근처에서의 진입
4. **리스크 관리**: 1:2 이상의 리스크 대비 수익 비율
5. **시장 심리 분석**: 전체적인 시장 분위기 고려

## 🚨 주의사항

- 이 봇은 투자 조언이 아닙니다
- 실제 거래에 사용하기 전에 충분한 테스트가 필요합니다
- 리스크 관리에 주의를 기울이세요
- API 키는 안전하게 보관하세요

## 📝 로그

봇 실행 시 `signal_bot.log` 파일에 상세한 로그가 기록됩니다.

## 🤝 기여

버그 리포트나 기능 제안은 언제든 환영합니다!

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
