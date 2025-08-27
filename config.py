import os
from dotenv import load_dotenv

load_dotenv()

# 텔레그램 설정
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# 바이낸스 설정
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# 시그널 설정
SIGNAL_INTERVAL = 30  # 분 단위
LEVERAGE_DEFAULT = 10  # 기본 레버리지
RISK_PERCENTAGE = 2.0  # 리스크 퍼센트
PROFIT_TARGETS = [1.5, 2.5, 3.5, 4.5]  # 수익 목표 (%)
ATR_PROFIT_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]  # ATR 기반 수익 목표 배수

# 리스크/사이징/트레일링 스탑 설정
TRADING_EQUITY_USDT = float(os.getenv('TRADING_EQUITY_USDT', '0'))  # 계정 운용 자본(선택)
MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', str(LEVERAGE_DEFAULT)))  # 최대 레버리지 상한
# per-trade 리스크는 RISK_PERCENTAGE 사용(%)
PARTIAL_TP_WEIGHTS = [0.3, 0.3, 0.2, 0.2]  # 분할 익절 비중 (합계 1.0)
TRAILING_ATR_MULTIPLIER = 1.0  # 트레일링 스탑 거리 (ATR 배수)
TRAILING_ACTIVATE_AFTER_TP_INDEX = 0  # TP 몇 번째 체결 후 트레일링 활성화(0 기반)
 
# 품질/필터/파생데이터 설정
QUALITY_MIN_CONFIDENCE = int(os.getenv('QUALITY_MIN_CONFIDENCE', '70'))  # 전송 최소 신뢰도
MIN_RR_AVG = float(os.getenv('MIN_RR_AVG', '2.0'))  # 평균 R:R 하한
MAX_FUNDING_ABS = float(os.getenv('MAX_FUNDING_ABS', '0.03'))  # 펀딩률 절대값 상한(%)
MIN_OPEN_INTEREST_USDT = float(os.getenv('MIN_OPEN_INTEREST_USDT', '5000000'))  # 최소 OI(명목 USDT)
LONG_SHORT_EXTREME_HIGH = float(os.getenv('LONG_SHORT_EXTREME_HIGH', '1.8'))  # 롱숏비 과열 상한
LONG_SHORT_EXTREME_LOW = float(os.getenv('LONG_SHORT_EXTREME_LOW', '0.55'))   # 롱숏비 과열 하한

# 분석 설정
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# 거래 설정
MIN_VOLUME_24H = 500000  # 최소 24시간 거래량 (USDT)
MIN_MARKET_CAP = 10000000  # 최소 시가총액 (USDT)
MAX_SPREAD = 0.8  # 최대 스프레드 (%)

# 멀티 타임프레임/쿨다운 설정
SCAN_INTERVALS = ['15m', '30m', '1h']  # 스캔할 타임프레임 목록
ENABLE_MTF_CONFIRM = True
MTF_CONFIRM_INTERVALS = ['1h', '4h']  # 상위 컨펌 타임프레임(엄격)
ADX_MIN = 15  # 추세 강도 임계치(완화)
COOLDOWN_CANDLES = 2  # 동일 심볼/타임프레임 최소 대기 캔들 수(완화)
MTF_CONFIRM_STRICT = os.getenv('MTF_CONFIRM_STRICT', 'true').lower() == 'true'  # True면 미컨펌 시 신호 폐기, False면 감점
MTF_CONFIRM_CACHE_MINUTES = int(os.getenv('MTF_CONFIRM_CACHE_MINUTES', '10'))

# 실시간 스캔/모니터링 주기(초)
REALTIME_SCAN_SECONDS = int(os.getenv('REALTIME_SCAN_SECONDS', '60'))
MONITOR_INTERVAL_SECONDS = int(os.getenv('MONITOR_INTERVAL_SECONDS', '10'))
WS_ENABLED = os.getenv('WS_ENABLED', 'false').lower() == 'true'  # WebSocket 실시간 모드 활성화

# 레짐 필터(변동성/밴드 폭)
ATR_PCT_MIN = float(os.getenv('ATR_PCT_MIN', '0.2'))  # % 단위 (예: 0.2 = 0.2%)
ATR_PCT_MAX = float(os.getenv('ATR_PCT_MAX', '5.0'))  # 과도 변동성 컷
BB_WIDTH_MIN = float(os.getenv('BB_WIDTH_MIN', '0.005'))  # 밴드폭/중심 (예: 0.005 = 0.5%)
BB_WIDTH_MAX = float(os.getenv('BB_WIDTH_MAX', '0.2'))

# 동적 사이징
DYNAMIC_SIZING_MULTIPLIER_MAX = float(os.getenv('DYNAMIC_SIZING_MULTIPLIER_MAX', '1.5'))
