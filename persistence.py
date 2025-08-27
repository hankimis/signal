import sqlite3
import json
import threading
from typing import Any, Dict, List, Optional

_DB_PATH = 'signals.db'
_LOCK = threading.Lock()


def _get_conn():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    return conn


def init_db() -> None:
    with _LOCK:
        conn = _get_conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    type TEXT NOT NULL,
                    entry_prices TEXT NOT NULL,
                    profit_targets TEXT NOT NULL,
                    stop_loss REAL NOT NULL,
                    leverage INTEGER,
                    confidence INTEGER,
                    quality TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_open INTEGER DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    event TEXT NOT NULL,
                    price REAL,
                    info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def save_signal(signal: Dict[str, Any]) -> None:
    with _LOCK:
        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT INTO signals (
                    symbol, interval, type, entry_prices, profit_targets, stop_loss,
                    leverage, confidence, quality, is_open
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    signal.get('symbol'),
                    signal.get('interval'),
                    signal.get('type'),
                    json.dumps(signal.get('entry_prices', [])),
                    json.dumps(signal.get('profit_targets', [])),
                    float(signal.get('stop_loss')) if signal.get('stop_loss') is not None else None,
                    int(signal.get('leverage')) if signal.get('leverage') is not None else None,
                    int(signal.get('confidence')) if signal.get('confidence') is not None else None,
                    signal.get('quality'),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def update_signal_event(symbol: str, interval: str, event: str, price: Optional[float] = None, info: Optional[Dict[str, Any]] = None, close: bool = False) -> None:
    info_str = json.dumps(info) if info else None
    with _LOCK:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO signal_events (symbol, interval, event, price, info) VALUES (?, ?, ?, ?, ?)",
                (symbol, interval, event, price, info_str),
            )
            if close:
                # 최신 미종료 시그널을 종료 처리
                conn.execute(
                    """
                    UPDATE signals
                    SET is_open = 0
                    WHERE id = (
                        SELECT id FROM signals
                        WHERE symbol = ? AND interval = ? AND is_open = 1
                        ORDER BY created_at DESC
                        LIMIT 1
                    )
                    """,
                    (symbol, interval),
                )
            conn.commit()
        finally:
            conn.close()


def load_open_signals() -> List[Dict[str, Any]]:
    with _LOCK:
        conn = _get_conn()
        try:
            cur = conn.execute(
                "SELECT symbol, interval, type, entry_prices, profit_targets, stop_loss, leverage, confidence, quality, created_at FROM signals WHERE is_open = 1"
            )
            rows = cur.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                results.append(
                    {
                        'symbol': r[0],
                        'interval': r[1],
                        'type': r[2],
                        'entry_prices': json.loads(r[3]) if r[3] else [],
                        'profit_targets': json.loads(r[4]) if r[4] else [],
                        'stop_loss': r[5],
                        'leverage': r[6],
                        'confidence': r[7],
                        'quality': r[8],
                        'created_at': r[9],
                    }
                )
            return results
        finally:
            conn.close()


