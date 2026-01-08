import time
from datetime import datetime, timezone


def now_ns() -> int:
    """現在時刻を ns 単位で返す"""
    return time.time_ns()


def ns_to_iso(ns: int) -> str:
    """ns タイムスタンプを UTC の ISO8601 文字列に変換"""
    sec = ns / 1e9
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return dt.isoformat()
