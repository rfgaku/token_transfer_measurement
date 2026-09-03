#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
native_bridge_arrival_survey.py

Hyperliquid ネイティブブリッジ（Arbitrum One 上の Bridge2）に対する
  - deposit 到着（USDC Transfer で to == Bridge2）
  - withdraw 放出（USDC Transfer で from == Bridge2）
をオンチェーンのイベントログから悉皆収集し、到着率と同時負荷を集計する。

ACM DLT 投稿論文 付録用の追加調査スクリプト（読み取り専用・送金は一切行わない）。

対象期間
  A) 2025-11-27 00:00 UTC 〜 2025-12-09 00:00 UTC（T1 実測期間を含む 12 日間）
  B) 付録 E（CCTP 利用実態調査）と同一の 2026 年 6 月の 7 窓

使い方
  python3 experiments/native_bridge_arrival_survey.py verify
  python3 experiments/native_bridge_arrival_survey.py events --period A
  python3 experiments/native_bridge_arrival_survey.py events --period B
  python3 experiments/native_bridge_arrival_survey.py rates
  python3 experiments/native_bridge_arrival_survey.py concurrency
  python3 experiments/native_bridge_arrival_survey.py all

出力先
  成果物  : result/native_bridge_survey/*.csv
  中間物  : result/native_bridge_survey/_tmp/（進捗・キャッシュ・partial。削除して差し支えない）

冪等性
  ・ログ取得の進捗は result/native_bridge_survey/_tmp/ 配下の
      progress.json                               … 取得済みブロック範囲
      native_bridge_events_<period>.partial.csv   … 取得済み生ログ（追記）
      blockts_cache.json                          … block_number -> timestamp キャッシュ
    に残す。途中で落ちても同じコマンドを再実行すれば続きから取得する。
    _tmp/ を消しても成果物 CSV は残り、再実行で中間物だけが作り直される。

依存
  requests / pandas / numpy / scipy（いずれも既存環境に導入済み）。
  requirements 等には手を加えていない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# --------------------------------------------------------------------------- #
# パス / 定数
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "result"

if load_dotenv is not None:
    load_dotenv(REPO / ".env")

# Bridge2（Hyperliquid: Deposit Bridge 2, Arbitrum One）
BRIDGE2 = os.getenv(
    "HL_DEPOSIT_BRIDGE_ADDRESS", "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7"
).lower()
# Arbitrum One ネイティブ USDC
USDC = os.getenv(
    "ARB_USDC_ADDRESS", "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
).lower()
# RPC（既存設定を再利用。ARB_RPC_URL / ARBITRUM_HTTP_RPC のどちらでも拾う）
RPC_URL = (
    os.getenv("ARB_RPC_URL")
    or os.getenv("ARBITRUM_HTTP_RPC")
    or "https://arb1.arbitrum.io/rpc"
)

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
USDC_DECIMALS = 6

# 期間 A（UTC, [start, end)）
PERIOD_A_START = datetime(2025, 11, 27, 0, 0, 0, tzinfo=timezone.utc)
PERIOD_A_END = datetime(2025, 12, 9, 0, 0, 0, tzinfo=timezone.utc)

# eth_getLogs の初期スパン（RPC 制限に合わせて自動縮小する）
INITIAL_SPAN = 50_000
MIN_SPAN = 500
MAX_SPAN = 100_000

# eth_getBlockByNumber のバッチ設定
#   arb1 公開 RPC は 1 リクエストあたりのコスト上限が厳しく、50 件バッチを
#   並列で投げ続けると 429 が連発する。単スレッド + 小バッチ + 自動縮小で回す。
TS_BATCH = 25
TS_BATCH_MIN = 5
TS_BATCH_MAX = 75

# block timestamp の一括解決に使う読み取り専用エンドポイント群。
#   期間A は 8 万ブロック超の timestamp が必要で、arb1 公開 RPC 単独では
#   レート制限のため 1 時間近くかかる。timestamp はチェーン上の客観値であり
#   どのノードから引いても同一なので、鍵不要の公開エンドポイントを併用して
#   並列化する（送信系は一切使わない。詳細は NOTES 参照）。
TS_RPC_URLS = [
    "https://arbitrum-one.public.blastapi.io",
    "https://arbitrum-one-rpc.publicnode.com",
    RPC_URL,
]
TS_FAST_BATCH = 200

# Bridge2 自身のイベント（hyperliquid-dex/contracts の Bridge2.sol より）
# 参考系列として収集する。必須は上記 Transfer ベースの 2 系列。
BRIDGE2_EVENT_TOPICS = {
    "0x0ee94a97c7c69ce2eb8cfb09bacc78d63a73b5e0fbed0d13a079190ff876ae3a": "Deposit",
    "0xcc10abf54af5c0718b10b0156dfe1e369ce3eee72423e9e86936a0082e9c5d1b": "RequestedWithdrawal",
    "0xe5c7fe3a4ffca1590f26d74c8ba8b0db69557f7f4607a2a43f82e93041611978": "FinalizedWithdrawal",
    "0x686cb4bac974cd11b0f8a75fc7c7764ed12cc46faaec53110f807aa802a7acb4": "FailedWithdrawal",
    "0x420bbe99bd2c52ec500d33614359525f3ef7bb3358c0e07d1312db0941cbf2f4": "RequestedValidatorSetUpdate",
    "0x87da17ff65d815d1e1c369cb3bbda9a11af181b92dc52681a2779419781c6270": "FinalizedValidatorSetUpdate",
    "0x26690dc5c5a9d2aa7ac3efa2b7c515652e4621a3e075d267bcac51c16fb97532": "ModifiedLocker",
    "0xa2dc875d1f90a167d873c30143e7631eb311ea851e74c8c4e9b92c80efeba489": "FailedPermitDeposit",
    "0x2526bb92d75e00cfad8c7c16cb75f3e1073c854339e49b16baaad3067c2ed65a": "ModifiedFinalizer",
    "0x04edaf680108675f58d2ea70e9e7886c39ed38b66439622f8362d36595fe8169": "ChangedDisputePeriodSeconds",
    "0x0ef2da393c3832a8f08ce447e14948d21e84f864facf7327137387bd0596a563": "ChangedBlockDurationMillis",
    "0x2dbe453726b24b2cee427a7d6e2dcc9f353f16bee104f3d21480157a0ee409f7": "ChangedLockerThreshold",
}

BRIDGE_EVENT_COLUMNS = [
    "block_number",
    "block_timestamp_utc",
    "block_timestamp_ms",
    "tx_hash",
    "log_index",
    "event",
    "topic0",
    "user",
    "amount_usdc",
    "nonce",
]

EVENT_COLUMNS = [
    "block_number",
    "block_timestamp_utc",
    "block_timestamp_ms",
    "tx_hash",
    "log_index",
    "from",
    "to",
    "amount_usdc",
    "direction",
]

# 成果物は result/native_bridge_survey/ に置き、result/ 直下は汚さない。
# 再実行で再生成できる中間物（進捗・キャッシュ・partial）は同フォルダ配下の _tmp/ に隔離する。
SURVEY_DIR = RESULT / "native_bridge_survey"
TMP_DIR = SURVEY_DIR / "_tmp"
SURVEY_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = TMP_DIR / "progress.json"
BLOCKTS_CACHE = TMP_DIR / "blockts_cache.json"
CCTP_EVENTS = RESULT / "T4_cctp" / "cctp_fast_standard_events.csv"
DEPOSIT_LATENCY = RESULT / "deposit_latency.csv"
WITHDRAW_LATENCY = RESULT / "withdraw_latency_no_offset.csv"

OUT_EVENTS_A = SURVEY_DIR / "native_bridge_events_2025-11.csv"
OUT_EVENTS_B = SURVEY_DIR / "native_bridge_events_2026-06.csv"
OUT_RATES = SURVEY_DIR / "native_bridge_arrival_rates.csv"
OUT_BRIDGE_A = SURVEY_DIR / "native_bridge_contract_events_2025-11.csv"
OUT_BRIDGE_B = SURVEY_DIR / "native_bridge_contract_events_2026-06.csv"
OUT_DEP_CONC = SURVEY_DIR / "t1_deposit_concurrency.csv"
OUT_WIT_CONC = SURVEY_DIR / "t1_withdraw_concurrency.csv"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# RPC クライアント（リトライ / レート制限対策つき）
# --------------------------------------------------------------------------- #
class Rpc:
    """JSON-RPC クライアント。429・タイムアウトに指数バックオフで耐える。"""

    def __init__(self, url: str = RPC_URL, min_interval: float = 0.12):
        self.url = url
        self.session = requests.Session()
        # 公開 RPC は User-Agent なしだと 403 を返すことがある
        self.session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (research-survey)"}
        )
        self._lock = threading.Lock()
        self._last = 0.0
        self.min_interval = min_interval
        self.n_calls = 0

    def _throttle(self) -> None:
        with self._lock:
            dt = time.time() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.time()
            self.n_calls += 1

    def _post(self, payload, timeout: int = 120):
        self._throttle()
        return self.session.post(self.url, json=payload, timeout=timeout)

    def call(self, method: str, params: list, retries: int = 6, timeout: int = 120):
        """単発呼び出し。復帰不能なエラーは RpcError を投げる。"""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        delay = 1.0
        last_err = None
        for _ in range(retries):
            try:
                r = self._post(payload, timeout=timeout)
                if r.status_code == 429:
                    last_err = "http 429"
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                r.raise_for_status()
                j = r.json()
                if "error" in j:
                    err = j["error"]
                    if err.get("code") == 429 or "too many" in str(err.get("message", "")).lower():
                        last_err = str(err)
                        time.sleep(delay)
                        delay = min(delay * 2, 30)
                        continue
                    raise RpcError(str(err))
                return j["result"]
            except RpcError:
                raise
            except Exception as e:  # ネットワーク断・タイムアウト等
                last_err = repr(e)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RpcError(f"{method} failed after {retries} retries: {last_err}")

    def batch(self, calls: list, retries: int = 6, timeout: int = 120) -> list:
        """[(method, params), ...] をまとめて投げ、id 順に result のリストを返す。"""
        payload = [
            {"jsonrpc": "2.0", "id": i, "method": m, "params": p}
            for i, (m, p) in enumerate(calls)
        ]
        delay = 1.0
        last_err = None
        for _ in range(retries):
            try:
                r = self._post(payload, timeout=timeout)
                if r.status_code == 429:
                    last_err = "http 429"
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                r.raise_for_status()
                j = r.json()
                if isinstance(j, dict):  # バッチ全体が拒否された（429 等）
                    last_err = str(j.get("error"))
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                out = [None] * len(calls)
                for item in j:
                    if "error" in item:
                        raise RpcError(str(item["error"]))
                    out[item["id"]] = item["result"]
                if any(o is None for o in out):
                    raise RpcError("incomplete batch response")
                return out
            except RpcError:
                raise
            except Exception as e:
                last_err = repr(e)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RpcError(f"batch failed after {retries} retries: {last_err}")


class RpcError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# ブロック <-> 時刻
# --------------------------------------------------------------------------- #
_ts_cache: dict[int, int] = {}
_ts_cache_lock = threading.Lock()


def load_ts_cache() -> None:
    global _ts_cache
    if BLOCKTS_CACHE.exists():
        try:
            _ts_cache = {int(k): int(v) for k, v in json.loads(BLOCKTS_CACHE.read_text()).items()}
            log(f"block-ts cache loaded: {len(_ts_cache)} entries")
        except Exception as e:
            log(f"[WARN] block-ts cache 読み込み失敗（無視して続行）: {e}")
            _ts_cache = {}


def save_ts_cache() -> None:
    tmp = BLOCKTS_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in _ts_cache.items()}))
    tmp.replace(BLOCKTS_CACHE)


def block_ts(rpc: Rpc, n: int) -> int:
    with _ts_cache_lock:
        if n in _ts_cache:
            return _ts_cache[n]
    b = rpc.call("eth_getBlockByNumber", [hex(n), False])
    ts = int(b["timestamp"], 16)
    with _ts_cache_lock:
        _ts_cache[n] = ts
    return ts


def _ts_batch_once(client: Rpc, blocks: list[int]) -> list[int]:
    """1 リクエストで timestamp をまとめて取得。取れなかったブロックを返す。"""
    payload = [
        {"jsonrpc": "2.0", "id": i, "method": "eth_getBlockByNumber", "params": [hex(b), False]}
        for i, b in enumerate(blocks)
    ]
    try:
        r = client._post(payload, timeout=90)
        if r.status_code != 200:
            return blocks
        j = r.json()
        if not isinstance(j, list):
            return blocks
    except Exception:
        return blocks
    missing = []
    got = {}
    for item in j:
        b = blocks[item["id"]]
        res = item.get("result")
        if res and "timestamp" in res:
            got[b] = int(res["timestamp"], 16)
        else:
            missing.append(b)
    if len(got) + len(missing) != len(blocks):
        seen = set(got) | set(missing)
        missing += [b for b in blocks if b not in seen]
    with _ts_cache_lock:
        _ts_cache.update(got)
    return missing


def fetch_block_ts_many(rpc: Rpc, blocks: list[int]) -> None:
    """未キャッシュのブロック時刻を、複数エンドポイントに分散したバッチで取得する。"""
    need = sorted({b for b in blocks if b not in _ts_cache})
    if not need:
        return
    total = len(need)
    clients = [Rpc(u, min_interval=0.05) for u in TS_RPC_URLS]
    log(f"block timestamp を取得: {total} blocks "
        f"(batch={TS_FAST_BATCH}, endpoints={len(clients)})")
    chunks = [need[i : i + TS_FAST_BATCH] for i in range(0, len(need), TS_FAST_BATCH)]
    t0 = time.time()
    done = [0]
    lock = threading.Lock()

    def work(idx_chunk):
        idx, chunk = idx_chunk
        pending = chunk
        delay = 0.5
        for attempt in range(12):
            client = clients[(idx + attempt) % len(clients)]
            pending = _ts_batch_once(client, pending)
            if not pending:
                break
            time.sleep(delay)
            delay = min(delay * 1.7, 20)
        if pending:
            # 最後の手段: 1 件ずつ確実に取りに行く
            for b in pending:
                block_ts(rpc, b)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"  block ts {min(done[0] * TS_FAST_BATCH, total)}/{total} "
                    f"({time.time() - t0:.0f}s)")
                save_ts_cache()

    with ThreadPoolExecutor(max_workers=len(clients) * 2) as ex:
        list(ex.map(work, list(enumerate(chunks))))
    save_ts_cache()
    log(f"  block ts done {total} blocks in {time.time() - t0:.0f}s")



def find_block_by_time(rpc: Rpc, target_ts: int, lo: int, hi: int) -> int:
    """timestamp >= target_ts となる最小のブロック番号を二分探索で求める。"""
    lo_ts, hi_ts = block_ts(rpc, lo), block_ts(rpc, hi)
    if lo_ts >= target_ts:
        return lo
    if hi_ts < target_ts:
        raise RpcError(f"target_ts {target_ts} は探索上限 {hi} (ts={hi_ts}) より後")
    steps = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if block_ts(rpc, mid) >= target_ts:
            hi = mid
        else:
            lo = mid + 1
        steps += 1
    log(f"binary search: ts={target_ts} -> block {lo} (ts={block_ts(rpc, lo)}, {steps} steps)")
    return lo


# --------------------------------------------------------------------------- #
# 進捗ファイル（冪等性）
# --------------------------------------------------------------------------- #
def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except Exception:
            pass
    return {}


def save_progress(prog: dict) -> None:
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prog, indent=2))
    tmp.replace(PROGRESS_PATH)


# --------------------------------------------------------------------------- #
# ログ収集
# --------------------------------------------------------------------------- #
def topic_addr(addr: str) -> str:
    return "0x" + "0" * 24 + addr.lower().replace("0x", "")


def log_filter(direction: str, frm: int, to: int) -> dict:
    """direction='deposit' なら to==Bridge2, 'withdraw' なら from==Bridge2。"""
    if direction == "deposit":
        topics = [TRANSFER_TOPIC, None, topic_addr(BRIDGE2)]
    else:
        topics = [TRANSFER_TOPIC, topic_addr(BRIDGE2), None]
    return {
        "fromBlock": hex(frm),
        "toBlock": hex(to),
        "address": USDC,
        "topics": topics,
    }


def parse_log(item: dict, direction: str) -> dict:
    data = item["data"]
    amount_raw = int(data, 16) if data not in ("0x", "") else 0
    return {
        "block_number": int(item["blockNumber"], 16),
        "block_timestamp_utc": "",
        "block_timestamp_ms": 0,
        "tx_hash": item["transactionHash"],
        "log_index": int(item["logIndex"], 16),
        "from": "0x" + item["topics"][1][-40:],
        "to": "0x" + item["topics"][2][-40:],
        "amount_usdc": amount_raw / 10**USDC_DECIMALS,
        "direction": direction,
    }


_RETRYABLE_LOG_ERRORS = ("timed out", "timeout", "limit", "too large", "exceed", "range")


def get_logs_adaptive(rpc: Rpc, direction: str, frm: int, to: int, span_state: dict) -> list[dict]:
    """[frm, to] をスパン分割して取得。エラー時はスパンを自動縮小する。"""
    out: list[dict] = []
    cur = frm
    while cur <= to:
        span = span_state["span"]
        end = min(cur + span - 1, to)
        try:
            res = rpc.call("eth_getLogs", [log_filter(direction, cur, end)], retries=4, timeout=120)
        except RpcError as e:
            msg = str(e).lower()
            if any(k in msg for k in _RETRYABLE_LOG_ERRORS) and span > MIN_SPAN:
                span_state["span"] = max(MIN_SPAN, span // 2)
                log(f"  [adapt] span {span} -> {span_state['span']} ({e})")
                continue
            raise
        out.extend(parse_log(x, direction) for x in res)
        cur = end + 1
        # 順調ならスパンを少しずつ戻す
        if span < MAX_SPAN and len(res) < 3000:
            span_state["span"] = min(MAX_SPAN, int(span * 1.25) + 1)
    return out


def get_raw_logs_adaptive(rpc: Rpc, mkfilter, frm: int, to: int, span_state: dict) -> list[dict]:
    """address 指定の生ログ取得（スパン自動縮小つき）。"""
    out: list[dict] = []
    cur = frm
    while cur <= to:
        span = span_state["span"]
        end = min(cur + span - 1, to)
        try:
            res = rpc.call("eth_getLogs", [mkfilter(cur, end)], retries=4, timeout=120)
        except RpcError as e:
            msg = str(e).lower()
            if any(k in msg for k in _RETRYABLE_LOG_ERRORS) and span > MIN_SPAN:
                span_state["span"] = max(MIN_SPAN, span // 2)
                log(f"  [adapt] span {span} -> {span_state['span']} ({e})")
                continue
            raise
        out.extend(res)
        cur = end + 1
        if span < MAX_SPAN and len(res) < 3000:
            span_state["span"] = min(MAX_SPAN, int(span * 1.25) + 1)
    return out


def parse_bridge_log(item: dict) -> dict:
    t0 = item["topics"][0]
    name = BRIDGE2_EVENT_TOPICS.get(t0, "")
    data = item["data"][2:]
    words = [data[i : i + 64] for i in range(0, len(data), 64)]
    user, amount, nonce = "", float("nan"), ""
    if name == "Deposit":
        user = "0x" + item["topics"][1][-40:]
        if words:
            amount = int(words[0], 16) / 10**USDC_DECIMALS
    elif name in ("RequestedWithdrawal", "FinalizedWithdrawal"):
        # (address indexed user, address destination, uint64 usd, uint64 nonce, bytes32 message, ...)
        user = "0x" + item["topics"][1][-40:]
        if len(words) >= 3:
            amount = int(words[1], 16) / 10**USDC_DECIMALS
            nonce = str(int(words[2], 16))
    return {
        "block_number": int(item["blockNumber"], 16),
        "block_timestamp_utc": "",
        "block_timestamp_ms": 0,
        "tx_hash": item["transactionHash"],
        "log_index": int(item["logIndex"], 16),
        "event": name or "unknown",
        "topic0": t0,
        "user": user,
        "amount_usdc": amount,
        "nonce": nonce,
    }


def partial_path(period: str) -> Path:
    return TMP_DIR / f"native_bridge_events_{period}.partial.csv"


def collect_events(rpc: Rpc, period_key: str, ranges: list[tuple], resume: bool = True) -> pd.DataFrame:
    """
    ranges: [(label, from_block, to_block), ...]
    生ログを partial CSV に追記しつつ、進捗を progress json に残す。
    """
    ppath = partial_path(period_key)
    prog = load_progress()
    pkey = f"{period_key}"
    state = prog.get(pkey, {}) if resume else {}
    if not resume and ppath.exists():
        ppath.unlink()

    rows_existing = []
    if resume and ppath.exists():
        rows_existing = [pd.read_csv(ppath, dtype={"tx_hash": str})]
        log(f"partial 再利用: {ppath.name} ({len(rows_existing[0])} rows)")

    new_rows: list[dict] = []
    header_written = ppath.exists()

    for label, b0, b1 in ranges:
        for direction in ("deposit", "withdraw"):
            skey = f"{label}|{direction}"
            done_to = state.get(skey, b0 - 1)
            if done_to >= b1:
                log(f"{period_key} {skey}: 済み ({b0}..{b1})")
                continue
            start = max(b0, done_to + 1)
            log(f"{period_key} {skey}: blocks {start}..{b1} ({b1 - start + 1:,})")
            span_state = {"span": INITIAL_SPAN}
            chunk_lo = start
            t0 = time.time()
            while chunk_lo <= b1:
                # 進捗保存の粒度（500k ブロックごとに commit）
                chunk_hi = min(chunk_lo + 500_000 - 1, b1)
                got = get_logs_adaptive(rpc, direction, chunk_lo, chunk_hi, span_state)
                if got:
                    df = pd.DataFrame(got, columns=EVENT_COLUMNS)
                    df.to_csv(ppath, mode="a", header=not header_written, index=False)
                    header_written = True
                    new_rows.append(df)
                state[skey] = chunk_hi
                prog[pkey] = state
                save_progress(prog)
                log(
                    f"  {skey} .. {chunk_hi} (+{len(got)} logs, span={span_state['span']}, "
                    f"{time.time() - t0:.0f}s, rpc_calls={rpc.n_calls})"
                )
                chunk_lo = chunk_hi + 1

    frames = rows_existing + new_rows
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["tx_hash", "log_index"]).reset_index(drop=True)
    return df


def attach_timestamps(rpc: Rpc, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    fetch_block_ts_many(rpc, df["block_number"].unique().tolist())
    ts = df["block_number"].map(_ts_cache)
    if ts.isna().any():
        missing = int(ts.isna().sum())
        raise RpcError(f"block timestamp 未取得が {missing} 件残っている")
    df = df.copy()
    df["block_timestamp_ms"] = (ts.astype("int64") * 1000).astype("int64")
    df["block_timestamp_utc"] = pd.to_datetime(ts.astype("int64"), unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    sort_keys = [c for c in ("block_number", "log_index", "direction") if c in df.columns]
    return df.sort_values(sort_keys).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 付録 E の 7 窓の復元
# --------------------------------------------------------------------------- #
def restore_windows() -> pd.DataFrame:
    """
    (a) t4_cctp/ 配下に DepositForBurn 収集スクリプトがあればその窓定義を正とする
        → 本リポジトリには該当スクリプトが存在しない（NOTES 参照）。
    (b) result/T4_cctp/cctp_fast_standard_events.csv の Arbitrum One 行について
        window_id ごとの block_time_utc（および block_number）の min/max を窓境界とする。
        これは t4_cctp/queueing_sim/queueing_sim.py の estimate_lambdas() と同じ扱い。
    """
    if not CCTP_EVENTS.exists():
        raise SystemExit(f"[STOP] {CCTP_EVENTS} が無いため窓を復元できない")
    ev = pd.read_csv(CCTP_EVENTS)
    arb = ev[ev["chain"] == "Arbitrum One"].copy()
    arb["t"] = pd.to_datetime(arb["block_time_utc"], utc=True)
    rows = []
    for wid, g in arb.groupby("window_id"):
        rows.append(
            {
                "window_id": int(wid),
                "start_block": int(g["block_number"].min()),
                "end_block": int(g["block_number"].max()),
                "start_utc": g["t"].min().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_utc": g["t"].max().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_s": float((g["t"].max() - g["t"].min()).total_seconds()),
                "cctp_events": int(len(g)),
            }
        )
    w = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)
    if len(w) != 7:
        raise SystemExit(f"[STOP] 窓が 7 つ復元できなかった（{len(w)} 個）。処理を中止する。")
    return w


# --------------------------------------------------------------------------- #
# サブコマンド
# --------------------------------------------------------------------------- #
def cmd_verify(rpc: Rpc) -> None:
    log("=== 前提の確認 ===")
    print(f"RPC                 : {RPC_URL}")
    print(f"Bridge2 (env)       : {BRIDGE2}")
    print(f"USDC    (env)       : {USDC}")
    code = rpc.call("eth_getCode", [BRIDGE2, "latest"])
    print(f"Bridge2 code size   : {(len(code) - 2) // 2} bytes")
    ucode = rpc.call("eth_getCode", [USDC, "latest"])
    print(f"USDC code size      : {(len(ucode) - 2) // 2} bytes")
    # USDC.balanceOf(Bridge2)
    data = "0x70a08231" + "0" * 24 + BRIDGE2.replace("0x", "")
    bal = rpc.call("eth_call", [{"to": USDC, "data": data}, "latest"])
    print(f"Bridge2 USDC balance: {int(bal, 16) / 10**USDC_DECIMALS:,.2f} USDC")
    # symbol()
    sym = rpc.call("eth_call", [{"to": USDC, "data": "0x95d89b41"}, "latest"])
    raw = bytes.fromhex(sym[2:])
    print(f"USDC symbol()       : {raw[64:64 + int.from_bytes(raw[32:64], 'big')].decode()}")
    latest = int(rpc.call("eth_blockNumber", []), 16)
    print(f"latest block        : {latest}")
    w = restore_windows()
    print("\n=== 付録 E の 7 窓（route (b): cctp_fast_standard_events.csv の Arbitrum One）===")
    print(w.to_string(index=False))


def cmd_events(rpc: Rpc, period: str, resume: bool = True) -> None:
    load_ts_cache()
    if period == "A":
        latest = int(rpc.call("eth_blockNumber", []), 16)
        # 二分探索の探索範囲は T1 実測の既知ブロックから十分広めに取る
        lo, hi = 400_000_000, min(latest, 410_000_000)
        b_start = find_block_by_time(rpc, int(PERIOD_A_START.timestamp()), lo, hi)
        b_end_excl = find_block_by_time(rpc, int(PERIOD_A_END.timestamp()), lo, hi)
        b_end = b_end_excl - 1
        log(f"期間A ブロック範囲: {b_start} .. {b_end} ({b_end - b_start + 1:,} blocks)")
        ranges = [("A", b_start, b_end)]
        df = collect_events(rpc, "2025-11", ranges, resume=resume)
        df = attach_timestamps(rpc, df)
        # 端点の取りこぼし/はみ出しを時刻で厳密化
        t0, t1 = int(PERIOD_A_START.timestamp()) * 1000, int(PERIOD_A_END.timestamp()) * 1000
        df = df[(df["block_timestamp_ms"] >= t0) & (df["block_timestamp_ms"] < t1)].reset_index(drop=True)
        df[EVENT_COLUMNS].to_csv(OUT_EVENTS_A, index=False)
        meta = {
            "period": "A",
            "start_utc": PERIOD_A_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": PERIOD_A_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_block": int(b_start),
            "end_block": int(b_end),
            "start_block_ts": block_ts(rpc, b_start),
            "end_block_ts": block_ts(rpc, b_end),
            "n_deposit": int((df["direction"] == "deposit").sum()),
            "n_withdraw": int((df["direction"] == "withdraw").sum()),
        }
        prog = load_progress()
        prog["meta_A"] = meta
        save_progress(prog)
        log(f"→ {OUT_EVENTS_A.name}: {len(df):,} rows  {meta}")
    else:
        w = restore_windows()
        ranges = [(f"w{int(r.window_id)}", int(r.start_block), int(r.end_block)) for r in w.itertuples()]
        df = collect_events(rpc, "2026-06", ranges, resume=resume)
        df = attach_timestamps(rpc, df)
        # window_id を付与（ブロック範囲で判定）
        wid = pd.Series(pd.NA, index=df.index, dtype="Int64")
        for r in w.itertuples():
            m = (df["block_number"] >= r.start_block) & (df["block_number"] <= r.end_block)
            wid[m] = int(r.window_id)
        df["window_id"] = wid
        df = df[df["window_id"].notna()].reset_index(drop=True)
        df[EVENT_COLUMNS + ["window_id"]].to_csv(OUT_EVENTS_B, index=False)
        prog = load_progress()
        prog["meta_B"] = {"windows": w.to_dict(orient="records"), "n_rows": int(len(df))}
        save_progress(prog)
        log(f"→ {OUT_EVENTS_B.name}: {len(df):,} rows")


def cmd_bridge_events(rpc: Rpc, period: str, resume: bool = True) -> None:
    """（参考系列）Bridge2 コントラクト自身のイベントを収集する。"""
    load_ts_cache()
    if period == "A":
        meta = load_progress().get("meta_A")
        if not meta:
            raise SystemExit("[STOP] 期間A のブロック範囲が未確定。先に events --period A を実行する。")
        ranges = [("A", int(meta["start_block"]), int(meta["end_block"]))]
        out_path, pkey = OUT_BRIDGE_A, "bridge_2025-11"
    else:
        w = restore_windows()
        ranges = [(f"w{int(r.window_id)}", int(r.start_block), int(r.end_block)) for r in w.itertuples()]
        out_path, pkey = OUT_BRIDGE_B, "bridge_2026-06"

    ppath = TMP_DIR / f"{pkey}.partial.csv"
    prog = load_progress()
    state = prog.get(pkey, {}) if resume else {}
    if not resume and ppath.exists():
        ppath.unlink()
    frames = []
    if resume and ppath.exists():
        frames.append(pd.read_csv(ppath, dtype={"tx_hash": str, "nonce": str}))
        log(f"partial 再利用: {ppath.name} ({len(frames[0])} rows)")
    header_written = ppath.exists()

    def mkfilter(a, b):
        return {"fromBlock": hex(a), "toBlock": hex(b), "address": BRIDGE2}

    for label, b0, b1 in ranges:
        done_to = state.get(label, b0 - 1)
        if done_to >= b1:
            log(f"{pkey} {label}: 済み")
            continue
        span_state = {"span": INITIAL_SPAN}
        lo = max(b0, done_to + 1)
        while lo <= b1:
            hi = min(lo + 500_000 - 1, b1)
            got = get_raw_logs_adaptive(rpc, mkfilter, lo, hi, span_state)
            if got:
                df = pd.DataFrame([parse_bridge_log(x) for x in got], columns=BRIDGE_EVENT_COLUMNS)
                df.to_csv(ppath, mode="a", header=not header_written, index=False)
                header_written = True
                frames.append(df)
            state[label] = hi
            prog[pkey] = state
            save_progress(prog)
            log(f"  {pkey} {label} .. {hi} (+{len(got)} logs, span={span_state['span']})")
            lo = hi + 1

    if not frames:
        log(f"{pkey}: 0 rows")
        return
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tx_hash", "log_index"])
    df = attach_timestamps(rpc, df.reset_index(drop=True))
    if period == "A":
        t0 = int(PERIOD_A_START.timestamp()) * 1000
        t1 = int(PERIOD_A_END.timestamp()) * 1000
        df = df[(df["block_timestamp_ms"] >= t0) & (df["block_timestamp_ms"] < t1)].reset_index(drop=True)
        df[BRIDGE_EVENT_COLUMNS].to_csv(out_path, index=False)
    else:
        w = restore_windows()
        wid = pd.Series(pd.NA, index=df.index, dtype="Int64")
        for r in w.itertuples():
            m = (df["block_number"] >= r.start_block) & (df["block_number"] <= r.end_block)
            wid[m] = int(r.window_id)
        df["window_id"] = wid
        df = df[df["window_id"].notna()].reset_index(drop=True)
        df[BRIDGE_EVENT_COLUMNS + ["window_id"]].to_csv(out_path, index=False)
    log(f"→ {out_path.name}: {len(df):,} rows")
    print(df["event"].value_counts().to_string())


def _stats(g: pd.DataFrame, duration_s: float, **extra) -> dict:
    d = {
        "n_events": int(len(g)),
        "duration_s": round(float(duration_s), 3),
        "rate_per_s": (len(g) / duration_s) if duration_s > 0 else float("nan"),
        "mean_amount_usdc": float(g["amount_usdc"].mean()) if len(g) else float("nan"),
        "median_amount_usdc": float(g["amount_usdc"].median()) if len(g) else float("nan"),
        "p90_amount_usdc": float(g["amount_usdc"].quantile(0.90)) if len(g) else float("nan"),
    }
    d.update(extra)
    return d


def cmd_rates() -> None:
    rows = []
    if OUT_EVENTS_A.exists():
        a = pd.read_csv(OUT_EVENTS_A)
        a["t"] = pd.to_datetime(a["block_timestamp_ms"], unit="ms", utc=True)
        total_s = (PERIOD_A_END - PERIOD_A_START).total_seconds()
        for direction in ("deposit", "withdraw"):
            g = a[a["direction"] == direction]
            rows.append(
                _stats(
                    g,
                    total_s,
                    period="A",
                    scope="overall",
                    key="2025-11-27..2025-12-09",
                    direction=direction,
                    start_utc=PERIOD_A_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end_utc=PERIOD_A_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
            )
        a["day"] = a["t"].dt.strftime("%Y-%m-%d")
        for day in sorted(a["day"].unique()):
            for direction in ("deposit", "withdraw"):
                g = a[(a["day"] == day) & (a["direction"] == direction)]
                rows.append(
                    _stats(
                        g,
                        86400.0,
                        period="A",
                        scope="day",
                        key=day,
                        direction=direction,
                        start_utc=f"{day}T00:00:00Z",
                        end_utc=f"{day}T24:00:00Z",
                    )
                )
    if OUT_EVENTS_B.exists():
        b = pd.read_csv(OUT_EVENTS_B)
        w = restore_windows().set_index("window_id")
        for wid in sorted(b["window_id"].unique()):
            dur = float(w.loc[wid, "duration_s"])
            for direction in ("deposit", "withdraw"):
                g = b[(b["window_id"] == wid) & (b["direction"] == direction)]
                rows.append(
                    _stats(
                        g,
                        dur,
                        period="B",
                        scope="window",
                        key=f"w{int(wid)}",
                        direction=direction,
                        start_utc=w.loc[wid, "start_utc"],
                        end_utc=w.loc[wid, "end_utc"],
                    )
                )
        tot = float(w["duration_s"].sum())
        for direction in ("deposit", "withdraw"):
            g = b[b["direction"] == direction]
            rows.append(
                _stats(
                    g,
                    tot,
                    period="B",
                    scope="all_windows",
                    key="7windows_sum",
                    direction=direction,
                    start_utc=w["start_utc"].min(),
                    end_utc=w["end_utc"].max(),
                )
            )
    cols = [
        "period", "scope", "key", "direction", "start_utc", "end_utc",
        "n_events", "duration_s", "rate_per_s",
        "mean_amount_usdc", "median_amount_usdc", "p90_amount_usdc",
    ]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(OUT_RATES, index=False)
    log(f"→ {OUT_RATES.name}: {len(out)} rows")
    print(out.to_string(index=False))


def _count_window(sorted_ts: np.ndarray, lo_ms: int, hi_ms: int) -> int:
    """sorted_ts（昇順 ms）のうち [lo_ms, hi_ms] に入る件数。"""
    return int(np.searchsorted(sorted_ts, hi_ms, side="right") - np.searchsorted(sorted_ts, lo_ms, side="left"))


def cmd_concurrency() -> None:
    if not OUT_EVENTS_A.exists():
        raise SystemExit(f"[STOP] {OUT_EVENTS_A} が無い。先に events --period A を実行する。")
    ev = pd.read_csv(OUT_EVENTS_A)
    ev["tx_hash_l"] = ev["tx_hash"].str.lower()

    dep_meas = pd.read_csv(DEPOSIT_LATENCY)
    wit_meas = pd.read_csv(WITHDRAW_LATENCY)
    own_dep = set(dep_meas["arb_tx_hash"].str.lower())
    own_wit = set(wit_meas["arb_tx_hash"].dropna().str.lower())
    own_all = own_dep | own_wit

    # --- deposit ---
    d_ev = ev[(ev["direction"] == "deposit") & (~ev["tx_hash_l"].isin(own_all))]
    d_ts = np.sort(d_ev["block_timestamp_ms"].to_numpy())
    rows = []
    for _, r in dep_meas.iterrows():
        t = int(float(r["arb_block_timestamp(ms)"]))
        rows.append(
            {
                "experiment_id": r["experiment_id"],
                "arb_tx_hash": r["arb_tx_hash"],
                "arb_block_number": int(r["arb_block_number"]),
                "arb_block_timestamp_ms": t,
                "arb_block_timestamp_utc": datetime.fromtimestamp(t / 1000, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "others_deposit_prev16s": _count_window(d_ts, t - 16_000, t - 1),
                "others_deposit_prev60s": _count_window(d_ts, t - 60_000, t - 1),
                "others_deposit_pm60s": _count_window(d_ts, t - 60_000, t + 60_000),
                "latency_ms": float(r["latency(ms)"]),
            }
        )
    dc = pd.DataFrame(rows)
    dc.to_csv(OUT_DEP_CONC, index=False)
    log(f"→ {OUT_DEP_CONC.name}: {len(dc)} rows")

    # --- withdraw ---
    w_ev = ev[(ev["direction"] == "withdraw") & (~ev["tx_hash_l"].isin(own_all))]
    w_ts = np.sort(w_ev["block_timestamp_ms"].to_numpy())
    rows = []
    for _, r in wit_meas.iterrows():
        t = int(float(r["arb_block_timestamp(ms)"]))
        rows.append(
            {
                "experiment_id": r["experiment_id"],
                "arb_tx_hash": r["arb_tx_hash"],
                "arb_block_number": int(r["arb_block_number"]),
                "arb_block_timestamp_ms": t,
                "arb_block_timestamp_utc": datetime.fromtimestamp(t / 1000, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "others_withdraw_prev200s": _count_window(w_ts, t - 200_000, t - 1),
                "others_withdraw_pm60s": _count_window(w_ts, t - 60_000, t + 60_000),
                "latency_ms": float(r["latency(ms)"]),
            }
        )
    wc = pd.DataFrame(rows)
    wc.to_csv(OUT_WIT_CONC, index=False)
    log(f"→ {OUT_WIT_CONC.name}: {len(wc)} rows")

    # 参考値: Spearman 相関
    from scipy.stats import spearmanr

    rho, p = spearmanr(dc["others_deposit_prev16s"], dc["latency_ms"])
    print(f"\n[Spearman] deposit: others_deposit_prev16s vs latency_ms  r={rho:.4f}  p={p:.4g}  n={len(dc)}")
    rho2, p2 = spearmanr(dc["others_deposit_prev60s"], dc["latency_ms"])
    print(f"[Spearman] deposit: others_deposit_prev60s vs latency_ms  r={rho2:.4f}  p={p2:.4g}")
    rho3, p3 = spearmanr(wc["others_withdraw_prev200s"], wc["latency_ms"])
    print(f"[Spearman] withdraw: others_withdraw_prev200s vs latency_ms r={rho3:.4f}  p={p3:.4g}  n={len(wc)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "task",
        choices=["verify", "events", "bridge-events", "rates", "concurrency", "all"],
        help="実行するタスク",
    )
    ap.add_argument("--period", choices=["A", "B", "both"], default="both", help="events の対象期間")
    ap.add_argument("--rpc", default=RPC_URL, help="Arbitrum One の RPC エンドポイント")
    ap.add_argument("--no-resume", action="store_true", help="partial を捨てて最初から取得し直す")
    args = ap.parse_args()

    rpc = Rpc(args.rpc)
    resume = not args.no_resume

    if args.task == "verify":
        cmd_verify(rpc)
    elif args.task == "events":
        for p in (["A", "B"] if args.period == "both" else [args.period]):
            cmd_events(rpc, p, resume=resume)
    elif args.task == "bridge-events":
        for p in (["A", "B"] if args.period == "both" else [args.period]):
            cmd_bridge_events(rpc, p, resume=resume)
    elif args.task == "rates":
        cmd_rates()
    elif args.task == "concurrency":
        cmd_concurrency()
    else:
        cmd_verify(rpc)
        cmd_events(rpc, "A", resume=resume)
        cmd_events(rpc, "B", resume=resume)
        cmd_rates()
        cmd_concurrency()
    log(f"done (rpc calls = {rpc.n_calls})")


if __name__ == "__main__":
    main()
