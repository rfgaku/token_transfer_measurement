#!/usr/bin/env python3
"""
t4_cctp/analysis/enrich_l1.py

T4 CCTP deposit「12確認 vs 32確認」二峰性の根本原因を、オンチェーンデータで
実証検証するための L1 付加スクリプト（フェーズ2）。

入力 (読み取り専用):  result/T4_cctp/deposit_cctp_latency.csv
出力 (新規):          result/T4_cctp/deposit_l1_enriched.csv

各行の arb_burn_tx_hash / t1_arb_burn_block_number をキーに、Arbitrum(L2) と
Ethereum(L1) の生 RPC から以下を付加する（捏造禁止・取得不能は空欄＋err列）:

  Arbitrum L2 から:
    arb_burn_l1_block          = burn の L2 ブロックの l1BlockNumber (Nitro 固有)
    att_l2_block               = burn_block + arb_confirmations_at_attestation
    att_l1_block               = att_l2_block の l1BlockNumber
    dL1_blocks                 = att_l1_block - arb_burn_l1_block  (= ΔL1, 確認ベース)
  Ethereum L1 から (上記 L1 ブロック番号→ Ethereum 側 timestamp):
    arb_burn_l1_block_ts(ms)   = arb_burn_l1_block の L1 block.timestamp*1000
    att_l1_block_ts(ms)        = att_l1_block の L1 block.timestamp*1000
    dL1_seconds(ms)            = att_l1_block_ts - arb_burn_l1_block_ts
                                 (burn→attestation の L1 時計での経過。系間 skew 不在)
    t2_l1_block                = attestation PC 時刻(t2) を覆う L1 block (timestamp 逆引き)
    t2_l1_block_ts(ms)         = その L1 block の timestamp*1000
    dL1_blocks_byT2            = t2_l1_block - arb_burn_l1_block
    t2_phase_in_l1(ms)         = t2 - t2_l1_block_ts  (attestation が L1 ブロック境界の
                                 どの位相で起きたか。L1 同期/境界整列仮説の検定用)
  派生:
    group                      = fast(conf<=20) / slow(20<conf<=45) / tail(その他)

使い方:
  サンプル:  python3 -u t4_cctp/analysis/enrich_l1.py --ids 1,103,3,107,19
  全件:      python3 -u t4_cctp/analysis/enrich_l1.py
  (公開 RPC は sandbox 無効で実行すること。長時間は run_in_background + tee)
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SRC_CSV = REPO / "result" / "T4_cctp" / "deposit_cctp_latency.csv"
OUT_CSV = REPO / "result" / "T4_cctp" / "deposit_l1_enriched.csv"

ARB_RPC = "https://arb1.arbitrum.io/rpc"
L1_RPC = "https://ethereum-rpc.publicnode.com"
ARB_FALLBACK = ["https://arbitrum-one-rpc.publicnode.com", "https://arbitrum.drpc.org"]
UA = "Mozilla/5.0 (research; cctp-latency-analysis)"
SLEEP = 0.06  # ~16 req/s soft throttle


class RpcError(Exception):
    pass


def _rpc(url, method, params, _try=0):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        if "error" in d:
            raise RpcError(str(d["error"]))
        return d["result"]
    except Exception as e:
        if _try < 4:
            time.sleep(0.4 * (2 ** _try))
            return _rpc(url, method, params, _try + 1)
        raise RpcError(f"{method}{params}: {e}")


# --- Arbitrum L2 helpers (with endpoint fallback) ---
_arb_urls = [ARB_RPC] + ARB_FALLBACK


def arb_call(method, params):
    last = None
    for u in _arb_urls:
        try:
            res = _rpc(u, method, params)
            time.sleep(SLEEP)
            return res
        except RpcError as e:
            last = e
    raise last


_arb_block_cache = {}


def arb_block(num):
    """getBlockByNumber -> dict with l1BlockNumber(int), timestamp(int)."""
    if num in _arb_block_cache:
        return _arb_block_cache[num]
    b = arb_call("eth_getBlockByNumber", [hex(num), False])
    if not b:
        raise RpcError(f"arb block {num} null")
    out = {
        "l1": int(b["l1BlockNumber"], 16) if b.get("l1BlockNumber") else None,
        "ts": int(b["timestamp"], 16),
    }
    _arb_block_cache[num] = out
    return out


# --- Ethereum L1 helpers + timestamp reverse-lookup ---
_l1_ts_cache = {}  # blknum -> ts(sec)


def l1_ts(blk):
    if blk in _l1_ts_cache:
        return _l1_ts_cache[blk]
    b = _rpc(L1_RPC, "eth_getBlockByNumber", [hex(blk), False])
    time.sleep(SLEEP)
    if not b:
        raise RpcError(f"l1 block {blk} null")
    ts = int(b["timestamp"], 16)
    _l1_ts_cache[blk] = ts
    return ts


def l1_block_at_time(t_sec):
    """Return L1 block b such that ts(b) <= t_sec < ts(b+1). Uses cached anchors + ~12s/block estimate, then steps."""
    if not _l1_ts_cache:
        raise RpcError("no L1 anchor cached yet")
    # nearest cached anchor
    anchor = min(_l1_ts_cache, key=lambda b: abs(_l1_ts_cache[b] - t_sec))
    b = anchor + int(round((t_sec - _l1_ts_cache[anchor]) / 12.0))
    # ensure ts(b) <= t
    while l1_ts(b) > t_sec:
        b -= 1
    # advance while ts(b+1) <= t
    while l1_ts(b + 1) <= t_sec:
        b += 1
    return b


def enrich_row(r):
    out = {}
    err = []
    try:
        burn_blk = int(r["t1_arb_burn_block_number"])
        conf_s = r["arb_confirmations_at_attestation(blocks)"]
        t2_ms = r["t2_iris_attestation_complete(ms)"]
        # group
        conf = float(conf_s) if conf_s not in ("", "N/A") else None
        if conf is None:
            grp = "unknown"
        elif conf <= 20:
            grp = "fast"
        elif conf <= 45:
            grp = "slow"
        else:
            grp = "tail"
        out["group"] = grp

        bb = arb_block(burn_blk)
        out["arb_burn_l1_block"] = bb["l1"]
        out["arb_burn_l1_block_ts(ms)"] = bb["ts"] and l1_ts(bb["l1"]) * 1000
        # real L1 block whose time window contains the burn's real time (L2 block ts).
        # arb_burn_l1_block is what the sequencer *referenced* (lags real L1 head).
        l1_ts(bb["l1"])  # seed anchor near this time
        burn_rt_blk = l1_block_at_time(bb["ts"])
        out["burn_rt_l1_block"] = burn_rt_blk
        out["burn_l1_lag_blocks"] = (burn_rt_blk - bb["l1"]) if bb["l1"] is not None else ""

        att_l1 = None
        if conf is not None:
            att_blk = burn_blk + int(round(conf))
            out["att_l2_block"] = att_blk
            ab = arb_block(att_blk)
            att_l1 = ab["l1"]
            out["att_l1_block"] = att_l1
            out["att_l1_block_ts(ms)"] = l1_ts(att_l1) * 1000
            out["dL1_blocks"] = (att_l1 - bb["l1"]) if (att_l1 is not None and bb["l1"] is not None) else ""
            if att_l1 is not None and bb["l1"] is not None:
                out["dL1_seconds(ms)"] = (l1_ts(att_l1) - l1_ts(bb["l1"])) * 1000
            else:
                out["dL1_seconds(ms)"] = ""
        else:
            out["att_l2_block"] = out["att_l1_block"] = out["att_l1_block_ts(ms)"] = ""
            out["dL1_blocks"] = out["dL1_seconds(ms)"] = ""

        # t2 -> L1 block reverse lookup (need an anchor; seed from burn L1 block)
        if t2_ms not in ("", "N/A", "0"):
            l1_ts(bb["l1"])  # ensure anchor cached near this time
            if att_l1 is not None:
                l1_ts(att_l1)
            t2_blk = l1_block_at_time(int(t2_ms) / 1000.0)
            out["t2_l1_block"] = t2_blk
            t2blk_ts = l1_ts(t2_blk)
            out["t2_l1_block_ts(ms)"] = t2blk_ts * 1000
            out["dL1_blocks_byT2"] = (t2_blk - bb["l1"]) if bb["l1"] is not None else ""
            out["dL1_realblocks_byT2"] = t2_blk - burn_rt_blk  # clean: real L1 blocks Iris waited
            out["t2_phase_in_l1(ms)"] = int(t2_ms) - t2blk_ts * 1000
        else:
            out["t2_l1_block"] = out["t2_l1_block_ts(ms)"] = ""
            out["dL1_blocks_byT2"] = out["dL1_realblocks_byT2"] = out["t2_phase_in_l1(ms)"] = ""
    except Exception as e:
        err.append(repr(e)[:160])
    out["l1_enrich_err"] = "; ".join(err)
    return out


NEW_COLS = [
    "group",
    "arb_burn_l1_block", "arb_burn_l1_block_ts(ms)",
    "burn_rt_l1_block", "burn_l1_lag_blocks",
    "att_l2_block", "att_l1_block", "att_l1_block_ts(ms)",
    "dL1_blocks", "dL1_seconds(ms)",
    "t2_l1_block", "t2_l1_block_ts(ms)", "dL1_blocks_byT2", "dL1_realblocks_byT2", "t2_phase_in_l1(ms)",
    "l1_enrich_err",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma list of experiment_id for sample run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--sample-only", action="store_true", help="print sample rows, do not write full csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC_CSV)))
    src_cols = list(rows[0].keys())
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["experiment_id"] in want]
    if args.limit:
        rows = rows[: args.limit]

    enriched = []
    t0 = time.time()
    for i, r in enumerate(rows):
        e = enrich_row(r)
        enriched.append({**r, **e})
        flag = "OK" if not e["l1_enrich_err"] else "ERR:" + e["l1_enrich_err"]
        print(f"[{i+1}/{len(rows)}] id={r['experiment_id']:>3} grp={e.get('group'):5} "
              f"conf={r['arb_confirmations_at_attestation(blocks)']:>4} iris_wait={r['iris_wait(ms)']:>6} "
              f"L1burn={e.get('arb_burn_l1_block')} L1att={e.get('att_l1_block')} "
              f"dL1={e.get('dL1_blocks')} dL1s={e.get('dL1_seconds(ms)')} "
              f"lag={e.get('burn_l1_lag_blocks')} t2L1blk={e.get('t2_l1_block')} "
              f"dL1byT2={e.get('dL1_blocks_byT2')} dL1RTbyT2={e.get('dL1_realblocks_byT2')} "
              f"t2phase={e.get('t2_phase_in_l1(ms)')} {flag}", flush=True)

    if not args.sample_only:
        out_cols = src_cols + [c for c in NEW_COLS if c not in src_cols]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_cols)
            w.writeheader()
            for r in enriched:
                w.writerow({k: r.get(k, "") for k in out_cols})
        print(f"\nWROTE {args.out}  ({len(enriched)} rows, {len(out_cols)} cols)  in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"\nSAMPLE done ({len(enriched)} rows) in {time.time()-t0:.1f}s — no file written", flush=True)


if __name__ == "__main__":
    main()
