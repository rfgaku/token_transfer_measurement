#!/usr/bin/env python3
"""
t4_cctp/analysis/finality_gap.py

T4 CCTP 追加検証：新規メトリクス「Finality Gap G ／ 連続Finality関数 F(t)」の実証。
各 CCTP 転送の soft→safe→hard ファイナリティ時刻を両方向で取得し、L + G ≈ τ_F を実データで示す。

定義（task準拠）:
  deposit（source=Arbitrum, dest=HyperEVM, Fast）:
    t_burn   = t1_arb_burn_block_ts            （source burn の L2 ブロック時刻）
    t_usable = t2_5_hevm_mint_block_ts         （dest で使える＝forwarder mint）
    t_safe   = burn を含む Sequencer バッチが L1 投稿されたブロックの timestamp
               （SequencerInbox の SequencerBatchDelivered を burn の実L1ブロック以降で対応付け。
                enrich_l1.py の突合を 40→210 へ拡張。最初の対応バッチ＝下界）
    t_hard   = その L1 投稿ブロックが Ethereum で finalized になる時刻
               （主: t_safe + 2エポック=768s の下界。副: epoch整列 finalize 推定も併記）
    L = t_usable - t_burn,  τ_F = t_hard - t_burn,  G = t_hard - t_usable  （恒等式 τ_F = L + G）
  withdraw（source=HyperEVM, dest=Arbitrum, Standard/Finalized）:
    HyperBFT は決定的BFT即時 hard finality → source hard ≈ t_burn。
    t_burn=t1_hevm_burn_block_ts, t_usable=t3_arb_mint_block_ts。
    τ_F_source = 0,  G = max(0, t_hard_source - t_usable) ≈ 0（usable 時点で source 既に不可逆）。

入力（読み取り専用・不変）:
  result/T4_cctp/deposit_l1_enriched.csv   （burn_rt_l1_block 等を流用）
  result/T4_cctp/withdraw_cctp_latency.csv
出力（新規のみ）:
  result/T4_cctp/finality_timeline.csv
  result/T4_cctp/findings_finality_gap.md（別途記述）

RPC:
  Ethereum L1 : 公開 https://ethereum-rpc.publicnode.com（enrich_l1.py と同じ・read-only）
  Arbitrum    : 既存 CSV の block ts を使用（再取得不要）
  HyperEVM    : 即時 finality は protocol property のため追加 RPC 不要

使い方:
  サンプル: python3 -u t4_cctp/analysis/finality_gap.py --sample-only
  本実行  : python3 -u t4_cctp/analysis/finality_gap.py
  (公開RPCは sandbox 無効で実行。長時間は run_in_background + tee)
"""
import argparse
import bisect
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ENR_CSV = REPO / "result" / "T4_cctp" / "deposit_l1_enriched.csv"
WIT_CSV = REPO / "result" / "T4_cctp" / "withdraw_cctp_latency.csv"
OUT_CSV = REPO / "result" / "T4_cctp" / "finality_timeline.csv"

L1_RPC = "https://ethereum-rpc.publicnode.com"
SEQ_INBOX = "0x1c479675ad559DC151F6Ec7ed3FbF8ceE79582B6"
TOPIC_BATCH = "0x7394f4a19a13c7b92b5bb71033245305946ef78452f7b4986ac1390b5df4ebd7"  # SequencerBatchDelivered
UA = "Mozilla/5.0 (research; cctp-finality-gap)"
SLEEP = 0.08

# Ethereum mainnet beacon chain genesis + finality constants
ETH_GENESIS = 1606824023
SECS_PER_SLOT = 12
SLOTS_PER_EPOCH = 32
EPOCH_SECS = SECS_PER_SLOT * SLOTS_PER_EPOCH  # 384s
TWO_EPOCHS = 2 * EPOCH_SECS                   # 768s = 12.8min (hard-finality lower bound)


class RpcError(Exception):
    pass


def _rpc(method, params, _try=0):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(L1_RPC, data=body, headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.load(r)
        if "error" in d:
            raise RpcError(str(d["error"]))
        return d["result"]
    except RpcError:
        raise
    except Exception as e:
        if _try < 4:
            time.sleep(0.5 * (2 ** _try))
            return _rpc(method, params, _try + 1)
        raise RpcError(f"{method}: {e}")


_ts_cache = {}


def l1_ts(blk):
    if blk in _ts_cache:
        return _ts_cache[blk]
    b = _rpc("eth_getBlockByNumber", [hex(blk), False])
    time.sleep(SLEEP)
    if not b:
        raise RpcError(f"l1 block {blk} null")
    ts = int(b["timestamp"], 16)
    _ts_cache[blk] = ts
    return ts


def sweep_batches(from_blk, to_blk, chunk=600):
    """SequencerBatchDelivered の投稿 L1 ブロック番号を昇順 unique で収集（ts は後で必要分だけ取得）。"""
    blocks = set()
    b = from_blk
    n_calls = 0
    while b <= to_blk:
        hi = min(b + chunk - 1, to_blk)
        try:
            logs = _rpc("eth_getLogs", [{"address": SEQ_INBOX, "topics": [TOPIC_BATCH],
                                         "fromBlock": hex(b), "toBlock": hex(hi)}])
            time.sleep(SLEEP)
            for l in logs:
                blocks.add(int(l["blockNumber"], 16))
            n_calls += 1
            if n_calls % 20 == 0:
                print(f"  [sweep] {b}..{hi} cumulative_batches={len(blocks)}", flush=True)
            b = hi + 1
        except RpcError as e:
            if chunk > 50:
                chunk //= 2  # result-size/limit → 細分化リトライ
                continue
            raise
    return sorted(blocks)


def epoch_aligned_finalize(ts):
    """ts を含む L1 ブロックの属する epoch e に対し、epoch (e+2) 開始時刻 = 現実的 finalize 推定。"""
    slot = (ts - ETH_GENESIS) // SECS_PER_SLOT
    epoch = slot // SLOTS_PER_EPOCH
    return ETH_GENESIS + (epoch + 2) * EPOCH_SECS


FIELDS = [
    "direction", "experiment_id", "group",
    "t_burn_ms", "t_usable_ms", "t_safe_ms", "t_hard_lb_ms", "t_hard_epoch_ms",
    "L_ms", "tau_F_lb_ms", "G_lb_ms", "G_epoch_ms",
    "safe_l1_block", "safe_lag_s", "method", "err",
]


def do_deposit(out, batch_blocks):
    enr = list(csv.DictReader(open(ENR_CSV)))
    print(f"[deposit] n={len(enr)}  batches_in_sweep={len(batch_blocks)}", flush=True)
    for i, r in enumerate(enr):
        eid = r["experiment_id"]
        g = r.get("group", "")
        err = ""
        rowout = {"direction": "deposit", "experiment_id": eid, "group": g,
                  "method": "t_safe=first SequencerBatchDelivered>=burn_rt_l1; t_hard_lb=t_safe+2epoch(768s)"}
        try:
            t_burn = int(r["t1_arb_burn_block_ts(ms)"])
            t_usable = int(r["t2_5_hevm_mint_block_ts(ms)"])
            anchor = int(r["burn_rt_l1_block"])
            # 最初の対応バッチ（>= burn 実L1ブロック）
            idx = bisect.bisect_left(batch_blocks, anchor)
            if idx >= len(batch_blocks):
                raise RpcError("no batch >= anchor in sweep range")
            safe_blk = batch_blocks[idx]
            t_safe_s = l1_ts(safe_blk)
            t_safe_ms = t_safe_s * 1000
            t_hard_lb_ms = t_safe_ms + TWO_EPOCHS * 1000
            t_hard_epoch_ms = epoch_aligned_finalize(t_safe_s) * 1000
            rowout.update({
                "t_burn_ms": t_burn, "t_usable_ms": t_usable, "t_safe_ms": t_safe_ms,
                "t_hard_lb_ms": t_hard_lb_ms, "t_hard_epoch_ms": t_hard_epoch_ms,
                "L_ms": t_usable - t_burn,
                "tau_F_lb_ms": t_hard_lb_ms - t_burn,
                "G_lb_ms": t_hard_lb_ms - t_usable,
                "G_epoch_ms": t_hard_epoch_ms - t_usable,
                "safe_l1_block": safe_blk,
                "safe_lag_s": t_safe_s - t_burn // 1000,
            })
        except Exception as e:
            err = repr(e)[:140]
        rowout["err"] = err
        out.append(rowout)
        if (i + 1) % 25 == 0:
            print(f"  [deposit] {i+1}/{len(enr)}", flush=True)


def do_withdraw(out):
    wit = list(csv.DictReader(open(WIT_CSV)))
    print(f"[withdraw] n={len(wit)}  (source=HyperEVM, HyperBFT instant hard finality)", flush=True)
    for r in wit:
        eid = r["experiment_id"]
        err = ""
        rowout = {"direction": "withdraw", "experiment_id": eid, "group": "",
                  "method": "source=HyperEVM HyperBFT deterministic instant finality → t_hard_source=t_burn"}
        try:
            t_burn = int(r["t1_hevm_burn_block_ts(ms)"])     # source burn (HyperEVM)
            t_usable = int(r["t3_arb_mint_block_ts(ms)"])    # dest usable (Arbitrum mint)
            t_safe_ms = t_burn      # 即時 safe=hard
            t_hard_ms = t_burn
            rowout.update({
                "t_burn_ms": t_burn, "t_usable_ms": t_usable, "t_safe_ms": t_safe_ms,
                "t_hard_lb_ms": t_hard_ms, "t_hard_epoch_ms": t_hard_ms,
                "L_ms": t_usable - t_burn,
                "tau_F_lb_ms": t_hard_ms - t_burn,            # = 0 (source instant)
                "G_lb_ms": max(0, t_hard_ms - t_usable),      # = 0 (usable後に source 既に final)
                "G_epoch_ms": max(0, t_hard_ms - t_usable),
                "safe_l1_block": "", "safe_lag_s": 0,
            })
        except Exception as e:
            err = repr(e)[:140]
        rowout["err"] = err
        out.append(rowout)


def sample_only():
    enr = list(csv.DictReader(open(ENR_CSV)))
    print("=== deposit サンプル: 最初の対応バッチ突合（生RPC） ===")
    for r in enr[:5]:
        anchor = int(r["burn_rt_l1_block"])
        logs = _rpc("eth_getLogs", [{"address": SEQ_INBOX, "topics": [TOPIC_BATCH],
                                     "fromBlock": hex(anchor), "toBlock": hex(anchor + 80)}])
        time.sleep(SLEEP)
        if not logs:
            print(f"  id={r['experiment_id']} anchor={anchor} NO BATCH in +80")
            continue
        safe_blk = min(int(l["blockNumber"], 16) for l in logs)
        t_safe = l1_ts(safe_blk)
        t_burn = int(r["t1_arb_burn_block_ts(ms)"]) // 1000
        t_usable = int(r["t2_5_hevm_mint_block_ts(ms)"]) // 1000
        t_hard_lb = t_safe + TWO_EPOCHS
        print(f"  id={r['experiment_id']:>3} grp={r['group']:5} anchor_l1={anchor} safe_l1={safe_blk} "
              f"| t_burn={t_burn} t_usable={t_usable} t_safe={t_safe} t_hard_lb={t_hard_lb} "
              f"| L={t_usable-t_burn}s safe_lag={t_safe-t_burn}s G_lb={t_hard_lb-t_usable}s tauF_lb={t_hard_lb-t_burn}s")
    print("\n=== withdraw サンプル: source=HyperEVM 即時finality（既存ts使用・追加RPC無し） ===")
    wit = list(csv.DictReader(open(WIT_CSV)))
    for r in wit[:5]:
        t_burn = int(r["t1_hevm_burn_block_ts(ms)"]) // 1000
        t_usable = int(r["t3_arb_mint_block_ts(ms)"]) // 1000
        print(f"  id={r['experiment_id']:>3} t_burn(hevm)={t_burn} t_usable(arb)={t_usable} fte={r['finalityThresholdExecuted']} "
              f"| L={t_usable-t_burn}s  tauF_source=0  G=max(0,{t_burn-t_usable})=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    if args.sample_only:
        sample_only()
        return

    enr = list(csv.DictReader(open(ENR_CSV)))
    anchors = [int(r["burn_rt_l1_block"]) for r in enr if r["burn_rt_l1_block"] not in ("", "N/A")]
    lo, hi = min(anchors) - 2, max(anchors) + 200
    t0 = time.time()
    print(f"[sweep] SequencerBatchDelivered over L1 [{lo}..{hi}] ({hi-lo} blocks)", flush=True)
    batch_blocks = sweep_batches(lo, hi)
    print(f"[sweep] done: {len(batch_blocks)} unique batch blocks in {time.time()-t0:.0f}s", flush=True)

    out = []
    do_deposit(out, batch_blocks)
    do_withdraw(out)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    errs = sum(1 for r in out if r.get("err"))
    print(f"\nWROTE {OUT_CSV}  ({len(out)} rows, err={errs})  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
