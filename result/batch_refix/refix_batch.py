#!/usr/bin/env python3
"""
refix_batch.py  —  t_safe の厳密再対応づけ（ACM DLT 論文 / 露出窓 G 再構成）

目的
----
着金後未確定期間 G の計算に使う t_safe（送金を含む Sequencer バッチが Ethereum L1 へ
投稿された時刻）を、

  旧: 「t_1 を含む実 L1 ブロック(anchor)以降で最初の SequencerBatchDelivered」
      = バースト時点以降の最初のバッチ、というヒューリスティック（= 下界）
  新: Arbitrum NodeInterface.findBatchContainingBlock(uint64 blockNum) による
      「その L2 ブロックを実際に含むバッチ」の厳密対応づけ

に置き換え、送金ごとの新しい t_safe を求める。
統計量の確定と論文反映は本スクリプトの範囲外（データ取得＋整合性チェックまで）。

方式
----
Phase 1: 327 件（CCTP deposit 210 + Native Bridge deposit 117）の Arbitrum L2
         ブロック番号について Arbitrum One RPC へ eth_call:
           to   = 0x00000000000000000000000000000000000000C8 (NodeInterface)
           data = selector(findBatchContainingBlock(uint64)) = 0x81f1adaf
                  + uint64 blockNum (32byte 右詰め)
         → batch sequence number を得る。L2 ブロック番号は重複排除して呼ぶ。
Phase 2: 得たバッチ番号を Ethereum mainnet の SequencerInbox
         0x1c479675ad559DC151F6Ec7ed3FbF8ceE79582B6 の
         SequencerBatchDelivered (topic0=0x7394f4a1..., topic1=indexed
         batchSequenceNumber) と突合し、投稿 L1 ブロック番号／ブロック時刻（秒）／
         取引ハッシュを得る。
         2a: 各送金の「旧バッチ L1 ブロック −50 〜 +1,000」区間を結合(merge)して
             topic0 のみで一括スイープし、seq → (l1_block, l1_tx) 表を作る。
             （指定区間の和集合を必ず覆う。1 件ずつ topic1 で引くのと同じ結果で、
               RPC 呼び出し回数のみ削減）
         2b: 2a で見つからなかった seq は topic1 で絞った個別検索を、
             ±1,000 → ±2,000 → ±4,000 … と範囲を倍々に広げて再試行。
Phase 3: new_l1_block の timestamp を eth_getBlockByNumber で取得（重複排除）。
Phase 4: batch_refix.csv（1 行 1 送金・327 行）と summary.txt（整合性チェック）を出力。

入力（読み取り専用。本スクリプトは入力を一切変更しない）
  result/T4_cctp/deposit_cctp_latency.csv   CCTP deposit 210 件
  result/T4_cctp/finality_timeline.csv      direction=='deposit' の 210 行（旧 t_safe）
  result/deposit_latency.csv                Native deposit 117 件
  result/deposit_t1_l1_enriched.csv         Native 117 件（旧 batch_l1_block/ts）

出力（すべて本スクリプトと同じディレクトリ）
  batch_refix.csv, summary.txt, run.log, fbcb_cache.json

アドレス／イベント定義の出典
  SequencerInbox アドレス・SequencerBatchDelivered topic0 は既存スクリプト
    t4_cctp/analysis/finality_gap.py
    experiments/t1_deposit_finality_gap.py
  から流用（両者一致。かつ task 指定の 0x1c4796...82B6 とも一致）。
  topic1 = indexed batchSequenceNumber であることは実ログで検証（run.log 参照）。

RPC 環境変数（既存スクリプトと同じ名前を流用。値はログ・出力に書かない）
  ARBITRUM_HTTP_RPC … Arbitrum One（.env。既存 t4_cctp/* と同じキー）
  ETH_RPC_URL        … Ethereum mainnet（任意。experiments/t1_deposit_finality_gap.py と同じキー）

使い方
  python3 -u refix_batch.py --probe       # 疎通・ABI 検証のみ
  python3 -u refix_batch.py --sample 5    # 先頭 5 件（ファイル出力なし）
  python3 -u refix_batch.py               # 本実行
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # result/batch_refix_20261001 -> result -> repo root

IN_CCTP = REPO / "result" / "T4_cctp" / "deposit_cctp_latency.csv"
IN_CCTP_OLD = REPO / "result" / "T4_cctp" / "finality_timeline.csv"
IN_NAT = REPO / "result" / "deposit_latency.csv"
IN_NAT_OLD = REPO / "result" / "deposit_t1_l1_enriched.csv"

OUT_CSV = HERE / "batch_refix.csv"
OUT_SUM = HERE / "summary.txt"
OUT_LOG = HERE / "run.log"
CACHE = HERE / "fbcb_cache.json"

# ---- chain constants ----
NODE_INTERFACE = "0x00000000000000000000000000000000000000C8"
SEL_FBCB = "0x81f1adaf"  # keccak("findBatchContainingBlock(uint64)")[:4]
SEQ_INBOX = "0x1c479675ad559DC151F6Ec7ed3FbF8ceE79582B6"  # Arbitrum One SequencerInbox
TOPIC_BATCH = "0x7394f4a19a13c7b92b5bb71033245305946ef78452f7b4986ac1390b5df4ebd7"  # SequencerBatchDelivered

WIN_BACK, WIN_FWD = 50, 1000  # 旧バッチ L1 ブロックからの探索窓（task 指定）
SWEEP_CHUNK = 8000            # topic0 スイープのチャンク幅（tenderly は 8k まで実測OK）
UA = "Mozilla/5.0 (research; acm-dlt-batch-refix)"
SLEEP = 0.05

# ---- endpoint pools（env 由来 URL は値をログに出さない） ----
_ENV_ARB = None   # .env / 環境変数から読む
_ENV_ETH = None


def _mask(url):
    if _ENV_ARB and url == _ENV_ARB:
        return "<env:ARBITRUM_HTTP_RPC>"
    if _ENV_ETH and url == _ENV_ETH:
        return "<env:ETH_RPC_URL>"
    return url


def load_env():
    """.env は読むだけ。値はログ・出力に書かない。"""
    global _ENV_ARB, _ENV_ETH
    _ENV_ARB = os.environ.get("ARBITRUM_HTTP_RPC")
    _ENV_ETH = os.environ.get("ETH_RPC_URL")
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "ARBITRUM_HTTP_RPC" and not _ENV_ARB:
                _ENV_ARB = v
            if k == "ETH_RPC_URL" and not _ENV_ETH:
                _ENV_ETH = v


def _dedupe(urls):
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def arb_urls():
    return _dedupe([
        _ENV_ARB,
        "https://arb1.arbitrum.io/rpc",
        "https://arbitrum-one-rpc.publicnode.com",
        "https://arbitrum.drpc.org",
    ])


def l1_log_urls():
    # アーカイブ範囲の eth_getLogs に応答する endpoint のみ（実測で選定。run.log 参照）
    return _dedupe([
        _ENV_ETH,
        "https://mainnet.gateway.tenderly.co",
        "https://gateway.tenderly.co/public/mainnet",
    ])


def l1_block_urls():
    return _dedupe([
        _ENV_ETH,
        "https://mainnet.gateway.tenderly.co",
        "https://ethereum-rpc.publicnode.com",
        "https://gateway.tenderly.co/public/mainnet",
    ])


STATS = {"rpc_calls": 0, "retries": 0, "failover": 0, "chunk_shrinks": 0,
         "fbcb_calls": 0, "getlogs_calls": 0, "getblock_calls": 0, "cache_hits": 0}
LOGF = None


def say(msg):
    print(msg, flush=True)
    if LOGF:
        LOGF.write(msg + "\n")
        LOGF.flush()


class RpcError(Exception):
    pass


def _post(url, method, params, timeout=60):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", "User-Agent": UA})
    STATS["rpc_calls"] += 1
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise RpcError(f"HTTP{e.code} {e.read()[:160].decode('utf8', 'replace')}")
    if "error" in d:
        raise RpcError(json.dumps(d["error"])[:200])
    return d["result"]


def rpc(method, params, urls, tries_per_url=4, timeout=60):
    """endpoint フェイルオーバ + 指数バックオフ（0.6/1.2/2.4/4.8s）。"""
    last = None
    for ui, u in enumerate(urls):
        for t in range(tries_per_url):
            try:
                res = _post(u, method, params, timeout)
                time.sleep(SLEEP)
                return res
            except Exception as e:
                s = repr(e)
                last = RpcError(f"{_mask(u)} {method}: {s[:200]}")
                # その endpoint では原理的に無理 → 即次へ
                if any(k in s for k in ("Archive requests require", "Method not found",
                                        "method not available", "Unauthorized",
                                        "must authenticate", "token is inva")):
                    break
                # 範囲が広すぎる系は呼び出し側が chunk を縮めて再試行する
                if any(k in s for k in ("range too large", "ranges o", "limited to",
                                        "-32602", "-32600", "-32047", "-32001")):
                    raise last
                if t < tries_per_url - 1:
                    STATS["retries"] += 1
                    time.sleep(0.6 * (2 ** t))
        if ui < len(urls) - 1:
            STATS["failover"] += 1
    raise last


# ---------------- Phase 1: findBatchContainingBlock ----------------
_fbcb = {}   # l2_block(int) -> batch_seq(int)


def load_cache():
    if CACHE.exists():
        try:
            d = json.loads(CACHE.read_text())
            _fbcb.update({int(k): v for k, v in d.get("fbcb", {}).items()})
            say(f"[cache] loaded fbcb={len(_fbcb)} from {CACHE.name}")
        except Exception as e:
            say(f"[cache] ignore ({e!r})")


def save_cache():
    CACHE.write_text(json.dumps({"fbcb": {str(k): v for k, v in _fbcb.items()}}))


def find_batch(l2_block, use_cache=True):
    if use_cache and l2_block in _fbcb:
        STATS["cache_hits"] += 1
        return _fbcb[l2_block]
    data = SEL_FBCB + "%064x" % l2_block
    res = rpc("eth_call", [{"to": NODE_INTERFACE, "data": data}, "latest"], arb_urls())
    STATS["fbcb_calls"] += 1
    if not res or res == "0x":
        raise RpcError(f"findBatchContainingBlock({l2_block}) returned empty")
    seq = int(res, 16)
    if use_cache:
        _fbcb[l2_block] = seq
        if len(_fbcb) % 50 == 0:
            save_cache()
    return seq


# ---------------- Phase 2: SequencerBatchDelivered 突合 ----------------
_seq2l1 = {}   # batch_seq -> (l1_block, l1_tx)


def merge_windows(centers):
    """[c-WIN_BACK, c+WIN_FWD] の和集合を昇順の非重複区間へ。"""
    iv = sorted((c - WIN_BACK, c + WIN_FWD) for c in centers)
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(x) for x in out]


def sweep_logs(lo, hi, chunk=SWEEP_CHUNK):
    """[lo,hi] の SequencerBatchDelivered を topic0 で一括取得し _seq2l1 を埋める。"""
    b = lo
    n_found = 0
    while b <= hi:
        top = min(b + chunk - 1, hi)
        try:
            logs = rpc("eth_getLogs", [{"address": SEQ_INBOX, "topics": [TOPIC_BATCH],
                                        "fromBlock": hex(b), "toBlock": hex(top)}],
                       l1_log_urls())
            STATS["getlogs_calls"] += 1
        except RpcError as e:
            if chunk > 100:
                chunk //= 2
                STATS["chunk_shrinks"] += 1
                say(f"  [sweep] shrink chunk -> {chunk} ({repr(e)[:110]})")
                continue
            raise
        for l in logs:
            seq = int(l["topics"][1], 16)
            _seq2l1[seq] = (int(l["blockNumber"], 16), l["transactionHash"])
        n_found += len(logs)
        b = top + 1
    return n_found


def find_batch_log(seq, center):
    """seq 単独検索（topic1 で絞る）。center から ±span を倍々に広げる。"""
    span = WIN_FWD
    while span <= 256_000:
        lo, hi = max(0, center - span), center + span
        try:
            logs = rpc("eth_getLogs", [{"address": SEQ_INBOX,
                                        "topics": [TOPIC_BATCH, "0x%064x" % seq],
                                        "fromBlock": hex(lo), "toBlock": hex(hi)}],
                       l1_log_urls())
            STATS["getlogs_calls"] += 1
            if logs:
                l = logs[0]
                return int(l["blockNumber"], 16), l["transactionHash"], span
        except RpcError as e:
            say(f"  [single] seq={seq} span={span} rpc err {repr(e)[:110]}")
        span *= 2
    return None, None, span


# ---------------- Phase 3: L1 block timestamp ----------------
_l1ts = {}


def l1_ts(blk):
    if blk in _l1ts:
        return _l1ts[blk]
    b = rpc("eth_getBlockByNumber", [hex(blk), False], l1_block_urls())
    STATS["getblock_calls"] += 1
    if not b:
        raise RpcError(f"l1 block {blk} null")
    _l1ts[blk] = int(b["timestamp"], 16)
    return _l1ts[blk]


# ---------------- 入力読み込み ----------------
def load_rows():
    """327 件を bridge 混合の list[dict] で返す（入力は読み取りのみ）。"""
    rows = []

    cctp = list(csv.DictReader(open(IN_CCTP)))
    old_c = {r["experiment_id"]: r for r in csv.DictReader(open(IN_CCTP_OLD))
             if r["direction"] == "deposit"}
    for r in cctp:
        eid = r["experiment_id"]
        o = old_c[eid]
        rows.append({
            "bridge": "cctp",
            "experiment_id": eid,
            "arb_block_number": int(r["t1_arb_burn_block_number"]),
            "t1_ms": int(float(r["t1_arb_burn_block_ts(ms)"])),
            "t2_ms": int(float(r["t2_iris_attestation_complete(ms)"])),
            "t3_ms": int(float(r["t3_hc_credit_ledger_time(ms)"])),
            "confirmations": r["arb_confirmations_at_attestation(blocks)"],
            "old_l1_block": int(o["safe_l1_block"]),
            "old_tsafe_s": int(int(o["t_safe_ms"]) / 1000),
        })

    nat = list(csv.DictReader(open(IN_NAT)))
    old_n = {r["experiment_id"]: r for r in csv.DictReader(open(IN_NAT_OLD))}
    for r in nat:
        eid = r["experiment_id"]
        o = old_n[eid]
        rows.append({
            "bridge": "native",
            "experiment_id": eid,
            "arb_block_number": int(r["arb_block_number"]),
            "t1_ms": int(float(r["arb_block_timestamp(ms)"])),
            "t2_ms": "",
            "t3_ms": int(float(r["hl_ledger_time(ms)"])),
            "confirmations": "",
            "old_l1_block": int(o["batch_l1_block"]),
            "old_tsafe_s": int(float(o["batch_l1_ts"])),
        })
    return rows


FIELDS = ["bridge", "experiment_id", "arb_block_number", "t1_ms", "t2_ms", "t3_ms",
          "confirmations", "old_l1_block", "old_tsafe_s", "batch_seq",
          "new_l1_block", "new_tsafe_s", "new_l1_tx"]


# ---------------- probe（疎通・ABI 検証） ----------------
def probe():
    from hashlib import sha3_256  # noqa: F401  (keccak は eth_utils を使う)
    say("=== probe: selector / event ABI / endpoint ===")
    try:
        from eth_utils import keccak
        sel = "0x" + keccak(text="findBatchContainingBlock(uint64)")[:4].hex()
        say(f"[abi] selector findBatchContainingBlock(uint64) = {sel} "
            f"(expect {SEL_FBCB}) match={sel == SEL_FBCB}")
    except Exception as e:
        say(f"[abi] keccak 検証スキップ ({e!r})")

    say(f"[arb] endpoints = {[_mask(u) for u in arb_urls()]}")
    bn = int(rpc("eth_blockNumber", [], arb_urls()), 16)
    say(f"[arb] latest L2 block = {bn}")
    for b in (469169954, 404580282):
        say(f"[arb] findBatchContainingBlock({b}) = {find_batch(b, use_cache=False)}")

    say(f"[l1 ] log endpoints = {[_mask(u) for u in l1_log_urls()]}")
    logs = rpc("eth_getLogs", [{"address": SEQ_INBOX, "topics": [TOPIC_BATCH],
                                "fromBlock": hex(25227528), "toBlock": hex(25227628)}],
               l1_log_urls())
    say(f"[l1 ] sample getLogs n={len(logs)}")
    if logs:
        l = logs[0]
        say(f"[l1 ] topics={len(l['topics'])} topic1(uint256)={int(l['topics'][1], 16)} "
            f"blk={int(l['blockNumber'], 16)} tx={l['transactionHash']}")
        say("[l1 ] → topic1 = indexed batchSequenceNumber (3 indexed params) を実ログで確認")


# ---------------- main ----------------
def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    LOGF = open(OUT_LOG, "a")
    load_env()
    t0 = time.time()
    say(f"\n===== run {time.strftime('%Y-%m-%dT%H:%M:%S')} "
        f"(probe={args.probe} sample={args.sample}) =====")
    say(f"[env] ARBITRUM_HTTP_RPC set={bool(_ENV_ARB)}  ETH_RPC_URL set={bool(_ENV_ETH)} "
        f"(値はログに出さない)")

    if args.probe:
        probe()
        return

    if not args.no_cache:
        load_cache()

    rows = load_rows()
    say(f"[in] cctp={sum(1 for r in rows if r['bridge'] == 'cctp')} "
        f"native={sum(1 for r in rows if r['bridge'] == 'native')} total={len(rows)}")
    say(f"[in] {IN_CCTP.relative_to(REPO)} / {IN_CCTP_OLD.relative_to(REPO)} / "
        f"{IN_NAT.relative_to(REPO)} / {IN_NAT_OLD.relative_to(REPO)}")
    if args.sample:
        rows = [r for r in rows if r["bridge"] == "cctp"][:args.sample] + \
               [r for r in rows if r["bridge"] == "native"][:args.sample]
        say(f"[in] SAMPLE mode → {len(rows)} rows")

    # --- Phase 1 ---
    uniq_blocks = sorted({r["arb_block_number"] for r in rows})
    say(f"[phase1] findBatchContainingBlock for {len(uniq_blocks)} unique L2 blocks")
    fails1 = {}
    for i, b in enumerate(uniq_blocks):
        try:
            find_batch(b)
        except Exception as e:
            fails1[b] = repr(e)[:180]
            say(f"  [phase1] FAIL blk={b} {fails1[b]}")
        if (i + 1) % 50 == 0 or i == 0:
            say(f"  [phase1] {i+1}/{len(uniq_blocks)} (calls={STATS['fbcb_calls']} "
                f"cache={STATS['cache_hits']})")
    save_cache()
    for r in rows:
        r["batch_seq"] = _fbcb.get(r["arb_block_number"], "")
        r["_err"] = "" if r["batch_seq"] != "" else \
            f"findBatchContainingBlock failed: {fails1.get(r['arb_block_number'], 'unknown')}"
    seqs = sorted({r["batch_seq"] for r in rows if r["batch_seq"] != ""})
    say(f"[phase1] done: unique batch_seq={len(seqs)} "
        f"(range {min(seqs)}..{max(seqs)})  {time.time()-t0:.0f}s")

    # --- Phase 2a: merged sweep ---
    centers = [r["old_l1_block"] for r in rows]
    windows = merge_windows(centers)
    tot = sum(hi - lo + 1 for lo, hi in windows)
    say(f"[phase2a] merged windows = {len(windows)} intervals, {tot} L1 blocks "
        f"(per-transfer window = old_l1_block {-WIN_BACK:+d}..{WIN_FWD:+d})")
    for k, (lo, hi) in enumerate(windows):
        n = sweep_logs(lo, hi)
        say(f"  [sweep {k+1}/{len(windows)}] L1 [{lo}..{hi}] logs={n} "
            f"map={len(_seq2l1)} ({time.time()-t0:.0f}s)")
    say(f"[phase2a] seq->L1 map size = {len(_seq2l1)}")

    # --- Phase 2b: 個別検索（2a で未発見の seq） ---
    missing = [s for s in seqs if s not in _seq2l1]
    say(f"[phase2b] seq not found in merged sweep: {len(missing)}")
    fails2 = {}
    for s in missing:
        center = next(r["old_l1_block"] for r in rows if r["batch_seq"] == s)
        blk, tx, span = find_batch_log(s, center)
        if blk is None:
            fails2[s] = f"SequencerBatchDelivered(seq={s}) not found within +-{span//2} of {center}"
            say(f"  [phase2b] FAIL {fails2[s]}")
        else:
            _seq2l1[s] = (blk, tx)
            say(f"  [phase2b] seq={s} found at L1 {blk} (widened span=+-{span})")

    # --- Phase 3: timestamps ---
    need_ts = sorted({_seq2l1[r["batch_seq"]][0] for r in rows
                      if r["batch_seq"] != "" and r["batch_seq"] in _seq2l1})
    say(f"[phase3] eth_getBlockByNumber for {len(need_ts)} unique L1 blocks")
    fails3 = {}
    for i, b in enumerate(need_ts):
        try:
            l1_ts(b)
        except Exception as e:
            fails3[b] = repr(e)[:180]
            say(f"  [phase3] FAIL blk={b} {fails3[b]}")
        if (i + 1) % 50 == 0:
            say(f"  [phase3] {i+1}/{len(need_ts)}")

    for r in rows:
        s = r["batch_seq"]
        if s == "":
            r["new_l1_block"] = r["new_tsafe_s"] = r["new_l1_tx"] = ""
            continue
        if s not in _seq2l1:
            r["new_l1_block"] = r["new_tsafe_s"] = r["new_l1_tx"] = ""
            r["_err"] = fails2.get(s, f"seq {s} not matched on L1")
            continue
        blk, tx = _seq2l1[s]
        r["new_l1_block"], r["new_l1_tx"] = blk, tx
        if blk in _l1ts:
            r["new_tsafe_s"] = _l1ts[blk]
        else:
            r["new_tsafe_s"] = ""
            r["_err"] = f"l1 block {blk} timestamp failed: {fails3.get(blk, 'unknown')}"

    ok = [r for r in rows if r["new_tsafe_s"] != ""]
    say(f"[phase3] complete rows = {len(ok)}/{len(rows)}   {time.time()-t0:.0f}s")

    # --- 検証: findBatchContainingBlock の境界（block-1/block/block+1） ---
    random.seed(20261001)
    picks = random.sample([r for r in rows if r["batch_seq"] != ""],
                          min(3, sum(1 for r in rows if r["batch_seq"] != "")))
    neighbor = []
    say("[check] findBatchContainingBlock at block-1 / block / block+1")
    for r in picks:
        b = r["arb_block_number"]
        trio = {}
        for d in (-1, 0, 1):
            try:
                trio[d] = find_batch(b + d, use_cache=False)
            except Exception as e:
                trio[d] = f"ERR {repr(e)[:80]}"
        neighbor.append((r, trio))
        say(f"  id={r['experiment_id']}({r['bridge']}) blk={b}: "
            f"{trio[-1]} / {trio[0]} / {trio[1]}")

    if args.sample:
        say("SAMPLE only — no files written")
        say(f"[stats] {STATS}")
        for r in rows:
            say(f"  {r['bridge']:6} id={r['experiment_id']:>3} blk={r['arb_block_number']} "
                f"seq={r['batch_seq']} old_l1={r['old_l1_block']} new_l1={r['new_l1_block']} "
                f"old_ts={r['old_tsafe_s']} new_ts={r['new_tsafe_s']} "
                f"t1={r['t1_ms']//1000} err={r['_err']}")
        return

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    say(f"WROTE {OUT_CSV.name} ({len(rows)} rows)")

    write_summary(rows, neighbor, windows, time.time() - t0)
    say(f"[done] {time.time()-t0:.0f}s stats={STATS}")


def describe(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"n": len(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
            "min": xs[0], "max": xs[-1]}


def write_summary(rows, neighbor, windows, elapsed):
    L = []
    A = L.append
    A("ACM DLT / 露出窓 G 再構成 — t_safe の厳密再対応づけ 整合性チェック")
    A("=" * 78)
    A(f"生成       : refix_batch.py  {time.strftime('%Y-%m-%dT%H:%M:%S')}  ({elapsed:.0f}s)")
    A(f"方式       : Arbitrum NodeInterface({NODE_INTERFACE}).findBatchContainingBlock(uint64)")
    A(f"             → SequencerInbox({SEQ_INBOX}) の")
    A(f"               SequencerBatchDelivered topic0={TOPIC_BATCH}")
    A(f"               topic1=indexed batchSequenceNumber で突合")
    A(f"探索窓     : 旧バッチ L1 ブロック {-WIN_BACK:+d}..{WIN_FWD:+d}（結合後 {len(windows)} 区間を一括スイープ）")
    A("")

    A("0. 件数")
    A("-" * 78)
    for br in ("cctp", "native", None):
        sub = rows if br is None else [r for r in rows if r["bridge"] == br]
        done = [r for r in sub if r["new_tsafe_s"] != ""]
        A(f"  {(br or 'TOTAL'):6} : n={len(sub):3}  new_tsafe 取得成功={len(done):3}  "
          f"空欄={len(sub)-len(done)}")
    A("")

    # 1. new_tsafe >= t1
    A("1. 全件で new_tsafe_s >= t1_ms/1000 か（バッチ投稿はL2ブロック生成より後、が必要条件）")
    A("-" * 78)
    done = [r for r in rows if r["new_tsafe_s"] != ""]
    viol = [r for r in done if r["new_tsafe_s"] < r["t1_ms"] / 1000.0]
    A(f"  検査対象 {len(done)} 件 / 違反 {len(viol)} 件 → "
      f"{'PASS（違反なし）' if not viol else 'FAIL'}")
    for r in viol:
        A(f"    - {r['bridge']} id={r['experiment_id']} arb_blk={r['arb_block_number']} "
          f"seq={r['batch_seq']} new_tsafe={r['new_tsafe_s']} t1={r['t1_ms']/1000:.3f} "
          f"diff={r['new_tsafe_s'] - r['t1_ms']/1000:.3f}s")
    A("")

    # 2. 旧 vs 新 の L1 ブロック
    A("2. 旧ヒューリスティックとの L1 ブロック比較")
    A("-" * 78)
    A(f"  {'bridge':7} {'n':>4} {'一致':>6} {'不一致':>7} {'new<old':>8} "
      f"{'new-old(blk) med':>17} {'new-old(s) med':>15}")
    for br in ("cctp", "native"):
        sub = [r for r in rows if r["bridge"] == br and r["new_l1_block"] != ""]
        same = [r for r in sub if r["new_l1_block"] == r["old_l1_block"]]
        diff = [r for r in sub if r["new_l1_block"] != r["old_l1_block"]]
        less = [r for r in sub if r["new_l1_block"] < r["old_l1_block"]]
        db = describe([r["new_l1_block"] - r["old_l1_block"] for r in sub])
        ds = describe([r["new_tsafe_s"] - r["old_tsafe_s"] for r in sub
                       if r["new_tsafe_s"] != ""])
        A(f"  {br:7} {len(sub):>4} {len(same):>6} {len(diff):>7} {len(less):>8} "
          f"{db['median']:>17.1f} {ds['median']:>15.1f}")
    A("")
    for br in ("cctp", "native"):
        less = [r for r in rows if r["bridge"] == br and r["new_l1_block"] != ""
                and r["new_l1_block"] < r["old_l1_block"]]
        if less:
            A(f"  new_l1_block < old_l1_block の内訳（{br}, {len(less)} 件）:")
            for r in less:
                A(f"    - id={r['experiment_id']} arb_blk={r['arb_block_number']} "
                  f"seq={r['batch_seq']} old_l1={r['old_l1_block']} "
                  f"new_l1={r['new_l1_block']} (diff={r['new_l1_block']-r['old_l1_block']}) "
                  f"old_tsafe={r['old_tsafe_s']} new_tsafe={r['new_tsafe_s']} "
                  f"t1={r['t1_ms']/1000:.0f}")
        else:
            A(f"  new_l1_block < old_l1_block の件（{br}）: 0 件")
    A("")
    A("  ※ 旧方式は「t_1 を含む L1 ブロック以降の最初のバッチ」= t_safe の下界であり、")
    A("    その最初のバッチは当該 L2 ブロックより前の L2 区間しか含まないことが多い。")
    A("    厳密化で new >= old の方向へ動くのが期待挙動。")
    A("")

    # 3. 旧 t_safe < t1 の解消
    A("3. 旧 t_safe が t_1 より前だった件（因果違反）の解消")
    A("-" * 78)
    for br, exp in (("cctp", 22), ("native", 20)):
        sub = [r for r in rows if r["bridge"] == br]
        old_bad = [r for r in sub if r["old_tsafe_s"] < r["t1_ms"] / 1000.0]
        still = [r for r in old_bad if r["new_tsafe_s"] != ""
                 and r["new_tsafe_s"] < r["t1_ms"] / 1000.0]
        nots = [r for r in old_bad if r["new_tsafe_s"] == ""]
        A(f"  {br:7}: 旧違反 {len(old_bad)} 件 (task 想定 {exp} 件 → "
          f"{'一致' if len(old_bad) == exp else '不一致!'})  "
          f"新方式でも違反 {len(still)} 件  未取得 {len(nots)} 件 → "
          f"{'全解消' if not still and not nots else '未解消あり'}")
        for r in old_bad:
            A(f"    - id={r['experiment_id']:>3} t1={r['t1_ms']/1000:.0f} "
              f"old_tsafe={r['old_tsafe_s']} (t1-old={r['t1_ms']/1000 - r['old_tsafe_s']:+.0f}s) "
              f"-> new_tsafe={r['new_tsafe_s']} "
              f"(new-t1={(r['new_tsafe_s'] - r['t1_ms']/1000) if r['new_tsafe_s'] != '' else float('nan'):+.0f}s) "
              f"old_l1={r['old_l1_block']} new_l1={r['new_l1_block']}")
        A("")

    # 4. 境界テスト
    A("4. findBatchContainingBlock の境界検証（無作為 3 件, block-1 / block / block+1）")
    A("-" * 78)
    A("   同一バッチ、または隣接バッチ（±1）であることを確認する。")
    for r, trio in neighbor:
        vals = [trio.get(d) for d in (-1, 0, 1)]
        nums = [v for v in vals if isinstance(v, int)]
        ok = all(abs(v - trio[0]) <= 1 for v in nums) if isinstance(trio.get(0), int) else False
        A(f"  {r['bridge']:6} id={r['experiment_id']:>3} blk={r['arb_block_number']}: "
          f"b-1={vals[0]}  b={vals[1]}  b+1={vals[2]}  -> "
          f"{'OK (同一 or 隣接)' if ok else 'NG'}")
    A("")

    # 5. 参考値
    A("5. 参考値: t1 -> new_tsafe（バッチ投稿遅延）[秒]")
    A("-" * 78)
    A(f"  {'bridge':7} {'n':>4} {'median':>9} {'mean':>9} {'min':>9} {'max':>9}")
    for br in ("cctp", "native"):
        xs = [r["new_tsafe_s"] - r["t1_ms"] / 1000.0
              for r in rows if r["bridge"] == br and r["new_tsafe_s"] != ""]
        d = describe(xs)
        if d:
            A(f"  {br:7} {d['n']:>4} {d['median']:>9.1f} {d['mean']:>9.1f} "
              f"{d['min']:>9.1f} {d['max']:>9.1f}")
    A("")
    A("  （参考）旧方式 t1 -> old_tsafe [秒]")
    A(f"  {'bridge':7} {'n':>4} {'median':>9} {'mean':>9} {'min':>9} {'max':>9}")
    for br in ("cctp", "native"):
        xs = [r["old_tsafe_s"] - r["t1_ms"] / 1000.0 for r in rows if r["bridge"] == br]
        d = describe(xs)
        if d:
            A(f"  {br:7} {d['n']:>4} {d['median']:>9.1f} {d['mean']:>9.1f} "
              f"{d['min']:>9.1f} {d['max']:>9.1f}")
    A("")

    # 6. 失敗・空欄
    A("6. 失敗・空欄")
    A("-" * 78)
    bad = [r for r in rows if r["new_tsafe_s"] == ""]
    A(f"  空欄 {len(bad)} 件")
    for r in bad:
        A(f"    - {r['bridge']} id={r['experiment_id']} arb_blk={r['arb_block_number']}: "
          f"{r.get('_err', '')}")
    A("")
    A("7. RPC 実行統計")
    A("-" * 78)
    A("  " + json.dumps(STATS))
    A("")
    OUT_SUM.write_text("\n".join(L) + "\n")
    say(f"WROTE {OUT_SUM.name}")


if __name__ == "__main__":
    sys.exit(main())
