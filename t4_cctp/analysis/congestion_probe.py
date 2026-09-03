#!/usr/bin/env python3
"""
t4_cctp/analysis/congestion_probe.py

T4 CCTP フェーズ3：オンチェーンの「ブロック占有（混雑）」を実測し、
  ③ deposit の長い裾野(tail, a2m>20s) の原因切り分け（forwarder提出遅延 vs HyperEVM混雑）
  ④ Deposit/Withdraw のブロック占有比較（どちらの処理時が混んでいたか／二峰は混雑起因か）
を実証する付加スクリプト。

入力（読み取り専用・改変しない）:
  result/T4_cctp/deposit_cctp_latency.csv
  result/T4_cctp/withdraw_cctp_latency.csv
  result/T4_cctp/deposit_l1_enriched.csv   （group=fast/slow/tail 参照のみ）
出力（新規のみ）:
  result/T4_cctp/congestion_enriched.csv   （long 形式：1行=1ブロック観測）

RPC:
  HyperEVM : .env の HL_EVM_RPC_ARCHIVE（QuickNode 専用EP）
  Arbitrum : .env の ARBITRUM_HTTP_RPC（公開）+ 公開フォールバック

占有指標の注意:
  - HyperEVM small block は gasLimit≈3,000,000（~1s間隔）。gasUsed/gasLimit が有効な占有率。
  - Arbitrum Nitro は gasLimit=2^50 のプレースホルダで占有率が無意味。
    → tx数・絶対 gasUsed・baseFeePerGas(gwei, floor=0.01gwei) で混雑を測る。

使い方:
  サンプル生表示: python3 -u t4_cctp/analysis/congestion_probe.py --sample-only
  本実行       : python3 -u t4_cctp/analysis/congestion_probe.py [--dep-n 210] [--wit-n 210]
  (公開RPCは sandbox 無効で実行。長時間は run_in_background + tee)
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEP_CSV = REPO / "result" / "T4_cctp" / "deposit_cctp_latency.csv"
WIT_CSV = REPO / "result" / "T4_cctp" / "withdraw_cctp_latency.csv"
ENR_CSV = REPO / "result" / "T4_cctp" / "deposit_l1_enriched.csv"
OUT_CSV = REPO / "result" / "T4_cctp" / "congestion_enriched.csv"


def _load_env():
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
HEVM_RPC = os.environ.get("HL_EVM_RPC_ARCHIVE")
ARB_RPC = os.environ.get("ARBITRUM_HTTP_RPC", "https://arb1.arbitrum.io/rpc")
ARB_FALLBACK = ["https://arbitrum-one-rpc.publicnode.com", "https://arbitrum.drpc.org"]
UA = "Mozilla/5.0 (research; cctp-congestion-analysis)"
HEVM_SLEEP = 0.05   # QuickNode 専用EP
ARB_SLEEP = 0.12    # 公開EP（レート制限配慮）


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


_hevm_cache = {}
_arb_cache = {}


def _parse_block(b):
    gu = int(b["gasUsed"], 16)
    gl = int(b["gasLimit"], 16)
    bf = int(b["baseFeePerGas"], 16) if b.get("baseFeePerGas") else None
    return {
        "block_number": int(b["number"], 16),
        "block_ts": int(b["timestamp"], 16),
        "gas_used": gu,
        "gas_limit": gl,
        "gas_ratio": (gu / gl) if gl else "",
        "tx_count": len(b.get("transactions", [])),
        "base_fee_gwei": (bf / 1e9) if bf is not None else "",
    }


def hevm_block(num):
    if num in _hevm_cache:
        return _hevm_cache[num]
    b = _rpc(HEVM_RPC, "eth_getBlockByNumber", [hex(num), False])
    time.sleep(HEVM_SLEEP)
    if not b:
        raise RpcError(f"hevm block {num} null")
    out = _parse_block(b)
    _hevm_cache[num] = out
    return out


def arb_block(num):
    if num in _arb_cache:
        return _arb_cache[num]
    last = None
    for u in [ARB_RPC] + ARB_FALLBACK:
        try:
            b = _rpc(u, "eth_getBlockByNumber", [hex(num), False])
            time.sleep(ARB_SLEEP)
            if not b:
                raise RpcError(f"arb block {num} null")
            out = _parse_block(b)
            _arb_cache[num] = out
            return out
        except RpcError as e:
            last = e
    raise last


FIELDS = [
    "analysis", "experiment_id", "direction", "group", "role", "rel_offset",
    "chain", "block_number", "block_ts", "gas_used", "gas_limit", "gas_ratio",
    "tx_count", "base_fee_gwei", "a2m_ms", "iris_wait_ms", "err",
]


def _row(analysis, r, direction, group, role, rel_offset, chain, blk, a2m, iris, err=""):
    base = {
        "analysis": analysis, "experiment_id": r["experiment_id"], "direction": direction,
        "group": group, "role": role, "rel_offset": rel_offset, "chain": chain,
        "a2m_ms": a2m, "iris_wait_ms": iris, "err": err,
    }
    if blk:
        base.update({k: blk[k] for k in ("block_number", "block_ts", "gas_used", "gas_limit", "gas_ratio", "tx_count", "base_fee_gwei")})
    else:
        for k in ("block_number", "block_ts", "gas_used", "gas_limit", "gas_ratio", "tx_count", "base_fee_gwei"):
            base[k] = ""
    return base


def grp_of(eid, enr):
    g = enr.get(eid)
    return g if g else ""


def probe1(dep, enr, out, n_normal, fwd=5, cap=150):
    """③ tail 切り分け: tail行(a2m>20s)について attestation完了→mint取込 の窓全体の HEVM block を
    後方に辿って取得（+ 前方 fwd 件）。窓内に空きブロック(ratio低)があれば forwarder提出遅延、
    窓が一貫して満杯なら HyperEVM混雑、と判定できる。通常行サンプルは mint block のみ（比較用）。"""
    tail = [r for r in dep if r["attestation_to_mint(ms)"] not in ("", "N/A") and int(r["attestation_to_mint(ms)"]) > 20000]
    tail_ids = {r["experiment_id"] for r in tail}
    # 通常行サンプル：a2m が中央値付近〜通常域。等間隔抽出で偏り回避。
    normal = [r for r in dep if r["experiment_id"] not in tail_ids
              and r["t2_5_hevm_mint_block_number"] not in ("", "0", "N/A")]
    step = max(1, len(normal) // n_normal)
    normal_s = normal[::step][:n_normal]

    print(f"[probe1] tail={sorted(int(i) for i in tail_ids)}  normal_sample={len(normal_s)}", flush=True)
    for r in tail:
        eid = r["experiment_id"]
        mint_bn = int(r["t2_5_hevm_mint_block_number"])
        a2m, iris = r["attestation_to_mint(ms)"], r["iris_wait(ms)"]
        g = grp_of(eid, enr)
        # mint block + 前方 fwd
        mint_blk = None
        for off in range(0, fwd + 1):
            try:
                blk = hevm_block(mint_bn + off)
                if off == 0:
                    mint_blk = blk
                out.append(_row("probe1", r, "deposit", g, "mint", off, "hevm", blk, a2m, iris))
            except RpcError as e:
                out.append(_row("probe1", r, "deposit", g, "mint", off, "hevm", None, a2m, iris, repr(e)[:120]))
        # 後方に attestation 完了時刻(t2 = mint_ts - a2m) まで辿る
        if mint_blk is not None:
            t2_ms = mint_blk["block_ts"] * 1000 - int(a2m)
            roomy = 0
            total = 0
            bn = mint_bn - 1
            steps = 0
            while steps < cap:
                try:
                    blk = hevm_block(bn)
                except RpcError as e:
                    out.append(_row("probe1", r, "deposit", g, "mint", bn - mint_bn, "hevm", None, a2m, iris, repr(e)[:120]))
                    break
                out.append(_row("probe1", r, "deposit", g, "mint", bn - mint_bn, "hevm", blk, a2m, iris))
                # small block(limit≈3M)で占有<0.7 ＝ mint tx(~250k)が入る余地あり
                if blk["gas_limit"] and blk["gas_limit"] <= 5_000_000:
                    total += 1
                    if blk["gas_ratio"] != "" and blk["gas_ratio"] < 0.7:
                        roomy += 1
                if blk["block_ts"] * 1000 <= t2_ms:
                    break
                bn -= 1
                steps += 1
            pct = f"{100*roomy/total:.0f}% had room" if total else "no small blocks"
            print(f"  TAIL id={eid:>3} a2m={a2m:>6} mint_blk={mint_bn} mint_ratio={mint_blk['gas_ratio']:.3f} "
                  f"window_blocks={steps+1} small={total} roomy(<0.7)={roomy} ({pct})", flush=True)
    for r in normal_s:
        eid = r["experiment_id"]
        mint_bn = int(r["t2_5_hevm_mint_block_number"])
        a2m, iris = r["attestation_to_mint(ms)"], r["iris_wait(ms)"]
        g = grp_of(eid, enr)
        try:
            blk = hevm_block(mint_bn)
            out.append(_row("probe1", r, "deposit", g, "mint", 0, "hevm", blk, a2m, iris))
        except RpcError as e:
            out.append(_row("probe1", r, "deposit", g, "mint", 0, "hevm", None, a2m, iris, repr(e)[:120]))


def probe2(dep, wit, enr, out, dep_n, wit_n):
    """④ 占有比較: deposit(arb burn + hevm mint) と withdraw(hevm burn + arb mint)。"""
    dep_s = dep[:dep_n] if dep_n else dep
    wit_s = wit[:wit_n] if wit_n else wit
    print(f"[probe2] deposit={len(dep_s)} withdraw={len(wit_s)}", flush=True)

    for i, r in enumerate(dep_s):
        eid = r["experiment_id"]
        g = grp_of(eid, enr)
        a2m, iris = r["attestation_to_mint(ms)"], r["iris_wait(ms)"]
        # Arbitrum burn
        if r["t1_arb_burn_block_number"] not in ("", "0", "N/A"):
            bn = int(r["t1_arb_burn_block_number"])
            try:
                out.append(_row("probe2", r, "deposit", g, "burn", 0, "arb", arb_block(bn), a2m, iris))
            except RpcError as e:
                out.append(_row("probe2", r, "deposit", g, "burn", 0, "arb", None, a2m, iris, repr(e)[:120]))
        # HyperEVM mint
        if r["t2_5_hevm_mint_block_number"] not in ("", "0", "N/A"):
            bn = int(r["t2_5_hevm_mint_block_number"])
            try:
                out.append(_row("probe2", r, "deposit", g, "mint", 0, "hevm", hevm_block(bn), a2m, iris))
            except RpcError as e:
                out.append(_row("probe2", r, "deposit", g, "mint", 0, "hevm", None, a2m, iris, repr(e)[:120]))
        if (i + 1) % 25 == 0:
            print(f"  [probe2 dep] {i+1}/{len(dep_s)}", flush=True)

    for i, r in enumerate(wit_s):
        eid = r["experiment_id"]
        a2m, iris = r["attestation_to_mint(ms)"], r["iris_wait(ms)"]
        # HyperEVM burn
        if r["t1_hevm_burn_block_number"] not in ("", "0", "N/A"):
            bn = int(r["t1_hevm_burn_block_number"])
            try:
                out.append(_row("probe2", r, "withdraw", "", "burn", 0, "hevm", hevm_block(bn), a2m, iris))
            except RpcError as e:
                out.append(_row("probe2", r, "withdraw", "", "burn", 0, "hevm", None, a2m, iris, repr(e)[:120]))
        # Arbitrum mint
        if r["t3_arb_mint_block_number"] not in ("", "0", "N/A"):
            bn = int(r["t3_arb_mint_block_number"])
            try:
                out.append(_row("probe2", r, "withdraw", "", "mint", 0, "arb", arb_block(bn), a2m, iris))
            except RpcError as e:
                out.append(_row("probe2", r, "withdraw", "", "mint", 0, "arb", None, a2m, iris, repr(e)[:120]))
        if (i + 1) % 25 == 0:
            print(f"  [probe2 wit] {i+1}/{len(wit_s)}", flush=True)


def sample_only():
    """承認ゲート用：各カテゴリ数件の生レスポンスを表示（ファイル書き込みなし）。"""
    dep = list(csv.DictReader(open(DEP_CSV)))
    wit = list(csv.DictReader(open(WIT_CSV)))
    tail = [r for r in dep if r["attestation_to_mint(ms)"] not in ("", "N/A") and int(r["attestation_to_mint(ms)"]) > 20000]
    print("=== HyperEVM: tail mint blocks (±1) ===")
    for r in tail:
        bn = int(r["t2_5_hevm_mint_block_number"])
        for off in (-1, 0, 1):
            b = hevm_block(bn + off)
            print(f"  id={r['experiment_id']} a2m={r['attestation_to_mint(ms)']} off={off:+d} blk={b['block_number']} "
                  f"ts={b['block_ts']} gasUsed={b['gas_used']} gasLimit={b['gas_limit']} ratio={b['gas_ratio']:.3f} "
                  f"tx={b['tx_count']} basefee={b['base_fee_gwei']:.3f}gwei")
    print("=== Arbitrum: deposit burn (id=1) vs withdraw mint (id=1) ===")
    b = arb_block(int(dep[0]["t1_arb_burn_block_number"]))
    print(f"  dep burn  blk={b['block_number']} gasUsed={b['gas_used']} gasLimit={b['gas_limit']} tx={b['tx_count']} basefee={b['base_fee_gwei']:.5f}gwei")
    b = arb_block(int(wit[0]["t3_arb_mint_block_number"]))
    print(f"  wit mint  blk={b['block_number']} gasUsed={b['gas_used']} gasLimit={b['gas_limit']} tx={b['tx_count']} basefee={b['base_fee_gwei']:.5f}gwei")
    print("=== HyperEVM: withdraw burn (id=1) ===")
    b = hevm_block(int(wit[0]["t1_hevm_burn_block_number"]))
    print(f"  wit burn  blk={b['block_number']} gasUsed={b['gas_used']} gasLimit={b['gas_limit']} ratio={b['gas_ratio']:.3f} tx={b['tx_count']} basefee={b['base_fee_gwei']:.3f}gwei")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--dep-n", type=int, default=0, help="probe2 deposit 件数（0=全件）")
    ap.add_argument("--wit-n", type=int, default=0, help="probe2 withdraw 件数（0=全件）")
    ap.add_argument("--normal-n", type=int, default=20, help="probe1 通常行サンプル数")
    ap.add_argument("--skip-probe1", action="store_true")
    ap.add_argument("--skip-probe2", action="store_true")
    args = ap.parse_args()

    if not HEVM_RPC:
        print("ERROR: HL_EVM_RPC_ARCHIVE not set in .env", file=sys.stderr)
        sys.exit(1)

    if args.sample_only:
        sample_only()
        return

    dep = list(csv.DictReader(open(DEP_CSV)))
    wit = list(csv.DictReader(open(WIT_CSV)))
    enr = {r["experiment_id"]: r.get("group", "") for r in csv.DictReader(open(ENR_CSV))} if ENR_CSV.exists() else {}

    out = []
    t0 = time.time()
    if not args.skip_probe1:
        probe1(dep, enr, out, args.normal_n)
    if not args.skip_probe2:
        probe2(dep, wit, enr, out, args.dep_n, args.wit_n)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    errs = sum(1 for r in out if r["err"])
    print(f"\nWROTE {OUT_CSV}  ({len(out)} rows, err={errs})  in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
