"""A. Bridge2 の disputePeriodSeconds / blockDurationMillis を4ブロックで eth_call する。"""
import csv, sys, requests
from pathlib import Path
from dotenv import dotenv_values
from eth_utils import keccak

# リポジトリのルートから実行する: python3 experiments/bridge2_params.py
OUT = Path("result/bridge2_params")
SURVEY = Path("result/native_bridge_survey")
SRC_SCRIPT = Path("experiments/native_bridge_arrival_survey.py")
BRIDGE2 = "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7"  # SRC_SCRIPT L73-74 の既定値（.env と一致確認済み）

env = dotenv_values(".env")
ENDPOINTS = [("<env:ARBITRUM_HTTP_RPC>", env.get("ARBITRUM_HTTP_RPC")),
             ("https://arbitrum-one-rpc.publicnode.com", "https://arbitrum-one-rpc.publicnode.com"),
             ("https://arbitrum.drpc.org", "https://arbitrum.drpc.org")]

def blocks(path):
    with open(path) as f:
        b = [int(r["block_number"]) for r in csv.DictReader(f)]
    return min(b), max(b)

def rpc(url, method, params):
    r = requests.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=30)
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]

FUNCS = ["disputePeriodSeconds()", "blockDurationMillis()"]
SEL = {f: "0x" + keccak(text=f)[:4].hex() for f in FUNCS}

mn11, mx11 = blocks(SURVEY / "native_bridge_contract_events_2025-11.csv")
mn06, _ = blocks(SURVEY / "native_bridge_contract_events_2026-06.csv")
points = [("(1) 2025-11 min", mn11), ("(2) 2025-11 max", mx11), ("(3) 2026-06 min", mn06), ("(4) latest", "latest")]

lines = [f"Bridge2 address: {BRIDGE2}",
         f"address source: {SRC_SCRIPT} (L72-74, BRIDGE2 default / HL_DEPOSIT_BRIDGE_ADDRESS; .env value identical)",
         "selectors (keccak256 first 4 bytes):"]
lines += [f"  {f} -> {s}" for f, s in SEL.items()]
lines.append("")
fails = []
for label, blk in points:
    tag = blk if blk == "latest" else hex(blk)
    if blk == "latest":
        for name, url in ENDPOINTS:
            try:
                blk_resolved = int(rpc(url, "eth_blockNumber", []), 16); break
            except Exception as e:
                fails.append(f"{label} eth_blockNumber @ {name}: {e}")
        tag = hex(blk_resolved)
        label += f" (resolved block {blk_resolved})"
    lines.append(f"{label}: block {int(tag,16)}")
    for f in FUNCS:
        got = None
        for name, url in ENDPOINTS:
            try:
                res = rpc(url, "eth_call", [{"to": BRIDGE2, "data": SEL[f]}, tag])
                if res in (None, "0x"):
                    raise RuntimeError(f"empty result {res!r}")
                got = (int(res, 16), name); break
            except Exception as e:
                fails.append(f"{label} {f} @ {name}: {str(e)[:200]}")
        lines.append(f"  {f} = {got[0] if got else 'FAILED'}   endpoint: {got[1] if got else '-'}")
lines.append("")
lines.append(f"failed attempts: {len(fails)}")
lines += ["  " + x for x in fails]
txt = "\n".join(lines) + "\n"
(OUT / "bridge2_params.txt").write_text(txt)
print(txt)
