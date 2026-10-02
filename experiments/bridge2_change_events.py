"""補足: Bridge2 の ChangedDisputePeriodSeconds / ChangedBlockDurationMillis を
2025-11 窓の最小ブロック〜latest で getLogs し、パラメータ変更の有無を確認する（値の推測はしない）。"""
import requests, sys
from pathlib import Path
from dotenv import dotenv_values
# リポジトリのルートから実行する: python3 experiments/bridge2_change_events.py
URL = dotenv_values(".env")["ARBITRUM_HTTP_RPC"]
BRIDGE2 = "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7"
TOPICS = ["0x04edaf680108675f58d2ea70e9e7886c39ed38b66439622f8362d36595fe8169",
          "0x0ef2da393c3832a8f08ce447e14948d21e84f864facf7327137387bd0596a563"]
def rpc(m, p):
    j = requests.post(URL, json={"jsonrpc":"2.0","id":1,"method":m,"params":p}, timeout=60).json()
    if "error" in j: raise RuntimeError(j["error"])
    return j["result"]
start = 404479991
latest = int(rpc("eth_blockNumber", []), 16)
step, b, logs, ok = 10_000_000, start, [], True
while b <= latest:
    e = min(b + step - 1, latest)
    try:
        logs += rpc("eth_getLogs", [{"address": BRIDGE2, "topics": [TOPICS], "fromBlock": hex(b), "toBlock": hex(e)}])
        b = e + 1
    except Exception as ex:
        if step <= 10_000: print("FAIL", b, e, ex); ok = False; break
        step //= 4
print(f"range {start}..{latest} complete={ok} n_logs={len(logs)}")
for l in logs: print(int(l["blockNumber"],16), l["topics"][0], l["data"])
