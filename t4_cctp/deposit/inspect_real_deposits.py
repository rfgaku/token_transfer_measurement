#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_real_deposits.py — 本物の自動着金 deposit の burn パラメータを実例から読む（read-only）。

HyperEVM の CctpForwarder への USDC mint(Transfer 0x0->forwarder) を辿り、その mintAndForward(message,
attestation) 入力を decode して、原 burn の destinationCaller / maxFee / mintRecipient / hookData /
finalityThresholdExecuted を抽出する。我々の送信予定値と突合する目的。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from decimal import Decimal
from web3 import Web3
from web3.middleware import geth_poa_middleware
from t4_cctp.deposit import config as cfg


def rpc_retry(fn, *a, tries=8, base=0.8, **kw):
    """rate limited(-32005)等に指数バックオフでリトライ。"""
    last = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            last = e
            if "rate limited" in str(e) or "-32005" in str(e):
                time.sleep(base * (i + 1))
                continue
            raise
    raise last

OUR_SENDER = "0x322f51d8191b7cb463c06113c527c28afde70321"
FORWARDER = Web3.to_checksum_address(cfg.CCTP_FORWARDER_HEVM)
USDC = Web3.to_checksum_address(cfg.HYPEREVM_USDC_ADDRESS)
TRANSFER = Web3.keccak(text="Transfer(address,address,uint256)").hex()

FORWARDER_ABI = [
    {"name": "mintAndForward", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "message", "type": "bytes"}, {"name": "attestation", "type": "bytes"}],
     "outputs": []},
]


def make_hl():
    w3 = Web3(Web3.HTTPProvider(cfg.HL_EVM_RPC_URL,
              request_kwargs={"timeout": 25, "headers": {"User-Agent": "Mozilla/5.0"}}))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3


def addr32(b):  # 32byte -> checksum address
    return Web3.to_checksum_address("0x" + b[-20:].hex())


def decode_message(msg: bytes) -> dict:
    # MessageV2 header
    version = int.from_bytes(msg[0:4], "big")
    src = int.from_bytes(msg[4:8], "big")
    dst = int.from_bytes(msg[8:12], "big")
    dest_caller = msg[108:140]
    min_ft = int.from_bytes(msg[140:144], "big")
    ft_exec = int.from_bytes(msg[144:148], "big")
    body = msg[148:]
    # BurnMessageV2 body
    burn_token = body[4:36]
    mint_recipient = body[36:68]
    amount = int.from_bytes(body[68:100], "big")
    msg_sender = body[100:132]
    max_fee = int.from_bytes(body[132:164], "big")
    fee_exec = int.from_bytes(body[164:196], "big")
    hook = body[228:]
    dc_is_zero = (int.from_bytes(dest_caller, "big") == 0)
    return {
        "src": src, "dst": dst, "minFinality": min_ft, "finalityExecuted": ft_exec,
        "destinationCaller": ("ZERO" if dc_is_zero else addr32(dest_caller)),
        "mintRecipient": addr32(mint_recipient),
        "amount": amount, "maxFee": max_fee, "feeExecuted": fee_exec,
        "messageSender": addr32(msg_sender),
        "hook_len": len(hook), "hook_hex": "0x" + hook.hex(),
    }


def main():
    w3 = make_hl()
    fwd = w3.eth.contract(address=FORWARDER, abi=FORWARDER_ABI)
    latest = w3.eth.block_number
    span = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    frm = max(0, latest - span)
    print(f"HyperEVM latest={latest} scan {frm}..{latest} (USDC mint 0x0->forwarder, 1000-block chunks)")
    from_topic = "0x" + ("0"*64)
    to_topic = "0x" + FORWARDER.lower()[2:].rjust(64, "0")
    logs = []
    # 新しい側のチャンクから取得し、十分集まったら止める
    hi = latest
    while hi >= frm and len(logs) < 60:
        lo = max(frm, hi - 999)
        try:
            chunk = rpc_retry(w3.eth.get_logs, {"fromBlock": lo, "toBlock": hi, "address": USDC,
                                                "topics": [TRANSFER, from_topic, to_topic]})
            logs.extend(chunk)
        except Exception as e:
            print(f"  [chunk {lo}-{hi} error] {e}")
        hi = lo - 1
        time.sleep(0.5)
    logs.sort(key=lambda l: (l["blockNumber"], l["logIndex"]))
    print(f"mint(0x0->forwarder) logs: {len(logs)}")
    seen_tx = []
    others = 0
    ours = 0
    for lg in reversed(logs):  # 新しい順
        if others >= 4:
            break
        txh = lg["transactionHash"].hex()
        if txh in seen_tx:
            continue
        seen_tx.append(txh)
        time.sleep(0.3)
        try:
            tx = rpc_retry(w3.eth.get_transaction, txh)
            func, args = fwd.decode_function_input(tx["input"])
            msg = args["message"] if isinstance(args["message"], (bytes, bytearray)) else Web3.to_bytes(args["message"])
            d = decode_message(bytes(msg))
        except Exception as e:
            print(f"  [skip {txh[:12]}] decode error: {e}")
            continue
        is_ours = d["messageSender"].lower() == OUR_SENDER
        tag = "OURS" if is_ours else "THIRD-PARTY"
        if is_ours:
            ours += 1
        else:
            others += 1
        # 第三者の実例を優先表示（最大4件）。ours は1件だけ参考表示。
        if (not is_ours and others <= 4) or (is_ours and ours <= 1):
            caller = tx["from"]
            print("-" * 70)
            print(f" tx={txh}  caller(from)={caller}  [{tag}]")
            print(f"   amount={Decimal(d['amount'])/Decimal(10**6)}USDC maxFee={d['maxFee']} feeExecuted={d['feeExecuted']}")
            print(f"   finalityExecuted={d['finalityExecuted']} minFinality={d['minFinality']}")
            print(f"   destinationCaller={d['destinationCaller']}")
            print(f"   mintRecipient={d['mintRecipient']}  messageSender={d['messageSender']}")
            print(f"   hook_len={d['hook_len']} hook={d['hook_hex'][:80]}...")
    print("=" * 70)
    print(f" 集計: THIRD-PARTY={others}件 OURS={ours}件  (destinationCaller の実例傾向を上で確認)")


if __name__ == "__main__":
    main()
