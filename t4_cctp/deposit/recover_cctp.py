#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recover_cctp.py — burn 済み・未配送の CCTP メッセージを自力でオンチェーン回収する。

元本(burn済みUSDC)は減らない。失敗しても HyperEVM ガスのみ消費。

段1 (--inspect): 両 burn の Iris 全文取得・message/attestation 保存・HyperEVMガス残高・
                 forwarder/transmitter のコード存在と関数セレクタを報告（read-only）。
段2 (--recover): attestation=complete のものから、CctpForwarder.mintAndForward を estimate_gas、
                 ダメなら MessageTransmitterV2.receiveMessage を estimate_gas。通った方を実行。
                 --execute を付けたときのみ実送信。付けなければ estimate_gas までで停止。
"""
import os
import sys
import json
import argparse
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
from web3 import Web3
from web3.middleware import geth_poa_middleware

from t4_cctp.deposit import config as cfg

ARB_SENDER_ADDRESS = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
ARB_SENDER_PRIVATE_KEY = os.environ.get("ARB_SENDER_PRIVATE_KEY")

# CCTP V2 は全 EVM チェーンで同一アドレス（config の Arb 値を HyperEVM でも使用）
HEVM_MESSAGE_TRANSMITTER_V2 = Web3.to_checksum_address(cfg.ARB_MESSAGE_TRANSMITTER_V2)
HEVM_FORWARDER = Web3.to_checksum_address(cfg.CCTP_FORWARDER_HEVM)

BURNS = {
    1: "0x70a60e0215d933ad48fdfb53573e1ba797be1860e7d7fbaa8d815bc765123608",
    2: "0x3b82a6df16c28a1e05eb1b5e737086a5325d242acdf19330fe96e78f0cdfd0e5",
}

# 想定 ABI（CCTP V2 一次ソース signature）
FORWARDER_ABI = [
    {"name": "mintAndForward", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "message", "type": "bytes"}, {"name": "attestation", "type": "bytes"}],
     "outputs": []},
]
TRANSMITTER_ABI = [
    {"name": "receiveMessage", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "message", "type": "bytes"}, {"name": "attestation", "type": "bytes"}],
     "outputs": [{"name": "", "type": "bool"}]},
]


def make_hl_web3():
    p = Web3.HTTPProvider(cfg.HL_EVM_RPC_URL,
                          request_kwargs={"timeout": 20, "headers": {"User-Agent": "Mozilla/5.0"}})
    w3 = Web3(p)
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3


def fetch_iris(burn_tx: str) -> dict:
    url = cfg.IRIS_API_HOST + cfg.IRIS_MESSAGES_PATH.format(source_domain=cfg.ARB_DOMAIN_ID) + f"?transactionHash={burn_tx}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    msgs = (r.json().get("messages") or [])
    return msgs[0] if msgs else {}


def sel(sig: str) -> str:
    return Web3.keccak(text=sig).hex()[:10]


def inspect():
    w3 = make_hl_web3()
    print("=" * 70)
    print(" 段1 read-only: CCTP 回収のための必要データ取得")
    print("=" * 70)
    print(f" HyperEVM connected : {w3.is_connected()}  block={w3.eth.block_number}")
    gas_bal = w3.eth.get_balance(ARB_SENDER_ADDRESS)
    print(f" sender             : {ARB_SENDER_ADDRESS}")
    print(f" HyperEVM gas (HYPE): {Decimal(gas_bal)/Decimal(10**18)}  (raw {gas_bal})")
    print("-" * 70)

    # コントラクト存在確認
    fwd_code = w3.eth.get_code(HEVM_FORWARDER)
    tx_code = w3.eth.get_code(HEVM_MESSAGE_TRANSMITTER_V2)
    print(f" CctpForwarder {HEVM_FORWARDER}: code={len(fwd_code)} bytes "
          f"({'コントラクトあり' if len(fwd_code) > 0 else 'コードなし!'})")
    print(f" MessageTransmitterV2 {HEVM_MESSAGE_TRANSMITTER_V2}: code={len(tx_code)} bytes "
          f"({'コントラクトあり' if len(tx_code) > 0 else 'コードなし!'})")
    print(f" selector mintAndForward(bytes,bytes) = {sel('mintAndForward(bytes,bytes)')}")
    print(f" selector receiveMessage(bytes,bytes) = {sel('receiveMessage(bytes,bytes)')}")
    print("-" * 70)

    out_dir = cfg.TEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp_id, burn in BURNS.items():
        m = fetch_iris(burn)
        status = m.get("status")
        fwd_state = m.get("forwardState")
        dec = m.get("decodedMessage") or {}
        body = (dec.get("decodedMessageBody") or {})
        msg_hex = m.get("message")
        att_hex = m.get("attestation")
        has_att = bool(att_hex and att_hex != "PENDING")
        print(f" [exp_id={exp_id}] burn={burn}")
        print(f"   status={status} forwardState={fwd_state} delayReason={m.get('delayReason')} "
              f"forwardErrorCode={m.get('forwardErrorCode')}")
        print(f"   finalityThresholdExecuted={dec.get('finalityThresholdExecuted')} "
              f"feeExecuted={body.get('feeExecuted')} maxFee={body.get('maxFee')}")
        print(f"   message取得={'YES' if msg_hex else 'NO'}  attestation(署名)取得={'YES' if has_att else 'NO(PENDING)'}")
        # complete のものは message/attestation を保存（段2 で使用）
        if msg_hex and has_att:
            p = out_dir / f"recover_msg_{exp_id}.json"
            p.write_text(json.dumps({"exp_id": exp_id, "burn_tx": burn,
                                     "message": msg_hex, "attestation": att_hex,
                                     "finalityThresholdExecuted": dec.get("finalityThresholdExecuted"),
                                     "feeExecuted": body.get("feeExecuted")}, indent=2))
            print(f"   → message/attestation を保存: {p}")
        print()
    print("=" * 70)
    print(" 段1 完了（read-only）。complete のものは段2 --recover で estimate_gas を試す。")


def estimate_and_maybe_execute(exp_id: int, execute: bool):
    w3 = make_hl_web3()
    p = cfg.TEST_DIR / f"recover_msg_{exp_id}.json"
    if not p.exists():
        print(f"[exp_id={exp_id}] 保存済み message/attestation なし（{p}）。complete でない可能性。スキップ。")
        return
    d = json.loads(p.read_text())
    message = d["message"]
    attestation = d["attestation"]
    msg_bytes = Web3.to_bytes(hexstr=message)
    att_bytes = Web3.to_bytes(hexstr=attestation)
    print("=" * 70)
    print(f" 段2 回収試行 exp_id={exp_id}  (元本は減らない・失敗時ガスのみ)")
    print(f"   message len={len(msg_bytes)}B  attestation len={len(att_bytes)}B")
    print("=" * 70)

    fwd = w3.eth.contract(address=HEVM_FORWARDER, abi=FORWARDER_ABI)
    tx = w3.eth.contract(address=HEVM_MESSAGE_TRANSMITTER_V2, abi=TRANSMITTER_ABI)

    candidates = [
        ("CctpForwarder.mintAndForward", HEVM_FORWARDER,
         fwd.functions.mintAndForward(msg_bytes, att_bytes)),
        ("MessageTransmitterV2.receiveMessage", HEVM_MESSAGE_TRANSMITTER_V2,
         tx.functions.receiveMessage(msg_bytes, att_bytes)),
    ]

    chosen = None
    for label, addr, fn in candidates:
        print(f"\n--- estimate_gas: {label} ({addr}) ---")
        try:
            g = fn.estimate_gas({"from": ARB_SENDER_ADDRESS})
            print(f"   OK: estimate_gas = {g}  → 実行可能")
            chosen = (label, addr, fn, g)
            break
        except Exception as e:
            print(f"   REVERT/ERROR: {type(e).__name__}: {e}")

    if chosen is None:
        print(f"\n[exp_id={exp_id}] どの経路も estimate_gas で revert。自力回収不可（段3）。上の revert 理由を参照。")
        return

    label, addr, fn, g = chosen
    if not execute:
        print(f"\n[exp_id={exp_id}] estimate_gas 成功（{label}）。--execute 未指定のため実送信せず停止。")
        return

    if not ARB_SENDER_PRIVATE_KEY:
        print("[ERROR] 秘密鍵が読めない。実行中止。")
        return

    nonce = w3.eth.get_transaction_count(ARB_SENDER_ADDRESS)
    try:
        gas_price = w3.eth.gas_price
    except Exception:
        gas_price = w3.to_wei(1, "gwei")
    gas_limit = int(g * 1.3)
    print(f"\n[send] contract={addr}")
    print(f"[send] function={label}")
    print(f"[send] args: message({len(msg_bytes)}B), attestation({len(att_bytes)}B)")
    print(f"[send] gas_limit={gas_limit} gas_price={gas_price} nonce={nonce}")
    txd = fn.build_transaction({"from": ARB_SENDER_ADDRESS, "nonce": nonce,
                                "gas": gas_limit, "gasPrice": gas_price, "value": 0})
    signed = w3.eth.account.sign_transaction(txd, private_key=ARB_SENDER_PRIVATE_KEY)
    h = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    print(f"[send] tx sent: {h}  待機中...")
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
    print(f"[result] status={rcpt.status} (1=success) block={rcpt.blockNumber} gasUsed={rcpt.gasUsed} tx={h}")
    if rcpt.status == 1:
        print(f"[result] mint→forward 実行成功。HyperCore credit を確認してください。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="段1 read-only データ取得")
    ap.add_argument("--recover", action="store_true", help="段2 estimate_gas（+ --execute で実送信）")
    ap.add_argument("--exp-id", type=int, default=None, help="対象 exp_id（未指定なら 1,2 両方）")
    ap.add_argument("--execute", action="store_true", help="estimate 成功時に実送信する")
    args = ap.parse_args()

    if args.inspect:
        inspect()
    elif args.recover:
        ids = [args.exp_id] if args.exp_id else [1, 2]
        for i in ids:
            estimate_and_maybe_execute(i, execute=args.execute)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
