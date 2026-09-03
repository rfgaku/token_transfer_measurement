#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recover_cctp.py — withdraw の自動 forwarding が失敗した場合の手動 mint（回収）用。

★通常は不要: action.data="0x" による Arbitrum 宛 自動 forwarding が成功していれば、
  Arbitrum 側で勝手に mint される。本スクリプトは「Arbitrum で mint されなかった」場合のみ使う。

Arbitrum の MessageTransmitterV2.receiveMessage(bytes message, bytes attestation) を自力実行する。
message / attestation は Iris の GET /v2/messages/19?transactionHash=<hevm_burn_tx> から取得して渡す。

使い方:
  ドライラン（送信しない・tx を組むだけ）:
    python3 -u t4_cctp/withdraw/recover_cctp.py --message 0x.. --attestation 0x..
  実送信:
    python3 -u t4_cctp/withdraw/recover_cctp.py --message 0x.. --attestation 0x.. --broadcast
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

from t4_cctp.withdraw import config as cfg

load_dotenv()

MESSAGE_TRANSMITTER_ABI = [
    {"name": "receiveMessage", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "message", "type": "bytes"}, {"name": "attestation", "type": "bytes"}],
     "outputs": [{"name": "", "type": "bool"}]},
]


def to_bytes(hexstr: str) -> bytes:
    return bytes.fromhex(hexstr[2:] if hexstr.startswith("0x") else hexstr)


def main():
    ap = argparse.ArgumentParser(description="CCTP withdraw 手動回収（Arbitrum receiveMessage）")
    ap.add_argument("--message", required=True, help="Iris の message hex")
    ap.add_argument("--attestation", required=True, help="Iris の attestation hex")
    ap.add_argument("--broadcast", action="store_true", help="実送信（未指定はドライラン）")
    args = ap.parse_args()

    sender = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    pk = os.environ["ARB_SENDER_PRIVATE_KEY"]

    w3 = Web3(Web3.HTTPProvider(cfg.ARB_RPC_URL))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    mt = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_MESSAGE_TRANSMITTER_V2),
                         abi=MESSAGE_TRANSMITTER_ABI)

    msg = to_bytes(args.message)
    att = to_bytes(args.attestation)
    fn = mt.functions.receiveMessage(msg, att)

    print("=" * 64)
    print(" CCTP withdraw 手動回収（Arbitrum MessageTransmitterV2.receiveMessage）")
    print("=" * 64)
    print(f" connected      : {w3.is_connected()} chainId={w3.eth.chain_id}")
    print(f" to(MsgTransmit): {cfg.ARB_MESSAGE_TRANSMITTER_V2}")
    print(f" from(sender)   : {sender}")
    print(f" message len    : {len(msg)} bytes")
    print(f" attestation len: {len(att)} bytes")

    gas_price = int(w3.eth.gas_price * 1.5)
    try:
        gas = int(fn.estimate_gas({"from": sender}) * 1.2)
    except Exception as e:
        gas = 300000
        print(f" [WARN] estimate_gas 失敗（既に mint 済み等の可能性）: {e}  → gas={gas} で続行")
    print(f" gasPrice(x1.5) : {gas_price} wei")
    print(f" gas            : {gas}")

    if not args.broadcast:
        print("-" * 64)
        print(" DRY RUN: tx を組みましたが送信しません（--broadcast で実送信）。")
        print("=" * 64)
        return

    nonce = w3.eth.get_transaction_count(sender)
    tx = fn.build_transaction({"from": sender, "nonce": nonce, "gasPrice": gas_price, "gas": gas, "value": 0})
    signed = w3.eth.account.sign_transaction(tx, private_key=pk)
    h = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    print(f" [SENT] receiveMessage tx={h}  receipt 待ち ...")
    receipt = w3.eth.wait_for_transaction_receipt(h)
    print(f" [DONE] block={receipt.blockNumber} status={receipt.status} (1=成功)")
    print("=" * 64)


if __name__ == "__main__":
    main()
