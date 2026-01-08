#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperliquid → Arbitrum USDC withdraw レイテンシ測定スクリプト（request→ledger 定義版）

測るもの:
  - withdraw request (nonce) → Arbitrum ブロック → Hyperliquid ledger 反映

レイテンシ定義:
  - latency_sec = request_time_iso → ledger_time_iso
    （＝ユーザーが署名した後〜着金までの時間。UIでボタン押す前の時間は含まない）

実験フロー:
  1. スクリプト起動
  2. ターミナルの指示に従って、Hyperliquid UI から Arbitrum への withdraw を実行
  3. info API で withdraw ledger を検出
  4. ledger の nonce から request_time を復元し、
     Arbitrum tx 情報と合わせて CSV に 1 行追記して終了
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

# .env ロード
load_dotenv()

# =====================================================================
# 環境変数・定数
# =====================================================================

ARB_RPC_URL = os.getenv("ARB_RPC_URL", "https://arb1.arbitrum.io/rpc")
ARB_PRIVATE_KEY = os.getenv("ARB_PRIVATE_KEY")  # 無くても動く

# Web3 セットアップ
w3 = Web3(Web3.HTTPProvider(ARB_RPC_URL))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

if not w3.is_connected():
    raise RuntimeError(f"Failed to connect to Arbitrum RPC: {ARB_RPC_URL}")

# 送信先アドレス（Arbitrum 側の自分のウォレット）
if ARB_PRIVATE_KEY:
    acct = w3.eth.account.from_key(ARB_PRIVATE_KEY)
    ARB_SENDER_ADDRESS = Web3.to_checksum_address(acct.address)
else:
    ARB_SENDER_ADDRESS = Web3.to_checksum_address(
        os.getenv("ARB_SENDER_ADDRESS", os.getenv("HL_USER_ADDRESS", "0x0"))
    )

# USDC コントラクト
ARB_USDC_ADDRESS = Web3.to_checksum_address(
    os.getenv(
        "ARB_USDC_ADDRESS",
        "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    )
)

# Hyperliquid Deposit Bridge 2
HL_DEPOSIT_BRIDGE_ADDRESS = Web3.to_checksum_address(
    os.getenv(
        "HL_DEPOSIT_BRIDGE_ADDRESS",
        "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7",
    )
)

# Hyperliquid アカウント (EVM と同じアドレス)
HL_USER_ADDRESS = Web3.to_checksum_address(
    os.getenv("HL_USER_ADDRESS", ARB_SENDER_ADDRESS)
)

# withdraw のネット額 (手数料 1 USDC は別カラムに記録)
WITHDRAW_AMOUNT_USDC = Decimal(os.getenv("HL_WITHDRAW_AMOUNT_USDC_NET", "5.0"))

# Hyperliquid info エンドポイント
HL_INFO_URL = os.getenv("HL_INFO_URL", "https://api.hyperliquid.xyz/info")

# 結果 CSV
RESULT_DIR = Path("result")
RESULT_CSV_PATH = RESULT_DIR / "withdraw_latency.csv"


# =====================================================================
# データクラス
# =====================================================================

@dataclass
class WithdrawLedger:
    ledger_time_ms: int          # HL ledger time (ms)
    ledger_usdc: Decimal         # withdraw usdc (net)
    fee_usdc: Decimal            # fee usdc
    tx_hash: str                 # Arbitrum tx hash
    raw_event: Dict[str, Any]    # 生の ledger event
    local_detect_ns: int         # ローカルで検出した時刻 (ns)
    request_time_ms: int         # nonce から復元した request time (ms)


@dataclass
class WithdrawTxInfo:
    tx_hash: str
    from_address: str
    to_address: str
    amount_raw: int
    block_number: int
    block_timestamp: int
    gas_used: int
    gas_price_wei: int
    tx_fee_eth: Decimal


# =====================================================================
# Hyperliquid info ポーリング
# =====================================================================

def fetch_user_non_funding_ledgers(
    user: str,
    start_time_ms: int,
    end_time_ms: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Hyperliquid info API (userNonFundingLedgerUpdates) から
    指定期間の ledger を取得する。
    """
    payload: Dict[str, Any] = {
        "type": "userNonFundingLedgerUpdates",
        "user": user,
        "startTime": start_time_ms,
    }
    if end_time_ms is not None:
        payload["endTime"] = end_time_ms

    resp = requests.post(HL_INFO_URL, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "data" in data:
        arr = data["data"]
    else:
        arr = data

    if not isinstance(arr, list):
        raise RuntimeError(f"Unexpected response from info API: {data}")

    return arr


def wait_for_withdraw_ledger(
    user: str,
    target_amount: Decimal,
    experiment_start_ms: int,
    timeout_sec: int = 900,
    poll_interval_sec: float = 2.0,
) -> WithdrawLedger:
    """
    最初の withdraw ledger (usdc ≈ target_amount) を見つけるまで
    info API をポーリングする。

    レイテンシの起点として、delta.nonce を "request time" として使う。
    （nonce は μs 単位のタイムスタンプとみなす）
    """
    print(
        f"Waiting Hyperliquid withdraw ledger via info API "
        f"(timeout {timeout_sec} sec, interval {poll_interval_sec} sec)..."
    )

    deadline = time.time() + timeout_sec

    while True:
        now = time.time()
        if now > deadline:
            raise TimeoutError("Timed out waiting for withdraw ledger on Hyperliquid.")

        end_time_ms = int(now * 1000)

        ledgers = fetch_user_non_funding_ledgers(
            user=user,
            start_time_ms=experiment_start_ms,
            end_time_ms=end_time_ms,
        )

        for ev in ledgers:
            delta = ev.get("delta", {})
            if delta.get("type") != "withdraw":
                continue

            usdc_str = delta.get("usdc")
            fee_str = delta.get("fee")
            nonce_raw = delta.get("nonce")

            if nonce_raw is None:
                continue

            try:
                usdc_val = Decimal(usdc_str)
            except Exception:
                continue

            # usdc ≈ target_amount (誤差 0.000001 以内)
            if abs(usdc_val - target_amount) > Decimal("0.000001"):
                continue

            fee_val = Decimal(fee_str) if fee_str is not None else Decimal("0")
            # nonce は μs とみなして ms に変換
            request_time_ms = int(int(nonce_raw) / 1000)

            ledger_time_ms = int(ev.get("time", 0))
            tx_hash = ev.get("hash")
            local_detect_ns = time.time_ns()

            request_time_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(request_time_ms / 1000)
            )
            ledger_time_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(ledger_time_ms / 1000)
            )
            local_detect_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(local_detect_ns / 1e9)
            )

            print("===== Detected withdraw ledger on Hyperliquid (via info API) =====")
            print(f"  request_time_ms  : {request_time_ms}")
            print(f"  request_time_iso : {request_time_iso}")
            print(f"  ledger_time_ms   : {ledger_time_ms}")
            print(f"  ledger_time_iso  : {ledger_time_iso}")
            print(f"  ledger_usdc      : {usdc_val}")
            print(f"  fee_usdc         : {fee_val}")
            print(f"  tx_hash          : {tx_hash}")
            print(f"  raw_event        : {ev}")
            print(f"  local_detect_iso : {local_detect_iso}")

            return WithdrawLedger(
                ledger_time_ms=ledger_time_ms,
                ledger_usdc=usdc_val,
                fee_usdc=fee_val,
                tx_hash=tx_hash,
                raw_event=ev,
                local_detect_ns=local_detect_ns,
                request_time_ms=request_time_ms,
            )

        # まだ見つからない場合は少し待つ
        time.sleep(poll_interval_sec)


# =====================================================================
# Arbitrum トランザクション情報取得
# =====================================================================

def fetch_withdraw_tx_info(
    w3: Web3,
    ledger: WithdrawLedger,
    amount_usdc_net: Decimal,
) -> WithdrawTxInfo:
    """
    ledger.hash = Arbitrum 上の withdraw tx の hash なので、
    これを元にトランザクション詳細を取得する。
    """
    tx_hash = ledger.tx_hash
    if not tx_hash:
        raise RuntimeError("Ledger does not contain tx hash.")

    print("Fetching Arbitrum tx info from hash on ledger...")
    tx = w3.eth.get_transaction(tx_hash)
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    block = w3.eth.get_block(receipt["blockNumber"])

    from_address = Web3.to_checksum_address(tx["from"])
    to_address = Web3.to_checksum_address(tx["to"])

    gas_used = int(receipt["gasUsed"])
    gas_price_wei = int(tx.get("gasPrice") or receipt.get("effectiveGasPrice", 0))
    tx_fee_eth = (Decimal(gas_used) * Decimal(gas_price_wei)) / Decimal(10**18)

    block_timestamp = int(block["timestamp"])
    amount_raw = int(amount_usdc_net * (10**6))  # USDC 6 decimals

    print("===== Arbitrum withdraw tx info =====")
    print(f"  tx_hash          : {tx_hash}")
    print(f"  from_address     : {from_address}")
    print(f"  to_address       : {to_address}")
    print(f"  amount_raw       : {amount_raw}")
    print(f"  block_number     : {receipt['blockNumber']}")
    print(
        f"  block_timestamp  : {block_timestamp} -> "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(block_timestamp))}"
    )
    print(f"  gas_used         : {gas_used}")
    print(f"  gas_price_wei    : {gas_price_wei}")
    print(f"  tx_fee_eth       : {tx_fee_eth}")

    return WithdrawTxInfo(
        tx_hash=tx_hash,
        from_address=from_address,
        to_address=to_address,
        amount_raw=amount_raw,
        block_number=int(receipt["blockNumber"]),
        block_timestamp=block_timestamp,
        gas_used=gas_used,
        gas_price_wei=gas_price_wei,
        tx_fee_eth=tx_fee_eth,
    )


# =====================================================================
# CSV 保存
# =====================================================================

def save_withdraw_csv(
    experiment_start_ns: int,
    ledger: WithdrawLedger,
    tx_info: WithdrawTxInfo,
):
    RESULT_DIR.mkdir(exist_ok=True)

    experiment_start_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(experiment_start_ns / 1e9)
    )
    request_time_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(ledger.request_time_ms / 1000)
    )
    ledger_time_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(ledger.ledger_time_ms / 1000)
    )
    arb_block_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(tx_info.block_timestamp)
    )

    # レイテンシ
    block_time_ms = tx_info.block_timestamp * 1000
    latency_request_to_block_sec = (block_time_ms - ledger.request_time_ms) / 1000.0
    latency_request_to_ledger_sec = (ledger.ledger_time_ms - ledger.request_time_ms) / 1000.0

    header = [
        "experiment_start_iso",
        "request_time_iso",
        "tx_hash",
        "from_address",
        "to_address",
        "amount_usdc_net",
        "amount_raw",
        "fee_usdc",
        "arb_block_number",
        "arb_block_timestamp_iso",
        "gas_used",
        "gas_price_wei",
        "tx_fee_eth",
        "ledger_time_iso",
        "latency_sec",  # request -> ledger
        "latency_request_to_block_sec",
    ]

    write_header = not RESULT_CSV_PATH.exists()

    with RESULT_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        writer.writerow(
            [
                experiment_start_iso,
                request_time_iso,
                tx_info.tx_hash,
                tx_info.from_address,
                tx_info.to_address,
                f"{WITHDRAW_AMOUNT_USDC:.1f}",
                tx_info.amount_raw,
                f"{ledger.fee_usdc:.1f}",
                tx_info.block_number,
                arb_block_iso,
                tx_info.gas_used,
                tx_info.gas_price_wei,
                f"{tx_info.tx_fee_eth:.18f}",
                ledger_time_iso,
                f"{latency_request_to_ledger_sec:.6f}",
                f"{latency_request_to_block_sec:.6f}",
            ]
        )

    print("[INFO] 1 row appended to", str(RESULT_CSV_PATH))


# =====================================================================
# メイン
# =====================================================================

def main():
    print("===== Hyperliquid → Arbitrum USDC withdraw latency measurement (request→ledger 定義) =====")
    print(f"HL user address           : {HL_USER_ADDRESS}")
    print(f"Arbitrum dest address     : {ARB_SENDER_ADDRESS}")
    print(f"Arbitrum RPC              : {ARB_RPC_URL}")
    print(f"USDC token                : {ARB_USDC_ADDRESS}")
    print(f"HL Deposit Bridge 2       : {HL_DEPOSIT_BRIDGE_ADDRESS}")
    print(f"Net withdraw amount USDC  : {WITHDRAW_AMOUNT_USDC}")

    chain_id = w3.eth.chain_id
    print(f"\nConnected Arbitrum. chainId = {chain_id}\n")

    # 実験開始時刻（参考用）
    experiment_start_ns = time.time_ns()
    experiment_start_ms = experiment_start_ns // 1_000_000

    print(f"Experiment start (ns)  : {experiment_start_ns}")
    print(
        "Experiment start (ISO) : "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(experiment_start_ns / 1e9))}"
    )
    print()
    print("※ このメッセージが出たら、Hyperliquid UI から Arbitrum への withdraw を実行してください。")
    print("  （perps 口座から Arbitrum wallet への USDC withdraw, net 5 USDC, fee 1 USDC を想定）")
    print()

    # 1) withdraw ledger 検出（ここで nonce = request_time を取得）
    ledger = wait_for_withdraw_ledger(
        user=HL_USER_ADDRESS,
        target_amount=WITHDRAW_AMOUNT_USDC,
        experiment_start_ms=experiment_start_ms,
        timeout_sec=900,
        poll_interval_sec=2.0,
    )

    # 2) Arbitrum tx 情報取得
    tx_info = fetch_withdraw_tx_info(
        w3=w3,
        ledger=ledger,
        amount_usdc_net=WITHDRAW_AMOUNT_USDC,
    )

    # 3) 各レイテンシを計算して表示
    block_time_ms = tx_info.block_timestamp * 1000
    latency_request_to_block_sec = (block_time_ms - ledger.request_time_ms) / 1000.0
    latency_request_to_ledger_sec = (ledger.ledger_time_ms - ledger.request_time_ms) / 1000.0
    latency_block_to_ledger_sec = (ledger.ledger_time_ms - block_time_ms) / 1000.0

    print("===== Latency (request → block / ledger) =====")
    print(f"  request_time_ms              : {ledger.request_time_ms}")
    print(f"  block_time_ms                : {block_time_ms}")
    print(f"  ledger_time_ms               : {ledger.ledger_time_ms}")
    print(f"  request → block   [sec]      : {latency_request_to_block_sec:.3f}")
    print(f"  request → ledger  [sec]      : {latency_request_to_ledger_sec:.3f}")
    print(f"  block   → ledger  [sec]      : {latency_block_to_ledger_sec:.3f}")
    print("  (CSV の latency_sec は request → ledger を採用)")
    print("=============================================\n")

    # 4) CSV 保存
    save_withdraw_csv(
        experiment_start_ns=experiment_start_ns,
        ledger=ledger,
        tx_info=tx_info,
    )


if __name__ == "__main__":
    main()
