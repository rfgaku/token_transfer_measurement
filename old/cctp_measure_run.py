#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitrum → Hyperliquid USDC deposit レイテンシ測定スクリプト（Deposit Bridge 2 経由）

やっていること:
1. Hyperliquid Info WebSocket に接続し、userNonFundingLedgerUpdates を購読
2. 実験開始時刻を記録
3. Arbitrum 上で USDC → Hyperliquid Deposit Bridge 2 へ 5 USDC を transfer()
4. その後、Hyperliquid 側の最初の deposit ledger を待ち、時刻差を計測
5. 結果を result/deposit_latency.csv に追記保存（ガス代付き）

※ CCTP TokenMessenger に対する depositForBurn は一切使わない。
   手動 deposit と同じ「USDC transfer → HL Deposit Bridge」フローに揃えている。
"""

import os
import threading
import time
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

from hyperliquid.info import Info

# =====================================================================
# 環境変数読み込み
# =====================================================================

load_dotenv()

ARB_RPC_URL = os.getenv("ARB_RPC_URL", "https://arb1.arbitrum.io/rpc")

ARB_SENDER_ADDRESS = Web3.to_checksum_address(
    os.environ["ARB_SENDER_ADDRESS"]
)
ARB_SENDER_PRIVATE_KEY = os.environ["ARB_SENDER_PRIVATE_KEY"]

# Arbitrum の USDC コントラクト (0xaf88...268e5831)
ARB_USDC_ADDRESS = Web3.to_checksum_address(
    os.getenv(
        "ARB_USDC_ADDRESS",
        "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    )
)

# Hyperliquid Deposit Bridge 2 のアドレス
# 手動 deposit 成功 tx の "To" に出ていたアドレス
HL_DEPOSIT_BRIDGE_ADDRESS = Web3.to_checksum_address(
    os.getenv(
        "HL_DEPOSIT_BRIDGE_ADDRESS",
        "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7",
    )
)

# テスト入金額（USDC）
DEPOSIT_AMOUNT_USDC = Decimal(os.getenv("ARB_CCTP_AMOUNT_USDC", "5.0"))

# Hyperliquid のアカウントアドレス（EVM と同じで OK）
HL_USER_ADDRESS = os.getenv("HL_USER_ADDRESS", ARB_SENDER_ADDRESS)

# 結果 CSV のパス
RESULT_DIR = Path("result")
RESULT_CSV_PATH = RESULT_DIR / "deposit_latency.csv"

# =====================================================================
# Web3 / コントラクト設定
# =====================================================================

ERC20_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "transfer",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
]


def make_web3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(ARB_RPC_URL))
    # Arbitrum は PoS 系扱い
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    assert w3.is_connected(), "Failed to connect Arbitrum RPC"
    return w3


# =====================================================================
# Hyperliquid WebSocket リスナー
# =====================================================================

@dataclass
class DepositResult:
    found: bool = False
    ledger_time_ms: int | None = None
    ledger_usdc: Decimal | None = None
    raw_event: dict | None = None


class HlListener:
    """
    userNonFundingLedgerUpdates を購読して、
    実験開始後に最初に来た deposit イベントを捕まえる。
    """

    def __init__(self, user_address: str, amount_usdc: Decimal, experiment_start_ms: int):
        self.user_address = user_address.lower()
        self.amount_usdc = amount_usdc
        self.experiment_start_ms = experiment_start_ms

        self.result = DepositResult()
        self._event = threading.Event()

    # Info.subscribe から呼ばれる
    def handle_msg(self, msg: dict):
        ch = msg.get("channel")
        if ch != "userNonFundingLedgerUpdates":
            return

        data = msg.get("data", {})
        if not data:
            return

        updates = data.get("nonFundingLedgerUpdates") or []

        for upd in updates:
            try:
                t_ms = int(upd.get("time", 0))
            except Exception:
                continue

            # 実験開始より前のものは無視
            if t_ms < self.experiment_start_ms:
                continue

            delta = upd.get("delta", {})
            if delta.get("type") != "deposit":
                continue

            # usdc 金額でフィルタ（5USDC 付近）
            try:
                usdc_amount = Decimal(str(delta.get("usdc", "0")))
            except Exception:
                continue

            if abs(usdc_amount - self.amount_usdc) > Decimal("0.0001"):
                continue

            # user は空のこともあるので、ここではチェックしない。
            # 最初の 1 件を採用。
            self.result.found = True
            self.result.ledger_time_ms = t_ms
            self.result.ledger_usdc = usdc_amount
            self.result.raw_event = upd
            self._event.set()
            return

    def wait_for_deposit(self, timeout_sec: int = 30) -> DepositResult:
        self._event.wait(timeout=timeout_sec)
        return self.result


# =====================================================================
# Arbitrum 側: USDC → Hyperliquid Deposit Bridge 2 へ transfer
# =====================================================================

@dataclass
class DepositTxInfo:
    tx_hash: str
    local_send_time_ns: int
    block_number: int
    block_timestamp: int
    gas_used: int
    gas_price_wei: int
    tx_fee_eth: Decimal
    amount_raw: int


def send_usdc_deposit(w3: Web3) -> DepositTxInfo:
    sender = ARB_SENDER_ADDRESS
    usdc = w3.eth.contract(address=ARB_USDC_ADDRESS, abi=ERC20_ABI)

    amount_raw = int(DEPOSIT_AMOUNT_USDC * (10 ** 6))  # USDC 6 decimals

    balance = usdc.functions.balanceOf(sender).call()
    if balance < amount_raw:
        raise RuntimeError(
            f"USDC balance insufficient: balance={balance}, needed={amount_raw}"
        )

    nonce = w3.eth.get_transaction_count(sender)

    # tx 準備
    # 現在のネットワーク状況に合わせて、ガス価格に少しマージンを乗せる
    base_gas_price = w3.eth.gas_price
    safe_gas_price = int(base_gas_price * 2)  # 2倍くらいにして baseFee を確実に上回る
    tx = usdc.functions.transfer(
        HL_DEPOSIT_BRIDGE_ADDRESS, amount_raw
    ).build_transaction(
        {
            "from": sender,
            "nonce": nonce,
            "gasPrice": safe_gas_price,
        }
    )

    # ガス推定 + 上乗せ
    gas_estimate = w3.eth.estimate_gas(tx)
    tx["gas"] = int(gas_estimate * 1.2)

    # 送信直前にローカル時間を取る
    local_send_time_ns = time.time_ns()

    signed = w3.eth.account.sign_transaction(tx, private_key=ARB_SENDER_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction).hex()

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    block = w3.eth.get_block(receipt.blockNumber)

    gas_used = receipt.gasUsed
    # EIP-1559 対応: effectiveGasPrice があればそちらを使う
    gas_price_wei = getattr(receipt, "effectiveGasPrice", tx["gasPrice"])
    tx_fee_eth = Decimal(gas_used) * Decimal(gas_price_wei) / Decimal(10**18)

    return DepositTxInfo(
        tx_hash=tx_hash,
        local_send_time_ns=local_send_time_ns,
        block_number=receipt.blockNumber,
        block_timestamp=block.timestamp,
        gas_used=gas_used,
        gas_price_wei=gas_price_wei,
        tx_fee_eth=tx_fee_eth,
        amount_raw=amount_raw,
    )


# =====================================================================
# 結果の CSV 保存
# =====================================================================

def save_result_csv(
    experiment_start_ns: int,
    deposit_tx: DepositTxInfo,
    hl_result: DepositResult,
    latency_local_ns: int,
):
    RESULT_DIR.mkdir(exist_ok=True)

    experiment_start_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(experiment_start_ns / 1e9)
    )
    local_send_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(deposit_tx.local_send_time_ns / 1e9)
    )
    arb_block_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(deposit_tx.block_timestamp)
    )
    hl_ledger_iso = (
        time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime((hl_result.ledger_time_ms or 0) / 1000)
        )
        if hl_result.ledger_time_ms is not None
        else ""
    )
    latency_sec = latency_local_ns / 1e9

    header = [
        "experiment_start_iso",
        "local_send_iso",
        "tx_hash",
        "from_address",
        "to_address",
        "amount_usdc",
        "amount_raw",
        "arb_block_number",
        "arb_block_timestamp_iso",
        "gas_used",
        "gas_price_wei",
        "tx_fee_eth",
        "hl_ledger_time_iso",
        "latency_sec",
    ]

    write_header = not RESULT_CSV_PATH.exists()

    with RESULT_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        writer.writerow(
            [
                experiment_start_iso,
                local_send_iso,
                deposit_tx.tx_hash,
                ARB_SENDER_ADDRESS,
                HL_DEPOSIT_BRIDGE_ADDRESS,
                str(DEPOSIT_AMOUNT_USDC),
                deposit_tx.amount_raw,
                deposit_tx.block_number,
                arb_block_iso,
                deposit_tx.gas_used,
                deposit_tx.gas_price_wei,
                f"{deposit_tx.tx_fee_eth:.18f}",
                hl_ledger_iso,
                f"{latency_sec:.6f}",
            ]
        )


# =====================================================================
# メインフロー
# =====================================================================

def main():
    print("===== Arbitrum → Hyperliquid USDC Deposit Latency Measurement =====")

    # 1) Hyperliquid WS リスナー起動前に実験開始時刻を決める
    experiment_start_ns = time.time_ns()
    experiment_start_ms = experiment_start_ns // 1_000_000

    print(f"HL User Address : {HL_USER_ADDRESS}")
    print(f"Experiment start (local) ns : {experiment_start_ns}")
    print(
        f"Experiment start (local) ISO: "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(experiment_start_ns / 1e9))}"
    )
    print(f"Deposit amount USDC        : {DEPOSIT_AMOUNT_USDC}")
    print()

    # 2) Hyperliquid Info WebSocket を別スレッドで開始
    info = Info()
    hl_listener = HlListener(
        user_address=HL_USER_ADDRESS,
        amount_usdc=DEPOSIT_AMOUNT_USDC,
        experiment_start_ms=experiment_start_ms,
    )

    sub = {
        "type": "userNonFundingLedgerUpdates",
        "user": HL_USER_ADDRESS,
    }

    def ws_thread():
        # これは戻ってこない前提
        info.subscribe(sub, hl_listener.handle_msg)

    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()
    print("Connected Hyperliquid Info WS, listening for deposit events...")
    print()

    # 3) Arbitrum で USDC → HL Deposit Bridge 2 へ transfer 実行
    w3 = make_web3()
    print(f"Connecting to Arbitrum RPC: {ARB_RPC_URL}")
    print(f"Connected. chainId = {w3.eth.chain_id}")
    print()
    print("===== Sending USDC deposit (ERC-20 transfer to HL Deposit Bridge 2) =====")
    print(f"  sender          : {ARB_SENDER_ADDRESS}")
    print(f"  usdc_address    : {ARB_USDC_ADDRESS}")
    print(f"  deposit_bridge  : {HL_DEPOSIT_BRIDGE_ADDRESS}")
    print(f"  amount_usdc     : {DEPOSIT_AMOUNT_USDC}")
    print()

    deposit_tx = send_usdc_deposit(w3)

    print(f"[deposit] tx_hash               = {deposit_tx.tx_hash}")
    print(f"[deposit] local_send_time_ns    = {deposit_tx.local_send_time_ns}")
    print(
        f"[deposit] local_send_time_iso   = "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(deposit_tx.local_send_time_ns / 1e9))}"
    )
    print(f"[deposit] blockNumber           = {deposit_tx.block_number}")
    print(
        f"[deposit] block_timestamp       = {deposit_tx.block_timestamp} "
        f"-> {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(deposit_tx.block_timestamp))}"
    )
    print(f"[deposit] gas_used              = {deposit_tx.gas_used}")
    print(f"[deposit] gas_price_wei         = {deposit_tx.gas_price_wei}")
    print(f"[deposit] tx_fee_eth            = {deposit_tx.tx_fee_eth}")
    print()

    # 4) Hyperliquid 側の deposit ledger を待機（最大 30 秒）
    print(
        "Waiting Hyperliquid deposit ledger...\n"
        f"  condition: type=deposit, usdc≈{DEPOSIT_AMOUNT_USDC}\n"
        "  timeout : 30 sec"
    )
    result = hl_listener.wait_for_deposit(timeout_sec=30)

    if not result.found:
        print("[ERROR] 30 秒以内に対象 deposit ledger を検出できませんでした。")
        # WebSocket リスナーなどのバックグラウンドスレッドが
        # 残っていてもプロセス全体を確実に終了させるため、
        # os._exit(1) で即時終了する。
        os._exit(1)

    # 5) レイテンシ計算
    ledger_ns_from_epoch = (result.ledger_time_ms or 0) * 1_000_000
    latency_local_ns = ledger_ns_from_epoch - deposit_tx.local_send_time_ns

    print()
    print("===== Detected target deposit on Hyperliquid =====")
    print(
        f" ledger_time (HL ms) : {result.ledger_time_ms} "
        f"-> {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(result.ledger_time_ms / 1000))}"
    )
    print(f" usdc (ledger)       : {result.ledger_usdc}")
    print(f" raw_event           : {result.raw_event}")
    print()
    print("===== Latency (rough) =====")
    print(f" local_send_time_ns      : {deposit_tx.local_send_time_ns}")
    print(f" hl_ledger_time_ns       : {ledger_ns_from_epoch}")
    print(f" Δ (ledger - send) [ns]  : {latency_local_ns}")
    print(f" Δ [sec]                 : {latency_local_ns / 1e9:.3f}")
    print("============================")

    # 6) CSV に保存
    save_result_csv(
        experiment_start_ns=experiment_start_ns,
        deposit_tx=deposit_tx,
        hl_result=result,
        latency_local_ns=latency_local_ns,
    )

    # CSV 書き込みと標準出力が完了したら、残っている
    # WebSocket 関連スレッドに関わらずプロセスを確実に終了させる。
    # （os._exit は Python の通常のシャットダウン処理をスキップするが、
    #  上の with ブロックでファイルはすでにクローズ済みなので問題ない）
    os._exit(0)


if __name__ == "__main__":
    main()
