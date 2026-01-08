#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deposit_latency_measure.py
Arbitrum → Hyperliquid USDC deposit レイテンシ測定スクリプト（Gas自動見積もり・高耐久版）

修正内容:
1. 固定のGas Limit(300,000)を廃止。
2. 実行時に `estimate_gas` で必要量を自動計算し、さらに1.2倍のバッファを乗せて送信する。
3. これにより "intrinsic gas too low" エラーを恒久的に回避する。
"""

import os
import threading
import time
import csv
import json
import websocket
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

# =====================================================================
# 環境変数・設定
# =====================================================================

load_dotenv()

# Arbitrum RPC
ARB_RPC_URL = os.getenv("ARB_RPC_URL", "https://arb1.arbitrum.io/rpc")

# Hyperliquid EVM RPC
HL_EVM_RPC_URL = "https://rpc.hyperliquid.xyz/evm"

try:
    ARB_SENDER_ADDRESS = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    ARB_SENDER_PRIVATE_KEY = os.environ["ARB_SENDER_PRIVATE_KEY"]
except KeyError as e:
    raise KeyError(f"環境変数 {e} が読み込めませんでした。.envファイルを確認してください。")

# Arbitrum USDC Contract
ARB_USDC_ADDRESS = Web3.to_checksum_address(
    os.getenv("ARB_USDC_ADDRESS", "0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
)

# Hyperliquid Deposit Bridge 2 Address
HL_DEPOSIT_BRIDGE_ADDRESS = Web3.to_checksum_address(
    os.getenv("HL_DEPOSIT_BRIDGE_ADDRESS", "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7")
)

# Deposit Amount
DEPOSIT_AMOUNT_USDC = Decimal(os.getenv("ARB_CCTP_AMOUNT_USDC", "5.0"))

# Hyperliquid Account Address
HL_USER_ADDRESS = os.getenv("HL_USER_ADDRESS", ARB_SENDER_ADDRESS)

# Result CSV Path
RESULT_DIR = Path("result")
RESULT_CSV_PATH = RESULT_DIR / "deposit_latency.csv"

# =====================================================================
# Web3 / Helper
# =====================================================================

ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view", "inputs": [{"name": "account", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "transfer", "type": "function", "stateMutability": "nonpayable", "inputs": [{"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}], "outputs": [{"name": "", "type": "bool"}]},
]

def make_arb_web3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(ARB_RPC_URL))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3

def make_hl_web3() -> Web3:
    provider = Web3.HTTPProvider(
        HL_EVM_RPC_URL,
        request_kwargs={'timeout': 10, 'headers': {'User-Agent': 'Mozilla/5.0'}}
    )
    w3 = Web3(provider)
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3

def get_next_experiment_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ids = [int(row["experiment_id"]) for row in reader if row.get("experiment_id")]
            return max(ids) + 1 if ids else 1
    except:
        return 1

# =====================================================================
# Direct WebSocket Listener
# =====================================================================

@dataclass
class DepositResult:
    found: bool = False
    ledger_time_ms: int | None = None
    ledger_usdc: Decimal | None = None
    raw_event: dict | None = None
    hl_block_number: int = 0
    hl_block_timestamp_ms: int = 0
    hl_tx_hash: str = "N/A"

class DirectHlListener:
    def __init__(self, user_address: str, amount_usdc: Decimal, experiment_start_ms: int):
        self.user_address = user_address.lower()
        self.amount_usdc = amount_usdc
        self.experiment_start_ms = experiment_start_ms
        self.result = DepositResult()
        self._event = threading.Event()
        self.ws = None
        self.hl_w3 = None

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
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

                if t_ms < self.experiment_start_ms:
                    continue

                delta = upd.get("delta", {})
                if delta.get("type") != "deposit":
                    continue

                try:
                    usdc_amount = Decimal(str(delta.get("usdc", "0")))
                except Exception:
                    continue

                if abs(usdc_amount - self.amount_usdc) > Decimal("0.0001"):
                    continue

                print(f"\n[WS-SUCCESS] Deposit Event Found: {delta}")
                
                hl_blk_num = 0
                hl_blk_ts_ms = 0
                
                if self.hl_w3 is None:
                    try:
                        self.hl_w3 = make_hl_web3()
                    except: pass

                if self.hl_w3:
                    try:
                        block = self.hl_w3.eth.get_block('latest')
                        hl_blk_num = block.number
                        hl_blk_ts_ms = block.timestamp * 1000
                    except Exception:
                        pass

                tx_hash = upd.get("hash", "Validator-Signed-Event")

                self.result.found = True
                self.result.ledger_time_ms = t_ms
                self.result.ledger_usdc = usdc_amount
                self.result.raw_event = upd
                self.result.hl_block_number = hl_blk_num
                self.result.hl_block_timestamp_ms = hl_blk_ts_ms
                self.result.hl_tx_hash = tx_hash
                
                self._event.set()
                ws.close()
                
        except Exception as e:
            print(f"[WS] Message Error: {e}")

    def on_error(self, ws, error):
        print(f"[WS] Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        pass

    def on_open(self, ws):
        print("[WS] Connected. Subscribing...")
        sub_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "userNonFundingLedgerUpdates",
                "user": self.user_address
            }
        }
        ws.send(json.dumps(sub_msg))

    def _run_ws(self):
        while not self.result.found:
            try:
                websocket.enableTrace(False)
                self.ws = websocket.WebSocketApp(
                    "wss://api.hyperliquid.xyz/ws",
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                self.ws.run_forever()
            except Exception as e:
                print(f"[WS] Setup Error: {e}")
            
            if not self.result.found:
                print("[WS] Connection lost. Retrying in 5s...")
                time.sleep(5)

    def start(self):
        wst = threading.Thread(target=self._run_ws)
        wst.daemon = True
        wst.start()

    def wait_for_deposit(self, timeout_sec: int = 900) -> DepositResult:
        self._event.wait(timeout=timeout_sec)
        return self.result

# =====================================================================
# Arbitrum 送金処理 (Gas自動計算)
# =====================================================================

@dataclass
class DepositTxInfo:
    tx_hash: str
    local_send_time_ns: int
    block_number: int
    block_timestamp_ms: int
    gas_used: int
    gas_price_wei: int
    tx_fee_eth: Decimal
    amount_raw: int

def send_usdc_deposit(w3: Web3) -> DepositTxInfo:
    sender = ARB_SENDER_ADDRESS
    usdc = w3.eth.contract(address=ARB_USDC_ADDRESS, abi=ERC20_ABI)

    amount_raw = int(DEPOSIT_AMOUNT_USDC * (10 ** 6))

    try:
        balance = usdc.functions.balanceOf(sender).call()
        if balance < amount_raw:
            raise RuntimeError(f"USDC balance insufficient: {balance} < {amount_raw}")
    except Exception as e:
        print(f"[WARN] Balance check skipped (RPC error): {e}")

    nonce = w3.eth.get_transaction_count(sender)
    base_gas_price = w3.eth.gas_price
    safe_gas_price = int(base_gas_price * 1.5)
    
    # ---------------------------------------------------------
    # 【修正】Gas Limitを動的に見積もる (固定値 300,000 -> estimate_gas)
    # ---------------------------------------------------------
    tx_params_base = {
        "from": sender,
        "nonce": nonce,
        "gasPrice": safe_gas_price,
        "value": 0
    }
    
    try:
        # チェーンに「この取引にはどれくらいGasが必要？」と聞く
        estimated_gas = usdc.functions.transfer(HL_DEPOSIT_BRIDGE_ADDRESS, amount_raw).estimate_gas(tx_params_base)
        # 念のため 1.2倍 のバッファを持たせる
        safe_gas_limit = int(estimated_gas * 1.2)
        print(f"[Info] Gas Estimated: {estimated_gas} -> Limit: {safe_gas_limit}")
    except Exception as e:
        print(f"[WARN] Gas estimation failed: {e}. Using fallback 1,000,000.")
        safe_gas_limit = 1000000 # 失敗時は大きめの値を設定

    # トランザクション構築
    tx = usdc.functions.transfer(HL_DEPOSIT_BRIDGE_ADDRESS, amount_raw).build_transaction({
        **tx_params_base,
        "gas": safe_gas_limit
    })

    local_send_time_ns = time.time_ns()
    
    signed = w3.eth.account.sign_transaction(tx, private_key=ARB_SENDER_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction).hex()

    print(f"[Arbitrum] Transaction sent: {tx_hash}. Waiting for receipt...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    block = w3.eth.get_block(receipt.blockNumber)

    gas_used = receipt.gasUsed
    gas_price_wei = getattr(receipt, "effectiveGasPrice", tx["gasPrice"])
    tx_fee_eth = Decimal(gas_used) * Decimal(gas_price_wei) / Decimal(10**18)

    return DepositTxInfo(
        tx_hash=tx_hash,
        local_send_time_ns=local_send_time_ns,
        block_number=receipt.blockNumber,
        block_timestamp_ms=block.timestamp * 1000,
        gas_used=gas_used,
        gas_price_wei=gas_price_wei,
        tx_fee_eth=tx_fee_eth,
        amount_raw=amount_raw,
    )

# =====================================================================
# CSV保存
# =====================================================================

def save_result_csv(
    experiment_id: int,
    experiment_start_ns: int,
    deposit_tx: DepositTxInfo,
    hl_result: DepositResult,
    latency_local_ms: float,
):
    RESULT_DIR.mkdir(exist_ok=True)

    experiment_start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(experiment_start_ns / 1e9))
    local_send_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(deposit_tx.local_send_time_ns / 1e9))
    arb_block_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(deposit_tx.block_timestamp_ms / 1000))
    
    hl_ledger_iso = ""
    if hl_result.ledger_time_ms is not None:
        hl_ledger_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(hl_result.ledger_time_ms / 1000))
        hl_ledger_iso += f".{hl_result.ledger_time_ms % 1000:03d}"

    header = [
        "experiment_id",
        "experiment_start_iso",
        "local_send_iso",
        "local_send_time(ns)",
        "amount_usdc",
        "amount_raw(usdc_atomic)",
        "arb_tx_hash",
        "arb_block_number",
        "arb_block_timestamp_iso",
        "arb_block_timestamp(ms)",
        "arb_gas_used(wei)",
        "arb_gas_price(wei)",
        "arb_tx_fee(eth)",
        "hl_ledger_time_iso",
        "hl_ledger_time(ms)",
        "hl_block_number",
        "hl_block_timestamp(ms)",
        "hl_tx_hash",
        "latency(ms)",
        "note"
    ]

    file_exists = RESULT_CSV_PATH.exists() and RESULT_CSV_PATH.stat().st_size > 0

    with RESULT_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            experiment_id,
            experiment_start_iso,
            local_send_iso,
            deposit_tx.local_send_time_ns,
            str(DEPOSIT_AMOUNT_USDC),
            deposit_tx.amount_raw,
            deposit_tx.tx_hash,
            deposit_tx.block_number,
            arb_block_iso,
            deposit_tx.block_timestamp_ms,
            deposit_tx.gas_used,
            deposit_tx.gas_price_wei,
            f"{deposit_tx.tx_fee_eth:.18f}",
            hl_ledger_iso,
            hl_result.ledger_time_ms,
            hl_result.hl_block_number,
            hl_result.hl_block_timestamp_ms,
            hl_result.hl_tx_hash,
            f"{latency_local_ms:.3f}",
            "Validator-Signed"
        ])

# =====================================================================
# Main
# =====================================================================

def main():
    print("===== Arbitrum → Hyperliquid USDC Deposit Latency (Gas Auto-Estimate) =====")

    exp_id = get_next_experiment_id(RESULT_CSV_PATH)
    experiment_start_ns = time.time_ns()
    experiment_start_ms = experiment_start_ns // 1_000_000

    print(f"Current Experiment ID: {exp_id}")
    print(f"HL User Address : {HL_USER_ADDRESS}")
    print(f"Deposit Amount  : {DEPOSIT_AMOUNT_USDC}")
    print()

    # 1. Start WebSocket Listener (Direct & Robust)
    hl_listener = DirectHlListener(
        user_address=HL_USER_ADDRESS,
        amount_usdc=DEPOSIT_AMOUNT_USDC,
        experiment_start_ms=experiment_start_ms
    )
    hl_listener.start()
    
    print("Waiting 2s for WebSocket connection...")
    time.sleep(2)

    # 2. Arbitrum Connection (Direct)
    w3_arb = make_arb_web3()
    
    # 3. Send Deposit
    try:
        deposit_tx = send_usdc_deposit(w3_arb)
    except Exception as e:
        print(f"[CRITICAL] Arbitrum Deposit Failed: {e}")
        os._exit(1)

    print(f"[Deposit] Tx Hash: {deposit_tx.tx_hash}")
    print(f"[Deposit] Block: {deposit_tx.block_number}")
    print()

    print("Waiting for Hyperliquid deposit ledger (timeout: 900s)...")
    result = hl_listener.wait_for_deposit(timeout_sec=900)

    if not result.found:
        print("[ERROR] Timeout: Deposit not detected.")
        os._exit(1)

    # レイテンシ計算 (ms)
    latency_ms = result.ledger_time_ms - (deposit_tx.local_send_time_ns / 1_000_000)

    print()
    print("===== Detected target deposit on Hyperliquid =====")
    print(f" Ledger Time (ms) : {result.ledger_time_ms}")
    print(f" HL Block Number  : {result.hl_block_number}")
    print(f" HL Block Time MS : {result.hl_block_timestamp_ms}")
    print(f" HL Tx Hash       : {result.hl_tx_hash}")
    print(f" Latency (ms)     : {latency_ms:.3f}")
    print("==================================================")

    save_result_csv(
        experiment_id=exp_id,
        experiment_start_ns=experiment_start_ns,
        deposit_tx=deposit_tx,
        hl_result=result,
        latency_local_ms=latency_ms
    )

    print(f"Saved to {RESULT_CSV_PATH}")
    os._exit(0)

if __name__ == "__main__":
    main()