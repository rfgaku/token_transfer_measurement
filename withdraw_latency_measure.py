#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
withdraw_latency_measure_final.py
Hyperliquid → Arbitrum USDC withdraw
【最終解決版：SDK正攻法 + Hash予知】

仕組み:
1. 送金処理は、過去に成功実績のある公式SDKメソッド `withdraw_from_bridge` をそのまま使用。
   -> これにより「送金されない」「署名エラー」といった問題を完全に解消。
2. SDKが内部で使用する「時刻(Nonce)」を一時的に固定化(Monkey Patch)し、
   送信されるデータのハッシュ(Action Hash)をクライアント側で正確に計算・記録する。
3. 監視はDepositで成功した「直接WebSocket」方式で行う。
"""

import os
import threading
import time
import csv
import json
import websocket
import msgpack
import requests
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_utils import keccak

# SDK
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
# timestampをジャックするためにインポート
import hyperliquid.utils.signing as signing_lib
# Infoクラスを無効化するためにインポート
import hyperliquid.info

# =====================================================================
# 【重要】SDKの通信遮断・無効化パッチ
# Exchange初期化時のフリーズを防ぐため、Infoクラスをダミーに差し替える
# =====================================================================
class DummyInfo:
    def __init__(self, *args, **kwargs):
        pass
    def user_state(self, *args, **kwargs):
        return {}

# 本物のInfoクラスをダミーで上書き
hyperliquid.info.Info = DummyInfo
# =====================================================================

# =====================================================================
# 設定
# =====================================================================

load_dotenv()

ARB_RPC_URL = os.getenv("ARB_RPC_URL", "https://arb1.arbitrum.io/rpc")
HL_API_URL = constants.MAINNET_API_URL
HL_EVM_RPC_URL = "https://rpc.hyperliquid.xyz/evm"

try:
    ARB_SENDER_ADDRESS = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    ARB_SENDER_PRIVATE_KEY = os.environ["ARB_SENDER_PRIVATE_KEY"]
except KeyError:
    print("[ERROR] 環境変数が設定されていません。")
    sys.exit(1)

HL_BRIDGE_ADDRESS = Web3.to_checksum_address(
    os.getenv("HL_DEPOSIT_BRIDGE_ADDRESS", "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7")
)
ARB_USDC_ADDRESS = Web3.to_checksum_address(
    os.getenv("ARB_USDC_ADDRESS", "0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
)

WITHDRAW_AMOUNT = 6.0
RESULT_DIR = Path("result")
RESULT_CSV_PATH = RESULT_DIR / "withdraw_latency.csv"

# =====================================================================
# Helper
# =====================================================================

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
    if not csv_path.exists(): return 1
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ids = [int(row["experiment_id"]) for row in reader if row.get("experiment_id")]
            return max(ids) + 1 if ids else 1
    except:
        return 1

def calculate_clock_offset_ms(w3: Web3) -> int:
    print("Synchronizing clock with Arbitrum...")
    try:
        t1 = time.time_ns()
        block = w3.eth.get_block('latest')
        t2 = time.time_ns()
        rtt_ns = t2 - t1
        local_est_ns = t1 + (rtt_ns / 2)
        chain_time_ns = block.timestamp * 1_000_000_000
        offset_ms = int((chain_time_ns - local_est_ns) / 1_000_000)
        print(f"[Clock Sync] Offset: {offset_ms} ms")
        return offset_ms
    except Exception as e:
        print(f"[WARN] Clock sync failed ({e}). Proceeding with 0 offset.")
        return 0

# =====================================================================
# Action Hash Logic (SDK + Timestamp Injection)
# =====================================================================

def execute_withdraw_via_sdk_with_hash(account: Account, amount: float, destination: str):
    """
    SDKの標準メソッド(withdraw_from_bridge)を使いつつ、
    タイムスタンプを固定することでAction Hashを正確に計算する。
    """
    exchange = Exchange(account, HL_API_URL)
    
    # 1. 使用するタイムスタンプをここで決める
    fixed_timestamp = int(time.time() * 1000)
    
    # 2. SDKの時刻取得関数を一時的に書き換え（モンキーパッチ）
    # これによりSDKは必ずこの timestamp を使って署名・送信する
    original_get_timestamp = signing_lib.get_timestamp_ms
    signing_lib.get_timestamp_ms = lambda: fixed_timestamp
    
    try:
        # 3. Action Hashを計算 (SDK内部のロジックを模倣)
        action = {
            "type": "withdraw3",
            "hyperliquidChain": "Mainnet",
            "signatureChainId": "0xa4b1", # Arbitrum ChainID
            "amount": str(amount),
            "time": fixed_timestamp, # 固定した時刻
            "destination": destination
        }
        packed = msgpack.packb(action)
        action_hash = "0x" + keccak(packed).hex()
        
        print(f"[DEBUG] Pre-calculated Action Hash: {action_hash}")
        
        # 4. SDKで実行 (成功実績のあるメソッド)
        # ここで内部的に固定時刻が使われる
        print("Sending request via SDK...")
        result = exchange.withdraw_from_bridge(amount, destination)
        
        return result, action_hash
        
    finally:
        # 5. 後始末: 関数を元に戻す
        signing_lib.get_timestamp_ms = original_get_timestamp

# =====================================================================
# Direct WebSocket Listener
# =====================================================================

@dataclass
class WithdrawEventResult:
    found: bool = False
    ledger_time_ms: int = 0
    tx_hash: str = "N/A"
    raw_event: dict = None

class DirectHlWithdrawListener:
    def __init__(self, user_address: str, experiment_start_ms: int):
        self.user_address = user_address.lower()
        self.experiment_start_ms = experiment_start_ms
        self.result = WithdrawEventResult()
        self.ws = None
        self.running = False 

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            ch = msg.get("channel")
            if ch != "userNonFundingLedgerUpdates": return
            data = msg.get("data", {})
            
            updates = data.get("nonFundingLedgerUpdates") or []
            for upd in updates:
                event_time = upd.get("time", 0)
                if event_time < self.experiment_start_ms: continue

                delta = upd.get("delta", {})
                if delta.get("type") != "withdraw": continue

                try:
                    usdc = float(delta.get("usdc", "0"))
                except: usdc = 0.0
                
                if abs(usdc - WITHDRAW_AMOUNT) < 0.1 or abs(usdc - (WITHDRAW_AMOUNT - 1.0)) < 0.1:
                    h = upd.get("hash", "N/A")
                    print(f"\n[WS-SUCCESS] Withdraw Detected! Hash: {h} Time: {event_time}")
                    
                    self.result.found = True
                    self.result.tx_hash = h
                    self.result.ledger_time_ms = event_time
                    self.result.raw_event = upd
                    ws.close()
                    return

        except Exception:
            pass

    def on_error(self, ws, error):
        pass

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
        while not self.result.found and self.running:
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
            except:
                pass
            
            if not self.result.found and self.running:
                time.sleep(5)

    def start_monitoring(self):
        self.running = True
        wst = threading.Thread(target=self._run_ws)
        wst.daemon = True
        wst.start()

# =====================================================================
# Main Logic
# =====================================================================

@dataclass
class ArbArrivalInfo:
    found: bool = False
    tx_hash: str = "N/A"
    block_number: int = 0
    block_timestamp_ms: int = 0
    block_timestamp_iso: str = ""
    gas_used: int = 0
    gas_price_wei: int = 0
    amount_raw: int = 0

def main():
    print("===== HL -> Arb Withdraw (SDK Safe & Wait) =====")
    
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = get_next_experiment_id(RESULT_CSV_PATH)
    
    account = Account.from_key(ARB_SENDER_PRIVATE_KEY)
    address = account.address
    print(f"User: {address} | ID: {exp_id}")

    hl_w3 = make_hl_web3()
    arb_w3 = make_arb_web3()

    # 1. Clock Sync
    clock_offset_ms = calculate_clock_offset_ms(arb_w3)

    # 2. Execute Withdraw
    print(f"Executing Withdraw ({WITHDRAW_AMOUNT} USDC)...")
    
    experiment_start_ns = time.time_ns()
    experiment_start_ms = int(experiment_start_ns / 1_000_000)
    
    local_broadcast_time_ns = time.time_ns()
    hl_action_hash = "N/A"

    try:
        # SDKを使って安全に送信 + ハッシュ予知
        resp, hl_action_hash = execute_withdraw_via_sdk_with_hash(account, WITHDRAW_AMOUNT, address)
        
        # レスポンスチェック
        if resp.get("status") == "ok":
            print(f"[SUCCESS] API Request Sent. Action Hash: {hl_action_hash}")
        else:
            print(f"[API ERROR] {resp}")
            os._exit(1)
            
    except Exception as e:
        print(f"[CRITICAL] Execution Failed: {e}")
        os._exit(1)

    # HL Block Info
    hl_block_num = 0
    hl_block_ts_ms = 0
    try:
        block = hl_w3.eth.get_block('latest')
        hl_block_num = block.number
        hl_block_ts_ms = block.timestamp * 1000
    except: pass

    # 3. Optimized Wait
    print("\nWaiting 180 seconds before monitoring (Log Silence)...")
    for _ in range(18):
        time.sleep(10)
        print(".", end="", flush=True)
    print("\nStarting Monitoring...")

    # 4. Start Monitoring
    listener = DirectHlWithdrawListener(address, experiment_start_ms)
    listener.start_monitoring()
    
    arb_arrival = ArbArrivalInfo()
    usdc_contract = arb_w3.eth.contract(address=ARB_USDC_ADDRESS, abi=[
        {"anonymous":False,"inputs":[{"indexed":True,"name":"from","type":"address"},{"indexed":True,"name":"to","type":"address"},{"indexed":False,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"}
    ])
    
    start_block = arb_w3.eth.block_number
    loop_start_time = time.time()
    
    while time.time() - loop_start_time < 1200:
        # A. Arb Check
        if not arb_arrival.found:
            try:
                current_block = arb_w3.eth.block_number
                if current_block > start_block:
                    logs = usdc_contract.events.Transfer.get_logs(
                        fromBlock=start_block, toBlock=current_block,
                        argument_filters={'from': HL_BRIDGE_ADDRESS, 'to': address}
                    )
                    if logs:
                        log = logs[-1]
                        rec = arb_w3.eth.get_transaction_receipt(log['transactionHash'])
                        blk = arb_w3.eth.get_block(rec.blockNumber)
                        
                        arb_arrival.found = True
                        arb_arrival.tx_hash = log['transactionHash'].hex()
                        arb_arrival.block_number = rec.blockNumber
                        arb_arrival.block_timestamp_ms = blk.timestamp * 1000
                        arb_arrival.block_timestamp_iso = datetime.fromtimestamp(blk.timestamp, timezone.utc).isoformat()
                        arb_arrival.gas_used = rec.gasUsed
                        arb_arrival.gas_price_wei = rec.effectiveGasPrice
                        arb_arrival.amount_raw = log['args']['value']
                        print(f"\n[ARB-SUCCESS] Arrival Detected! Hash: {arb_arrival.tx_hash}")
                    start_block = current_block
            except Exception:
                pass

        # B. Completion Check
        if arb_arrival.found:
            if not listener.result.found:
                print("Arb arrived. Waiting 30s more for WS...")
                time.sleep(30)
            
            # WSで見つかった場合はそのハッシュも表示（デバッグ用）
            ws_hash = listener.result.tx_hash if listener.result.found else "N/A"
            print(f"\n>>> COMPLETE. HL Hash: {hl_action_hash} | Arb Hash: {arb_arrival.tx_hash} | WS Hash: {ws_hash}")
            break
        
        print(".", end="", flush=True)
        time.sleep(5)

    # 5. Save
    latency_ms = 0
    if arb_arrival.found:
        broadcast_ms = local_broadcast_time_ns / 1_000_000
        corrected_broadcast_ms = broadcast_ms + clock_offset_ms
        latency_ms = arb_arrival.block_timestamp_ms - corrected_broadcast_ms

    local_broadcast_iso = datetime.fromtimestamp(local_broadcast_time_ns / 1e9).isoformat()

    headers = [
        "experiment_id", "local_broadcast_iso", "local_broadcast_time(ns)",
        "amount_usdc", "hl_block_number", "hl_block_timestamp(ms)", "hl_tx_hash",
        "arb_tx_hash", "arb_block_number", "arb_block_timestamp_iso", "arb_block_timestamp(ms)",
        "arb_gas_used(wei)", "arb_gas_price(wei)", "amount_received_raw",
        "clock_offset(ms)", "latency(ms)", "note"
    ]

    write_header = not (RESULT_CSV_PATH.exists() and RESULT_CSV_PATH.stat().st_size > 0)
    
    try:
        with open(RESULT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header: writer.writerow(headers)
            writer.writerow([
                exp_id, local_broadcast_iso, local_broadcast_time_ns,
                WITHDRAW_AMOUNT, hl_block_num, hl_block_ts_ms, 
                hl_action_hash,
                arb_arrival.tx_hash, arb_arrival.block_number,
                arb_arrival.block_timestamp_iso, arb_arrival.block_timestamp_ms,
                arb_arrival.gas_used, arb_arrival.gas_price_wei, arb_arrival.amount_raw,
                clock_offset_ms, 
                f"{latency_ms:.3f}" if arb_arrival.found else "Timeout",
                "Final-SDK-Safe"
            ])
        print(f"Done. Saved to {RESULT_CSV_PATH}")
    except Exception as e:
        print(f"Save failed: {e}")

    print("Exiting process.")
    os._exit(0)

if __name__ == "__main__":
    main()