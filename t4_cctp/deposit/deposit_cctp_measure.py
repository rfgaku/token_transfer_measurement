#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deposit_cctp_measure.py
T4: Arbitrum → HyperCore USDC Deposit via CCTP V2 Fast Transfer のレイテンシ測定。

■ 方式選定（Phase A-2 / 一次ソース確認済み）
- Arbitrum 上の TokenMessengerV2.depositForBurnWithHook を「直接」呼ぶ方式を採用。
  理由: 単一ユーザーの HyperCore 宛 deposit では最もシンプル。CctpExtension
  (batchDepositForBurnWithAuth) は ERC-3009 receiveWithAuthorization 署名・バッチ前提で
  過剰。CctpExtension も内部的には depositForBurnWithHook へ委譲するだけ（hyperevm-circle-
  contracts で確認）。
- mintRecipient = destinationCaller = CctpForwarder(HyperEVM)。hookData に HyperCore 受取人を
  エンコード。Circle Forwarding Service が destination 側で CctpForwarder.mintAndForward を
  自動実行し、CoreDepositWallet 経由で HyperCore に credit する。
- Deposit は Fast：minFinalityThreshold=1000 → finalityThresholdExecuted=1000 を期待。
  Arbitrum→HyperCore Fast は CCTP 手数料 0（maxFee=0 / feeExecuted=0 を期待）。

■ 4点タイミング
  t1   = Arbitrum burn block timestamp (+block_number)
  t2   = Iris attestation complete（poll 検知時刻, local）
  t2.5 = HyperEVM mint block timestamp (+block_number)  ※USDC Transfer(0x0→Forwarder) を観測
  t3   = HyperCore credit ledger time（WS userNonFundingLedgerUpdates, T1 検知器を流用）

■ 安全
  既定はドライラン（ブロードキャストしない）。実送信は CLI に --broadcast を明示した時のみ。
  出力は config.RESULT_DIR/test/deposit_cctp_test.csv（追記）。本番用 CSV とは分離。
"""

import os
import sys
import csv
import json
import time
import argparse
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

# --- 修正0: リポジトリルートを sys.path に追加（PYTHONPATH 無し・任意ディレクトリから実行可能に） ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- T4 設定 ---
from t4_cctp.deposit import config as cfg

# --- T1 の HyperCore credit 検知器を「流用」（リポジトリ直下の既存モジュール） ---
sys.path.insert(0, str(cfg.REPO_ROOT))
try:
    from deposit_latency_measure import DirectHlListener, DepositResult  # noqa: E402
    _T1_LISTENER_OK = True
except Exception as _e:  # pragma: no cover - 流用可否は実行時に報告
    DirectHlListener = None  # type: ignore
    DepositResult = None  # type: ignore
    _T1_LISTENER_OK = False
    _T1_IMPORT_ERR = _e

load_dotenv()

# =====================================================================
# 環境変数（鍵は .env のみ。ログには出さない）
# =====================================================================
try:
    ARB_SENDER_ADDRESS = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    ARB_SENDER_PRIVATE_KEY = os.environ["ARB_SENDER_PRIVATE_KEY"]
except KeyError as e:
    raise KeyError(f"環境変数 {e} が読み込めませんでした。.env を確認してください。")

HL_USER_ADDRESS = Web3.to_checksum_address(os.getenv("HL_USER_ADDRESS", ARB_SENDER_ADDRESS))
DEPOSIT_AMOUNT_USDC = Decimal(str(cfg.DEPOSIT_AMOUNT_USDC))

# 既定は test CSV。本番は --prod で cfg.PROD_CSV を使う（run_broadcast 内で解決）。
RESULT_CSV_PATH = cfg.TEST_CSV

# =====================================================================
# ABI（最小限。一次ソースの署名に一致）
# =====================================================================
ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "value", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "decimals", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint8"}]},
]

# TokenMessengerV2.depositForBurnWithHook（evm-cctp-contracts src/v2/TokenMessengerV2.sol で確認）
TOKEN_MESSENGER_V2_ABI = [
    {"name": "depositForBurnWithHook", "type": "function", "stateMutability": "nonpayable",
     "inputs": [
         {"name": "amount", "type": "uint256"},
         {"name": "destinationDomain", "type": "uint32"},
         {"name": "mintRecipient", "type": "bytes32"},
         {"name": "burnToken", "type": "address"},
         {"name": "destinationCaller", "type": "bytes32"},
         {"name": "maxFee", "type": "uint256"},
         {"name": "minFinalityThreshold", "type": "uint32"},
         {"name": "hookData", "type": "bytes"},
     ], "outputs": []},
]

# Transfer(address,address,uint256) topic
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# =====================================================================
# Web3 helper
# =====================================================================
def make_arb_web3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(cfg.ARB_RPC_URL))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3

def make_hl_web3() -> Web3:
    provider = Web3.HTTPProvider(
        cfg.HL_EVM_RPC_URL,
        request_kwargs={"timeout": 10, "headers": {"User-Agent": "Mozilla/5.0"}},
    )
    w3 = Web3(provider)
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3

def address_to_bytes32(addr: str) -> bytes:
    """20-byte EVM アドレス → 左ゼロ詰め 32 byte。"""
    return bytes(12) + Web3.to_bytes(hexstr=Web3.to_checksum_address(addr))

def build_hook_data(recipient: str, destination_id: int) -> bytes:
    """CctpForwarder hookData（CctpForwarderHookData.sol 準拠・生バイト連結=encodePacked）。

    レイアウト（big-endian, 計56byte）:
      [0:24]  magic   = "cctp-forward" + 右0詰め(24byte)
      [24:28] version = uint32(0)
      [28:32] dataLen = uint32(24)   (recipient20 + destId4)
      [32:52] recipient = HyperCore 受取人 EVM アドレス(20byte)
      [52:56] destinationId = uint32 (0=perp, 0xFFFFFFFF=spot)
    """
    magic = cfg.HOOK_MAGIC.ljust(24, b"\x00")
    version = (0).to_bytes(4, "big")
    data_len = (24).to_bytes(4, "big")
    recip = Web3.to_bytes(hexstr=Web3.to_checksum_address(recipient))  # 20 byte
    dest = (destination_id & 0xFFFFFFFF).to_bytes(4, "big")
    hook = magic + version + data_len + recip + dest
    assert len(hook) == 56, f"hookData length {len(hook)} != 56"
    return hook

def get_next_experiment_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            ids = [int(r["experiment_id"]) for r in csv.DictReader(f) if r.get("experiment_id")]
            return max(ids) + 1 if ids else 1
    except Exception:
        return 1

# =====================================================================
# Iris API（attestation 取得）
# =====================================================================
@dataclass
class IrisResult:
    complete_local_ms: int = 0
    message_hex: Optional[str] = None
    attestation_hex: Optional[str] = None
    event_nonce: Optional[str] = None
    message_hash: Optional[str] = None
    finality_threshold_executed: Optional[int] = None
    fee_executed: Optional[int] = None
    forward_state: Optional[str] = None  # complete時点では未確定なことがある（forwarding は後で起きる）
    raw: dict = field(default_factory=dict)


def fetch_forward_state(burn_tx: str) -> Optional[str]:
    """Iris から forwardState(COMPLETE/FAILED/...) のみ best-effort で取得する。
    forwarding は attestation complete の後に起きるため、credit 検知後に呼ぶのが確実。"""
    try:
        url = (cfg.IRIS_API_HOST + cfg.IRIS_MESSAGES_PATH.format(source_domain=cfg.ARB_DOMAIN_ID)
               + f"?transactionHash={burn_tx}")
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            msgs = resp.json().get("messages") or []
            if msgs:
                return msgs[0].get("forwardState")
    except Exception as e:
        print(f"[iris] forwardState 取得失敗: {e}")
    return None

def poll_iris_attestation(source_domain: int, burn_tx: str, timeout_sec: int = 300) -> Optional[IrisResult]:
    """messages[0].status == 'complete' まで poll（2〜4 req/s）。"""
    url = (cfg.IRIS_API_HOST + cfg.IRIS_MESSAGES_PATH.format(source_domain=source_domain)
           + f"?transactionHash={burn_tx}")
    start = time.time()
    deadline = start + timeout_sec
    last_hb = start
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                msgs = data.get("messages") or []
                if msgs:
                    m = msgs[0]
                    status = m.get("status")
                    if status == "complete":
                        msg_hex = m.get("message")
                        dec = m.get("decodedMessage") or {}
                        body = dec.get("decodedMessageBody") or {}
                        res = IrisResult(
                            complete_local_ms=int(time.time() * 1000),
                            message_hex=msg_hex,
                            attestation_hex=m.get("attestation"),
                            event_nonce=str(m.get("eventNonce") or dec.get("nonce") or ""),
                            message_hash=(Web3.keccak(hexstr=msg_hex).hex() if msg_hex else None),
                            finality_threshold_executed=_to_int(
                                dec.get("finalityThresholdExecuted")
                                or m.get("finalityThresholdExecuted")),
                            fee_executed=_to_int(body.get("feeExecuted") or m.get("feeExecuted")),
                            raw=m,
                        )
                        return res
        except Exception as e:
            print(f"[Iris] poll error: {e}")
        now = time.time()
        if now - last_hb >= 10:  # 修正4: 10秒ごとにハートビート
            print(f"[heartbeat] still waiting iris ({int(now - start)}秒経過)")
            last_hb = now
        time.sleep(cfg.IRIS_POLL_INTERVAL_SEC)
    print(f"[iris] TIMEOUT after {timeout_sec}s")
    return None

def _to_int(v):
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None

# =====================================================================
# Iris fee API（Fast の maxFee を一次ソースから取得。憶測で埋めない）
# =====================================================================
def fetch_fast_fee(amount_raw: int, buffer_pct: float = 0.10) -> dict:
    """GET /v2/burn/USDC/fees/{src}/{dst} を叩き、Fast(finalityThreshold=1000) の
    minimumFee(bps) から送金額に対する maxFee(atomic) を算出する。
    返り値の max_fee_atomic を depositForBurnWithHook の maxFee に使う。
    bps が取得できない/0 の場合 min_fee_atomic / max_fee_atomic は 0 or None になり、
    呼び出し側の安全弁で送信中止できるようにする。"""
    import math
    url = cfg.IRIS_API_HOST + cfg.IRIS_FEE_PATH.format(
        source_domain=cfg.ARB_DOMAIN_ID, dest_domain=cfg.HYPEREVM_DOMAIN_ID)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    fast = standard = None
    for e in (data or []):
        if e.get("finalityThreshold") == cfg.FINALITY_THRESHOLD_FAST:
            fast = e
        elif e.get("finalityThreshold") == cfg.FINALITY_THRESHOLD_STANDARD:
            standard = e
    bps = fast.get("minimumFee") if fast else None
    min_fee_atomic = None
    max_fee_atomic = None
    if bps is not None:
        min_fee_atomic = int(math.ceil(amount_raw * float(bps) / 10000.0))
        max_fee_atomic = int(math.ceil(min_fee_atomic * (1.0 + buffer_pct)))
    return {
        "url": url, "raw": data, "fast_entry": fast, "standard_entry": standard,
        "fast_bps": bps, "min_fee_atomic": min_fee_atomic,
        "max_fee_atomic": max_fee_atomic, "buffer_pct": buffer_pct,
    }

# Forwarding 手数料の保守的フォールバック（Circle forwarding-service doc の $0.20 固定）。
# forward=true が forwardFee を返せばそちらを優先採用する。
FORWARD_FEE_FALLBACK_ATOMIC = 200_000  # $0.20 (USDC 6 decimals)

def compute_deposit_max_fee(amount_raw: int, buffer_pct: float = 0.05,
                            forward_tier: str = "med") -> dict:
    """HyperCore 自動配送を成立させる maxFee を「CCTP基本手数料 + Forwarding手数料」で積み上げる。

      maxFee = ceil( (base_fast_fee + forward_fee) * (1 + buffer_pct) )

    - base_fast_fee = ceil(amount_raw * Fast minimumFee(bps) / 10000)   ← /fees の Fast bps
    - forward_fee   = /fees?forward=true の Fast entry forwardFee[tier]  ← 実値優先
                      （無ければ FORWARD_FEE_FALLBACK_ATOMIC=200000）
    生レスポンス(a/b)も返し、check_fee.py で全文表示できるようにする。"""
    import math
    base_url = cfg.IRIS_API_HOST + cfg.IRIS_FEE_PATH.format(
        source_domain=cfg.ARB_DOMAIN_ID, dest_domain=cfg.HYPEREVM_DOMAIN_ID)
    url_a = base_url
    url_b = base_url + "?forward=true"
    ra = requests.get(url_a, timeout=10); ra.raise_for_status(); data_a = ra.json()
    rb = requests.get(url_b, timeout=10); rb.raise_for_status(); data_b = rb.json()

    def _fast(data):
        for e in (data or []):
            if e.get("finalityThreshold") == cfg.FINALITY_THRESHOLD_FAST:
                return e
        return None
    fast_a = _fast(data_a)
    fast_b = _fast(data_b)

    bps = fast_a.get("minimumFee") if fast_a else None
    base_fee = int(math.ceil(amount_raw * float(bps) / 10000.0)) if bps is not None else None

    fwd_obj = (fast_b or {}).get("forwardFee") if fast_b else None
    forward_fee = None
    forward_fee_med = None  # 安全弁の下限比較に使う
    forward_source = None
    if isinstance(fwd_obj, dict):
        forward_fee_med = int(fwd_obj["med"]) if fwd_obj.get("med") is not None else None
    if isinstance(fwd_obj, dict) and fwd_obj.get(forward_tier) is not None:
        forward_fee = int(fwd_obj[forward_tier])
        forward_source = f"API forwardFee[{forward_tier}]"
    elif isinstance(fwd_obj, (int, float)):
        forward_fee = int(fwd_obj)
        forward_fee_med = int(fwd_obj)
        forward_source = "API forwardFee(scalar)"
    else:
        forward_fee = FORWARD_FEE_FALLBACK_ATOMIC
        forward_fee_med = FORWARD_FEE_FALLBACK_ATOMIC
        forward_source = "fallback $0.20(200000)"

    max_fee = None
    if base_fee is not None and forward_fee is not None:
        subtotal = base_fee + forward_fee
        max_fee = int(math.ceil(subtotal * (1.0 + buffer_pct)))

    return {
        "url_a": url_a, "url_b": url_b, "raw_a": data_a, "raw_b": data_b,
        "fast_a": fast_a, "fast_b": fast_b,
        "fast_bps": bps, "base_fee_atomic": base_fee,
        "forward_fee_obj": fwd_obj, "forward_fee_atomic": forward_fee,
        "forward_fee_med": forward_fee_med,
        "forward_tier": forward_tier, "forward_source": forward_source,
        "buffer_pct": buffer_pct, "max_fee_atomic": max_fee,
    }

# =====================================================================
# HyperEVM mint 観測（t2.5）: USDC Transfer(0x0 → Forwarder, value=amount)
# =====================================================================
@dataclass
class HevmMintResult:
    found: bool = False
    block_number: int = 0
    block_ts_ms: int = 0
    tx_hash: str = "N/A"

def watch_hevm_mint(hl_w3: Web3, amount_raw: int, start_block: int, timeout_sec: int = 180,
                    accept_values: Optional[set] = None) -> HevmMintResult:
    """Forwarder への mint(Transfer from 0x0) を get_logs で監視。
    Fast では feeExecuted が控除され mint 額 = amount - feeExecuted になるため、
    accept_values（許容する atomic 値の集合）で net/gross 双方を受け付ける。
    None のときは gross(amount_raw) のみ。"""
    if accept_values is None:
        accept_values = {amount_raw}
    forwarder = Web3.to_checksum_address(cfg.CCTP_FORWARDER_HEVM)
    usdc = Web3.to_checksum_address(cfg.HYPEREVM_USDC_ADDRESS)
    to_topic = "0x" + forwarder[2:].lower().rjust(64, "0")
    from_topic = "0x" + ZERO_ADDRESS[2:].rjust(64, "0")
    start = time.time()
    deadline = start + timeout_sec
    last_hb = start
    frm = start_block
    while time.time() < deadline:
        try:
            latest = hl_w3.eth.block_number
            if latest >= frm:
                logs = hl_w3.eth.get_logs({
                    "fromBlock": frm, "toBlock": latest, "address": usdc,
                    "topics": [TRANSFER_TOPIC, from_topic, to_topic],
                })
                for lg in logs:
                    val = int(lg["data"], 16) if isinstance(lg["data"], str) else int.from_bytes(lg["data"], "big")
                    if val in accept_values:
                        blk = hl_w3.eth.get_block(lg["blockNumber"])
                        return HevmMintResult(True, lg["blockNumber"], blk.timestamp * 1000,
                                              lg["transactionHash"].hex())
                frm = latest + 1
        except Exception as e:
            print(f"[HEVM] mint watch error: {e}")
        now = time.time()
        if now - last_hb >= 10:  # 修正4: 10秒ごとにハートビート
            print(f"[heartbeat] still waiting mint ({int(now - start)}秒経過)")
            last_hb = now
        time.sleep(2)
    print(f"[hevm] mint TIMEOUT after {timeout_sec}s")
    return HevmMintResult(False)

# =====================================================================
# Arbitrum: approve & depositForBurnWithHook
# =====================================================================
@dataclass
class BurnTxInfo:
    tx_hash: str = ""
    local_send_ns: int = 0
    block_number: int = 0
    block_ts_ms: int = 0
    gas_used: int = 0
    gas_price_wei: int = 0
    tx_fee_eth: Decimal = Decimal(0)

def _build_burn_params(amount_raw: int):
    mint_recipient = address_to_bytes32(cfg.CCTP_FORWARDER_HEVM)
    dest_caller = address_to_bytes32(cfg.CCTP_FORWARDER_HEVM)
    hook = build_hook_data(HL_USER_ADDRESS, cfg.DEPOSIT_HC_DESTINATION_ID)
    return mint_recipient, dest_caller, hook

def assert_recipient_is_self():
    """【送信直前チェック】hookData[32:52] の recipient(20B) が HL_USER_ADDRESS と
    完全一致することを検証。一致しなければ送信中止（例外）。"""
    hook = build_hook_data(HL_USER_ADDRESS, cfg.DEPOSIT_HC_DESTINATION_ID)
    embedded = Web3.to_checksum_address("0x" + hook[32:52].hex())
    expected = Web3.to_checksum_address(HL_USER_ADDRESS)
    dest_id = int.from_bytes(hook[52:56], "big")
    if embedded != expected:
        raise RuntimeError(
            f"[ABORT] hookData recipient {embedded} != HL_USER_ADDRESS {expected}. 送信中止。")
    if dest_id != cfg.HC_DEST_PERP:
        raise RuntimeError(
            f"[ABORT] destinationId {dest_id} != perp(0)。送信中止。")
    print(f"[precheck] OK: hookData recipient = {embedded} (= HL_USER_ADDRESS), destId = perp(0)")

def estimate_and_summarize(w3: Web3, amount_raw: int) -> dict:
    """ドライラン用：残高・allowance・ガス見積り・送信予定要約を返す（送信しない）。"""
    sender = ARB_SENDER_ADDRESS
    usdc = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), abi=ERC20_ABI)
    tm = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2),
                         abi=TOKEN_MESSENGER_V2_ABI)

    usdc_bal = usdc.functions.balanceOf(sender).call()
    eth_bal = w3.eth.get_balance(sender)
    allowance = usdc.functions.allowance(sender, Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2)).call()

    mint_recipient, dest_caller, hook = _build_burn_params(amount_raw)
    gas_price = int(w3.eth.gas_price * 1.5)

    # approve ガス見積り（allowance 不足時に必要）
    approve_gas = None
    try:
        approve_gas = usdc.functions.approve(
            Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2), amount_raw
        ).estimate_gas({"from": sender})
    except Exception as e:
        approve_gas = f"estimate失敗: {e}"

    # burn ガス見積り（allowance >= amount のときのみ成功する）
    burn_gas = None
    if allowance >= amount_raw:
        try:
            burn_gas = tm.functions.depositForBurnWithHook(
                amount_raw, cfg.HYPEREVM_DOMAIN_ID, mint_recipient,
                Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), dest_caller,
                cfg.DEPOSIT_MAX_FEE, cfg.DEPOSIT_MIN_FINALITY_THRESHOLD, hook,
            ).estimate_gas({"from": sender})
        except Exception as e:
            burn_gas = f"estimate失敗: {e}"
    else:
        burn_gas = "（allowance 不足のため承認後に見積り可能）"

    return {
        "usdc_balance_raw": usdc_bal,
        "usdc_balance": Decimal(usdc_bal) / Decimal(10**6),
        "eth_balance": Decimal(eth_bal) / Decimal(10**18),
        "allowance_raw": allowance,
        "gas_price_wei": gas_price,
        "approve_gas": approve_gas,
        "burn_gas": burn_gas,
        "mint_recipient_b32": "0x" + mint_recipient.hex(),
        "dest_caller_b32": "0x" + dest_caller.hex(),
        "hook_hex": "0x" + hook.hex(),
    }

def send_approve(w3: Web3, amount_raw: int):
    sender = ARB_SENDER_ADDRESS
    usdc = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), abi=ERC20_ABI)
    nonce = w3.eth.get_transaction_count(sender)
    gas_price = int(w3.eth.gas_price * 1.5)
    fn = usdc.functions.approve(Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2), amount_raw)
    gas = int(fn.estimate_gas({"from": sender}) * 1.2)
    tx = fn.build_transaction({"from": sender, "nonce": nonce, "gasPrice": gas_price, "gas": gas, "value": 0})
    signed = w3.eth.account.sign_transaction(tx, private_key=ARB_SENDER_PRIVATE_KEY)
    h = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    print(f"[Arbitrum] approve sent: {h}")
    w3.eth.wait_for_transaction_receipt(h)
    return h

def send_burn(w3: Web3, amount_raw: int, max_fee: Optional[int] = None) -> BurnTxInfo:
    """max_fee=None のときは後方互換で cfg.DEPOSIT_MAX_FEE(=0) を使う。
    Fast 送信時は呼び出し側が fee API 由来の正の値を渡すこと。"""
    if max_fee is None:
        max_fee = cfg.DEPOSIT_MAX_FEE
    sender = ARB_SENDER_ADDRESS
    tm = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2),
                         abi=TOKEN_MESSENGER_V2_ABI)
    mint_recipient, dest_caller, hook = _build_burn_params(amount_raw)
    nonce = w3.eth.get_transaction_count(sender)
    gas_price = int(w3.eth.gas_price * 1.5)
    fn = tm.functions.depositForBurnWithHook(
        amount_raw, cfg.HYPEREVM_DOMAIN_ID, mint_recipient,
        Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), dest_caller,
        max_fee, cfg.DEPOSIT_MIN_FINALITY_THRESHOLD, hook,
    )
    gas = int(fn.estimate_gas({"from": sender}) * 1.2)
    tx = fn.build_transaction({"from": sender, "nonce": nonce, "gasPrice": gas_price, "gas": gas, "value": 0})
    local_send_ns = time.time_ns()
    signed = w3.eth.account.sign_transaction(tx, private_key=ARB_SENDER_PRIVATE_KEY)
    h = w3.eth.send_raw_transaction(signed.rawTransaction).hex()
    print(f"[Arbitrum] depositForBurnWithHook sent: {h}. Waiting receipt...")
    receipt = w3.eth.wait_for_transaction_receipt(h)
    blk = w3.eth.get_block(receipt.blockNumber)
    gas_used = receipt.gasUsed
    gas_price_wei = getattr(receipt, "effectiveGasPrice", gas_price)
    return BurnTxInfo(
        tx_hash=h, local_send_ns=local_send_ns, block_number=receipt.blockNumber,
        block_ts_ms=blk.timestamp * 1000, gas_used=gas_used, gas_price_wei=gas_price_wei,
        tx_fee_eth=Decimal(gas_used) * Decimal(gas_price_wei) / Decimal(10**18),
    )

# =====================================================================
# CSV 出力（スキーマは Phase A-3 指定）
# =====================================================================
# 列順（B 是正）: 識別子 → 時刻/ブロック → 方式 → コスト → クロスチェック → 【右端】内訳+E2E
# 列名には T1(deposit_latency.csv) 同様 () で単位を付ける。
CSV_HEADER = [
    # --- 識別子 ---
    "experiment_id", "direction", "amount_usdc(usdc)", "cctp_nonce", "message_hash",
    # --- 時刻 / ブロック（4点タイミング）---
    "t1_arb_burn_block_ts(ms)", "t1_arb_burn_block_number", "arb_burn_tx_hash",
    "t2_iris_attestation_complete(ms)",  # status==complete を高頻度poll(0.25s)で検知した raw ローカル時刻
    "t2_iris_complete_local(ms)",        # 区間計算に使う t2（= min(raw検知, t2_5) で t2.5 超過をクランプ）
    "t2_5_hevm_mint_block_ts(ms)", "t2_5_hevm_mint_block_number", "hevm_mint_tx_hash",
    "t3_hc_credit_ledger_time(ms)",
    # --- 方式 ---
    "minFinalityThreshold_set", "finalityThresholdExecuted", "forward_state",
    # --- コスト ---
    "maxFee(usdc_atomic)", "feeExecuted(usdc_atomic)", "arb_gas_used(gas)",
    "arb_gas_price(wei)", "arb_tx_fee(eth)",
    # --- クロスチェック（PC時計・RTT・確認数）---
    "t0_local_send(ns)", "rtt_offset(ms)", "arb_confirmations_at_attestation(blocks)",
    # --- 【右端】内訳 → E2E（最重要指標）---
    # 物理量ベースの構造分解（t2=attestation完了を高頻度pollで物理時刻化, ±0.25s精度）:
    #   iris_wait(ms)          = t2_iris_complete_local - t1_arb_burn_block_ts   … burn→attestation完了（CCTP信頼層レイテンシ）
    #   attestation_to_mint(ms)= t2_5_hevm_mint_block_ts - t2_iris_complete_local … attestation→HEVM mint（relay+mint実行）
    #   credit_wait(ms)        = t3_hc_credit_ledger - t2_5_hevm_mint_block_ts    … mint→HyperCore credit
    #   src_inclusion(ms)      = t2_5 - t1（burn→mint合算・参考保持）
    #   検算: iris_wait + attestation_to_mint + credit_wait == E2E_dep_onchain（t2が打ち消し telescoping で厳密一致）
    "iris_wait(ms)", "attestation_to_mint(ms)", "credit_wait(ms)", "src_inclusion(ms)",
    "E2E_dep_onchain(ms)",    # = t3_hc_credit_ledger_time - t1_arb_burn_block_ts（チェーン固有TS起点）
    "E2E_dep_wallclock(ms)",  # = t3_hc_credit_ledger_time - t0_local_send/1e6（PC送信時刻起点・T1 latency(ms)と同義）
]

def save_csv(row: dict, csv_path: Path = None):
    csv_path = csv_path or RESULT_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)

# =====================================================================
# ドライラン（A-4）
# =====================================================================
def run_dry_run():
    amount_raw = int(DEPOSIT_AMOUNT_USDC * (10**6))
    print("=" * 64)
    print(" T4 Deposit (Arbitrum → HyperCore, CCTP V2 Fast) — DRY RUN")
    print(" ※ ブロードキャストしません（--broadcast 未指定）")
    print("=" * 64)
    print(f" amount         : {DEPOSIT_AMOUNT_USDC} USDC ({amount_raw} atomic)")
    print(f" HL_USER (recip): {HL_USER_ADDRESS}")
    print(f" T1 検知器流用  : {'OK (import 成功)' if _T1_LISTENER_OK else 'NG: '+str(_T1_IMPORT_ERR)}")
    print()

    # Arbitrum 接続・残高・見積り
    print("[1/3] Arbitrum 接続・残高・ガス見積り ...")
    w3 = make_arb_web3()
    print(f"   chainId={w3.eth.chain_id}  connected={w3.is_connected()}")
    s = estimate_and_summarize(w3, amount_raw)
    print(f"   USDC 残高      : {s['usdc_balance']} USDC")
    print(f"   ETH(gas) 残高  : {s['eth_balance']} ETH")
    print(f"   現 allowance   : {Decimal(s['allowance_raw'])/Decimal(10**6)} USDC "
          f"({'十分' if s['allowance_raw'] >= amount_raw else '不足→approve必要'})")
    print(f"   gasPrice(x1.5) : {s['gas_price_wei']} wei")
    print(f"   approve gas    : {s['approve_gas']}")
    print(f"   burn gas       : {s['burn_gas']}")
    print()

    # HyperEVM 接続
    print("[2/3] HyperEVM 接続性 ...")
    try:
        hl = make_hl_web3()
        print(f"   connected={hl.is_connected()}  block={hl.eth.block_number}")
    except Exception as e:
        print(f"   [WARN] HyperEVM 接続失敗: {e}")
    print()

    # HyperCore WS 接続性（短時間テスト）
    print("[3/3] HyperCore WS 接続性（5s テスト） ...")
    print(f"   WS 接続テスト : {'OK' if test_hc_ws(timeout=5) else 'NG/timeout'}")
    print()

    # 送信予定トランザクション要約
    print("-" * 64)
    print(" 送信予定トランザクション要約（Phase B で送る内容）")
    print("-" * 64)
    print(f"   contract        : TokenMessengerV2  {cfg.ARB_TOKEN_MESSENGER_V2}")
    print(f"   function        : depositForBurnWithHook")
    print(f"   amount          : {amount_raw} (= {DEPOSIT_AMOUNT_USDC} USDC, 6 decimals)")
    print(f"   destinationDomain: {cfg.HYPEREVM_DOMAIN_ID} (HyperEVM)")
    print(f"   burnToken       : {cfg.ARB_USDC_ADDRESS} (USDC)")
    print(f"   mintRecipient   : {s['mint_recipient_b32']}")
    print(f"                     (= CctpForwarder {cfg.CCTP_FORWARDER_HEVM})")
    print(f"   destinationCaller: {s['dest_caller_b32']}")
    print(f"                     (= CctpForwarder, Forwarding Service 限定)")
    print(f"   maxFee          : {cfg.DEPOSIT_MAX_FEE} (Fast Arb→HyperCore は 0)")
    print(f"   minFinalityThr. : {cfg.DEPOSIT_MIN_FINALITY_THRESHOLD} (≤1000 → Fast)")
    print(f"   hookData        : {s['hook_hex']}")
    print(f"                     (magic 'cctp-forward' + ver0 + len24 + recip {HL_USER_ADDRESS} + destId {cfg.DEPOSIT_HC_DESTINATION_ID})")
    print(f"   CoreDepositWallet: {cfg.CORE_DEPOSIT_WALLET} (forwarder の転送先)")
    print("-" * 64)
    print(" DRY RUN 完了。実送信は行っていません。")
    print("=" * 64)

def test_hc_ws(timeout: int = 5) -> bool:
    """HyperCore WS に接続して subscribe 応答が来るか短時間テスト。"""
    import websocket
    ok = {"v": False}
    ev = threading.Event()

    def on_open(ws):
        ws.send(json.dumps({"method": "subscribe", "subscription": {
            "type": "userNonFundingLedgerUpdates", "user": HL_USER_ADDRESS.lower()}}))

    def on_message(ws, message):
        ok["v"] = True
        ev.set()
        ws.close()

    def on_error(ws, error):
        ev.set()

    def run():
        try:
            ws = websocket.WebSocketApp(cfg.HL_WS_URL, on_open=on_open,
                                        on_message=on_message, on_error=on_error)
            ws.run_forever()
        except Exception:
            ev.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    ev.wait(timeout=timeout)
    return ok["v"]

# =====================================================================
# 修正1: HyperCore WS 生ログ取得（独立スレッド・全件記録）
# =====================================================================
class HcRawWsLogger:
    """DirectHlListener とは別接続で wss://api.hyperliquid.xyz/ws に繋ぎ、
    HL_USER_ADDRESS の userNonFundingLedgerUpdates を subscribe。受信した全メッセージを
    ローカル受信時刻(ms)付きで jsonl に1行ずつ append・flush する。matcher で絞らず全件記録。
    CCTP credit が T1 と同じ delta.type=='deposit'・同額で来るか未知なため、空振りでも
    実構造を残す目的。burn 送信前に start、credit 確定 or タイムアウトで stop。"""

    def __init__(self, user_address: str, out_path: Path):
        self.user_address = user_address.lower()
        self.out_path = out_path
        self.ws = None
        self._stop = threading.Event()
        self._fh = None
        self.msg_count = 0

    def on_open(self, ws):
        ws.send(json.dumps({"method": "subscribe", "subscription": {
            "type": "userNonFundingLedgerUpdates", "user": self.user_address}}))
        print(f"[hc-raw] WS connected & subscribed (user={self.user_address})")

    def on_message(self, ws, message):
        recv_ms = int(time.time() * 1000)
        try:
            parsed = json.loads(message)
        except Exception:
            parsed = None
        rec = {"recv_local_ms": recv_ms, "raw": (message if parsed is None else parsed)}
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            self.msg_count += 1
        except Exception as e:
            print(f"[hc-raw] write error: {e}")

    def on_error(self, ws, error):
        print(f"[hc-raw] WS error: {error}")

    def _run(self):
        import websocket
        while not self._stop.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    cfg.HL_WS_URL, on_open=self.on_open,
                    on_message=self.on_message, on_error=self.on_error)
                self.ws.run_forever()
            except Exception as e:
                print(f"[hc-raw] setup error: {e}")
            if not self._stop.is_set():
                time.sleep(3)

    def start(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "a", encoding="utf-8")
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        print(f"[hc-raw] logger started -> {self.out_path}")

    def stop(self):
        self._stop.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

# =====================================================================
# CCTP credit 検知器（昨日判明: credit は delta.type=="send" / from=CoreDepositWallet / spot）
# =====================================================================
@dataclass
class CreditResult:
    found: bool = False
    ledger_time_ms: Optional[int] = None
    ledger_usdc: Optional[Decimal] = None
    raw_event: Optional[dict] = None


class HcCctpDepositListener:
    """CCTP 入金の HyperCore credit を検知する。ネイティブブリッジの delta.type=='deposit' と異なり、
    CCTP は CoreDepositWallet からの delta.type=='send'（token=USDC, spot 着金）で届く（昨日実証）。
    マッチ条件: type=='send' ∧ delta.user==CoreDepositWallet ∧ token=='USDC'
               ∧ time>=experiment_start_ms ∧ amount>=min_amount。"""

    def __init__(self, user_address: str, experiment_start_ms: int,
                 core_deposit_wallet: str = cfg.CORE_DEPOSIT_WALLET,
                 token: str = "USDC", min_amount: Decimal = Decimal("0")):
        self.user_address = user_address.lower()
        self.core_wallet = Web3.to_checksum_address(core_deposit_wallet).lower()
        self.token = token
        self.experiment_start_ms = experiment_start_ms
        self.min_amount = min_amount
        self.result = CreditResult()
        self._event = threading.Event()
        self.ws = None
        self._stop = False

    def on_open(self, ws):
        ws.send(json.dumps({"method": "subscribe", "subscription": {
            "type": "userNonFundingLedgerUpdates", "user": self.user_address}}))
        print("[hc] credit listener connected & subscribed (send/CoreDepositWallet)")

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            if msg.get("channel") != "userNonFundingLedgerUpdates":
                return
            for upd in (msg.get("data", {}) or {}).get("nonFundingLedgerUpdates", []) or []:
                try:
                    t_ms = int(upd.get("time", 0))
                except Exception:
                    continue
                if t_ms < self.experiment_start_ms:
                    continue
                d = upd.get("delta", {}) or {}
                if d.get("type") != "send":
                    continue
                if str(d.get("user", "")).lower() != self.core_wallet:
                    continue
                if str(d.get("token", "")) != self.token:
                    continue
                try:
                    amt = Decimal(str(d.get("amount") or d.get("usdcValue") or "0"))
                except Exception:
                    amt = Decimal("0")
                if amt < self.min_amount:
                    continue
                print(f"\n[hc] CCTP credit found (send from CoreDepositWallet): {d}")
                self.result.found = True
                self.result.ledger_time_ms = t_ms
                self.result.ledger_usdc = amt
                self.result.raw_event = upd
                self._event.set()
                try:
                    ws.close()
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[hc] credit listener msg error: {e}")

    def on_error(self, ws, error):
        print(f"[hc] credit listener WS error: {error}")

    def _run(self):
        import websocket
        while not self._stop and not self.result.found:
            try:
                self.ws = websocket.WebSocketApp(cfg.HL_WS_URL, on_open=self.on_open,
                                                 on_message=self.on_message, on_error=self.on_error)
                self.ws.run_forever()
            except Exception as e:
                print(f"[hc] credit listener setup error: {e}")
            if not self._stop and not self.result.found:
                time.sleep(3)

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def wait_for_deposit(self, timeout_sec: int = 300) -> CreditResult:
        self._event.wait(timeout=timeout_sec)
        return self.result

# =====================================================================
# 修正2: burn 確定直後の即チェックポイント保存
# =====================================================================
def write_burn_checkpoint(exp_id: int, burn: "BurnTxInfo", out_dir: Path = None) -> Path:
    out_dir = out_dir or cfg.TEST_DIR
    path = out_dir / f"burn_checkpoint_{exp_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "exp_id": exp_id,
        "burn_tx_hash": burn.tx_hash,
        "block_number": burn.block_number,
        "block_ts_ms": burn.block_ts_ms,
        "gas_used": burn.gas_used,
        "gas_price_wei": burn.gas_price_wei,
        "tx_fee_eth": str(burn.tx_fee_eth),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
    print(f"[checkpoint] burn saved -> {path}")
    return path

# =====================================================================
# 修正4: credit 待ちのハートビート付きラッパ（T1 listener を非改変で利用）
# =====================================================================
def wait_for_credit_with_heartbeat(listener, timeout_sec: int = 300):
    start = time.time()
    deadline = start + timeout_sec
    last_hb = start
    while time.time() < deadline:
        if listener._event.wait(timeout=1.0):
            return listener.result
        now = time.time()
        if now - last_hb >= 10:
            print(f"[heartbeat] still waiting credit ({int(now - start)}秒経過)")
            last_hb = now
    print(f"[hc] credit TIMEOUT after {timeout_sec}s")
    return listener.result

# =====================================================================
# CSV row 構築（finally から取得済み値のみで安全に組む）
# =====================================================================
def build_row(exp_id, burn, iris, mint, hc, arb_height_at_attest,
              max_fee_used=None, forward_state="") -> dict:
    if max_fee_used is None:
        max_fee_used = cfg.DEPOSIT_MAX_FEE
    t1 = burn.block_ts_ms if burn else 0          # Arbitrum burn block ts (CHAIN, ms=秒×1000)
    t2_raw = iris.complete_local_ms if iris else 0  # status==complete を高頻度poll検知した raw ローカル時刻 (LOCAL, ms)
    t2_5 = mint.block_ts_ms if (mint and mint.found) else 0  # HyperEVM mint block ts (CHAIN, ms)
    t3 = hc.ledger_time_ms if (hc and hc.found and hc.ledger_time_ms) else 0  # HC credit ledger (CHAIN, ms)
    t0_ms = (burn.local_send_ns / 1e6) if burn else 0  # PC送信時刻 (LOCAL, ms)
    # 改修1: t2 クランプ。relayer が poll 検知より先に mint した場合(t2_raw > t2_5)、attestation は
    #   遅くとも t2_5 までに利用可能だったはずなので t2 = min(t2_raw, t2_5) に上限クランプ
    #   （t2 が物理的に t2.5 を超えないよう整合）。= 「attestation即mint(relay一瞬)・信頼層律速」を意味する。
    if t2_raw and t2_5:
        t2 = min(t2_raw, t2_5)
    else:
        t2 = t2_raw
    # 改修2: 物理量ベースの構造分解（t1=burn block 起点で統一。t2は±0.25s精度の物理時刻）
    #   iris_wait           = t2 - t1   : burn→attestation完了 = CCTP信頼層レイテンシ
    #   attestation_to_mint = t2_5 - t2 : attestation→HEVM mint = relay+mint実行
    #   credit_wait         = t3 - t2_5 : mint→HyperCore credit
    #   src_inclusion       = t2_5 - t1 : burn→mint合算（参考）
    iris_wait = (t2 - t1) if (t1 and t2) else ""
    attestation_to_mint = (t2_5 - t2) if (t2 and t2_5) else ""
    credit_wait = (t3 - t2_5) if (t2_5 and t3) else ""
    src_inclusion = (t2_5 - t1) if (t1 and t2_5) else ""
    # E2E を起点別に2列へ分離（終点はどちらも t3）
    #   onchain   = t3 - t1（= iris_wait + attestation_to_mint + credit_wait, telescoping で厳密一致）
    #   wallclock = t3 - t0（PC送信時刻ベース。T1 の latency(ms) と同義）
    e2e_onchain = (t3 - t1) if (t1 and t3) else ""
    e2e_wallclock = round(t3 - t0_ms, 3) if (t3 and t0_ms) else ""
    rtt_offset = round(t1 - t0_ms, 3) if (t1 and t0_ms) else ""
    return {
        "experiment_id": exp_id, "direction": "deposit", "amount_usdc(usdc)": str(DEPOSIT_AMOUNT_USDC),
        "cctp_nonce": iris.event_nonce if iris else "", "message_hash": iris.message_hash if iris else "",
        "t1_arb_burn_block_ts(ms)": t1, "t1_arb_burn_block_number": (burn.block_number if burn else ""),
        "arb_burn_tx_hash": (burn.tx_hash if burn else ""),
        "t2_iris_attestation_complete(ms)": t2_raw,
        "t2_iris_complete_local(ms)": t2,
        "t2_5_hevm_mint_block_ts(ms)": t2_5, "t2_5_hevm_mint_block_number": (mint.block_number if mint else ""),
        "hevm_mint_tx_hash": (mint.tx_hash if mint else ""),
        "t3_hc_credit_ledger_time(ms)": t3,
        "minFinalityThreshold_set": cfg.DEPOSIT_MIN_FINALITY_THRESHOLD,
        "finalityThresholdExecuted": iris.finality_threshold_executed if iris else "",
        "forward_state": forward_state or "",
        "maxFee(usdc_atomic)": max_fee_used, "feeExecuted(usdc_atomic)": iris.fee_executed if iris else "",
        "arb_gas_used(gas)": (burn.gas_used if burn else ""), "arb_gas_price(wei)": (burn.gas_price_wei if burn else ""),
        "arb_tx_fee(eth)": (f"{burn.tx_fee_eth:.18f}" if burn else ""),
        "t0_local_send(ns)": (burn.local_send_ns if burn else ""),
        "rtt_offset(ms)": f"{rtt_offset:.3f}" if rtt_offset != "" else "",
        "arb_confirmations_at_attestation(blocks)": (arb_height_at_attest - burn.block_number) if (arb_height_at_attest and burn) else "",
        "iris_wait(ms)": iris_wait, "attestation_to_mint(ms)": attestation_to_mint,
        "credit_wait(ms)": credit_wait, "src_inclusion(ms)": src_inclusion,
        "E2E_dep_onchain(ms)": e2e_onchain, "E2E_dep_wallclock(ms)": e2e_wallclock,
    }

# =====================================================================
# 本番計測（Phase B / --broadcast 時のみ）
# =====================================================================
def run_broadcast(exp_id_override: Optional[int] = None, prod: bool = False):
    if not _T1_LISTENER_OK:
        print(f"[CRITICAL] T1 検知器の import に失敗: {_T1_IMPORT_ERR}")
        sys.exit(1)

    # D3: 出力先を test/prod で切替。本番は1ファイル(cfg.PROD_CSV)に追記、連番継続。
    csv_path = cfg.PROD_CSV if prod else cfg.TEST_CSV
    out_dir = cfg.PROD_DIR if prod else cfg.TEST_DIR

    amount_raw = int(DEPOSIT_AMOUNT_USDC * (10**6))
    exp_id = exp_id_override if exp_id_override is not None else get_next_experiment_id(csv_path)
    exp_start_ms = int(time.time() * 1000)
    print(f"===== T4 Deposit CCTP Fast (BROADCAST{'/PROD' if prod else '/TEST'}) exp_id={exp_id} =====")
    print(f"[out] csv={csv_path}  intermediate_dir={out_dir}")

    # 0. 【送信直前チェック・修正5】recipient == HL_USER_ADDRESS を assert（不一致なら中止）
    assert_recipient_is_self()

    # 0b. 【Fast+Forwarding 化・C】fee API(forward=true) から maxFee を積み上げ動的取得（憶測で埋めない）。
    #     maxFee = ceil((base_fast_fee + forwardFee[med]) * 1.05)。
    #     ★安全弁: maxFee < forwardFee[med]（=Forwarding最低見積） or 取得失敗なら送信中止
    #       （exp_id=1 の INSUFFICIENT_FEE 配送失敗を二度と繰り返さない）。
    fee_info = compute_deposit_max_fee(amount_raw, buffer_pct=0.05, forward_tier="med")
    max_fee = fee_info["max_fee_atomic"]
    fwd_med = fee_info["forward_fee_med"]
    print(f"[fee] (a)={fee_info['raw_a']}")
    print(f"[fee] (b forward=true)={fee_info['raw_b']}")
    print(f"[fee] base={fee_info['base_fee_atomic']} forwardFee[med]={fwd_med} "
          f"[{fee_info['forward_source']}] +{int(fee_info['buffer_pct']*100)}% => maxFee={max_fee}")
    if not max_fee or not fwd_med or max_fee < fwd_med:
        print(f"[ABORT] maxFee={max_fee} < forwardFee[med]={fwd_med} or 取得失敗。配送失敗防止のため送信中止。")
        sys.exit(1)

    # 修正1: HyperCore WS 生ログ（独立スレッド）を burn 送信前に起動
    raw_path = out_dir / f"hc_ws_raw_{exp_id}.jsonl"
    raw_logger = HcRawWsLogger(HL_USER_ADDRESS, raw_path)
    raw_logger.start()

    # 1. HyperCore credit listener 起動（CCTP は send/CoreDepositWallet で届く・昨日実証）
    #    min_amount は手数料控除を見込んで burn 額の半分を下限ガードに。
    listener = HcCctpDepositListener(
        user_address=HL_USER_ADDRESS, experiment_start_ms=exp_start_ms,
        core_deposit_wallet=cfg.CORE_DEPOSIT_WALLET, token="USDC",
        min_amount=DEPOSIT_AMOUNT_USDC / 2)
    listener.start()
    time.sleep(2)

    # 2. Arbitrum 接続・approve（必要時・修正5）・burn
    w3 = make_arb_web3()
    usdc = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), abi=ERC20_ABI)
    allowance = usdc.functions.allowance(
        ARB_SENDER_ADDRESS, Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2)).call()
    if allowance < amount_raw:
        print(f"[approve] allowance {allowance} < {amount_raw} → approve 実行")
        send_approve(w3, amount_raw)

    # 送信直前に maxFee / minFinalityThreshold / destinationCaller を明示（実例と一致確認）
    print(f"[send] maxFee={max_fee} (atomic, ~{Decimal(max_fee)/Decimal(10**6)} USDC) "
          f"minFinalityThreshold={cfg.DEPOSIT_MIN_FINALITY_THRESHOLD} (<=1000 → Fast)")
    print(f"[send] destinationCaller={cfg.CCTP_FORWARDER_HEVM} (=CctpForwarder, 実例と一致) "
          f"mintRecipient={cfg.CCTP_FORWARDER_HEVM}")
    burn = send_burn(w3, amount_raw, max_fee)
    print(f"[burn] tx={burn.tx_hash} block={burn.block_number}")

    # 修正2: burn 確定直後に即チェックポイント保存・flush
    write_burn_checkpoint(exp_id, burn, out_dir)

    # 修正3: 検知 phase（Iris待ち〜mint観測〜credit待ち〜save_csv）を crash-safe に。
    #         例外/中断でも finally で取得済み値だけ save_csv を必ず実行する。
    iris = None
    mint = HevmMintResult(False)
    hc = None
    arb_height_at_attest = 0
    try:
        # 3. Iris attestation（修正4: timeout=300s, ハートビート）
        print("[iris] attestation 待ち ...")
        iris = poll_iris_attestation(cfg.ARB_DOMAIN_ID, burn.tx_hash, timeout_sec=300)
        if iris is None:
            print("[ERROR] Iris attestation timeout")
        try:
            arb_height_at_attest = w3.eth.block_number
        except Exception:
            pass

        # Fast は feeExecuted が控除され、mint/credit は net = amount - feeExecuted になる。
        # iris から feeExecuted を取得し、mint 観測と credit 検知を net 額に合わせる。
        fee_exec = iris.fee_executed if (iris and iris.fee_executed is not None) else None
        expected_net_raw = (amount_raw - fee_exec) if (fee_exec is not None) else amount_raw
        accept_vals = {amount_raw, expected_net_raw}  # gross/net 双方を許容
        if fee_exec is not None:
            print(f"[fee] feeExecuted={fee_exec} (atomic, ~{Decimal(fee_exec)/Decimal(10**6)} USDC) "
                  f"→ net mint/credit ~{Decimal(expected_net_raw)/Decimal(10**6)} USDC")

        # 4. HyperEVM mint 観測（t2.5・修正4: timeout=180s, ハートビート）
        print("[hevm] mint 観測 ...")
        hl_w3 = make_hl_web3()
        start_block = max(0, hl_w3.eth.block_number - 5)
        mint = watch_hevm_mint(hl_w3, amount_raw, start_block, timeout_sec=180,
                               accept_values=accept_vals)

        # 5. HyperCore credit（t3・修正4: timeout=300s, ハートビート）
        #    新検知器は send/CoreDepositWallet でマッチするため net 額更新は不要
        #    （credit は手数料控除後 ~amount-feeExecuted で届く）。
        print("[hc] credit 待ち（send from CoreDepositWallet）...")
        hc = wait_for_credit_with_heartbeat(listener, timeout_sec=300)

        # forwardState は forwarding 完了後に確定するため credit 検知後に取得（best-effort）
        forward_state = fetch_forward_state(burn.tx_hash) or ""
        print(f"[iris] forwardState = {forward_state}")
    finally:
        # finally: 取得済み値だけで row を組み、必ず save_csv（未取得は空欄）
        forward_state = locals().get("forward_state", "") or ""
        row = build_row(exp_id, burn, iris, mint, hc, arb_height_at_attest,
                        max_fee_used=max_fee, forward_state=forward_state)
        save_csv(row, csv_path)
        print("===== 計測結果（finally 保存）=====")
        for k in CSV_HEADER:
            print(f"  {k} = {row[k]}")
        print(f"Saved to {csv_path}")
        raw_logger.stop()
        print(f"[hc-raw] logger stopped. messages logged = {raw_logger.msg_count}, file = {raw_path}")

# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="T4 Deposit CCTP Fast measurement")
    ap.add_argument("--broadcast", action="store_true",
                    help="実送信して計測する（指定しなければドライラン）")
    ap.add_argument("--exp-id", type=int, default=None,
                    help="experiment_id を明示指定（未指定なら CSV から自動採番）")
    ap.add_argument("--prod", action="store_true",
                    help="本番モード（cfg.PROD_CSV に追記・連番継続）。未指定は test。")
    args = ap.parse_args()
    if args.broadcast:
        run_broadcast(exp_id_override=args.exp_id, prod=args.prod)
    else:
        run_dry_run()

if __name__ == "__main__":
    main()
