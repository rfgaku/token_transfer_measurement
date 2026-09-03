#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
withdraw_cctp_measure.py
T4: HyperCore → Arbitrum USDC Withdraw via CCTP V2 のレイテンシ測定。

■ 方式（Circle公式 "Withdraw USDC from HyperCore to EVM chains" で確定・推測実装禁止）
- HyperCore へ単一の EIP-712 署名アクション sendToEvmWithData を /exchange に POST するだけで、
  HyperCore 引落 → HyperEVM での CCTP burn → Iris attestation → Arbitrum で自動 mint
  （action.data="0x" による Arbitrum 宛 自動 forwarding）まで完結する。
  ※実測（test exp_id=2）: finalityThresholdExecuted=2000（Standard/Finalized）で実行（"Fast default" 表現と異なる）。
    HyperBFT の高速 finality により Finalized でも数秒で完結。maxFee≈0.2 USDC(=forwarding満額・feeExecuted=200000)、
    着金=amount−0.2（額に非比例の固定手数料）。値はハードコードせず必ず実測記録。
- 署名は hyperliquid SDK の user_signed_payload/sign_inner をそのまま再利用（同一の EIP-712 規約）。
  ただし SDK の sign_user_signed_action は signatureChainId を 0x66eee に強制するため使わず、
  Circle 仕様の signatureChainId="0xa4b1"(=42161, Arbitrum) を自前で設定して署名する。
  ★T1 の withdraw_from_bridge(msgpack/withdraw3) の署名は流用しない。

■ 5点タイミング（deposit と対称＋withdraw 固有の多段分解）
  t0   = PC が sendToEvmWithData を /exchange へ送信した時刻 (ns)
  t_hc_debit = HyperCore が残高引落した ledger 時刻 (ms, WS userNonFundingLedgerUpdates)
               ※CCTP withdraw の delta.type は不明 → 生 WS ログを送信前から全件記録し実構造から特定。
  t1   = HyperEVM 上で CCTP burn が起きたブロック ts (ms, +block_number, +burn_tx_hash)
         検知: HyperEVM MessageTransmitterV2(0x81D4..) の MessageSent(bytes) を送信ブロック以降 polling。
               message 内に自分の Arbitrum 宛先(20byte) を含み destinationDomain=3 のものを burn tx とする。
  t2   = burn_tx で Iris /v2/messages/19 を 0.25s poll、status=="complete" 検知時刻 (ms)。t2=min(t2_raw,t3)。
  t3   = Arbitrum で自分宛に USDC mint されたブロック ts (ms, +block_number, +arb_mint_tx_hash)
         検知: Arbitrum USDC Transfer(from=ZERO, to=ARB_SENDER_ADDRESS) を polling（CCTP mint は from=ZERO）。

■ 安全
  既定はドライラン（送信しない）。実送信は --broadcast 明示時のみ。--prod 明示時のみ 5 USDC。
  --prod 無しは必ず 1 USDC（test）。出力 test は cfg.TEST_CSV、本番は cfg.PROD_CSV に連番追記。
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

# --- リポジトリルートを sys.path に追加（PYTHONPATH 無し・任意ディレクトリから実行可能に） ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from t4_cctp.withdraw import config as cfg

# --- hyperliquid SDK の EIP-712 署名ヘルパを再利用（同一規約・signatureChainId のみ自前設定） ---
from eth_account import Account
from hyperliquid.utils.signing import (
    user_signed_payload,
    sign_inner,
    recover_user_from_user_signed_action,
)

load_dotenv()

# =====================================================================
# 環境変数（鍵は .env のみ。ログには出さない）
# =====================================================================
try:
    ARB_SENDER_ADDRESS = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    ARB_SENDER_PRIVATE_KEY = os.environ["ARB_SENDER_PRIVATE_KEY"]
except KeyError as e:
    raise KeyError(f"環境変数 {e} が読み込めませんでした。.env を確認してください。")

_WALLET = Account.from_key(ARB_SENDER_PRIVATE_KEY)
assert Web3.to_checksum_address(_WALLET.address) == ARB_SENDER_ADDRESS, (
    f"鍵から導出したアドレス {_WALLET.address} が ARB_SENDER_ADDRESS {ARB_SENDER_ADDRESS} と不一致")

# Transfer(address,address,uint256) / MessageSent(bytes) topic
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()
MESSAGE_SENT_TOPIC = Web3.keccak(text="MessageSent(bytes)").hex()
# CCTP V2 MessageReceived（受信=mint側 MessageTransmitterV2 が発行。evm-cctp-contracts v2 で確認）
#   event MessageReceived(address indexed caller, uint32 sourceDomain, bytes32 indexed nonce,
#                         bytes32 sender, uint32 indexed finalityThresholdExecuted, bytes messageBody)
#   indexed: caller=topic[1], nonce=topic[2], finalityThresholdExecuted=topic[3]
#   data(非indexed): sourceDomain, sender, messageBody
MESSAGE_RECEIVED_TOPIC = Web3.keccak(
    text="MessageReceived(address,uint32,bytes32,bytes32,uint32,bytes)").hex()
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "decimals", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint8"}]},
]

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

def fmt_amount(amount: str) -> str:
    """金額文字列を Circle 例（"1" / "5"）に合わせ末尾0を除去した正準10進文字列にする。
    "1.0"→"1", "5.0"→"5", "1.50"→"1.5"。指数表記は避ける。"""
    d = Decimal(str(amount)).normalize()
    return format(d, "f")

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
# sourceDex 確定（送金前チェック・info API で実残高所在を確認）
# =====================================================================
def determine_source_dex(amount: Decimal) -> dict:
    """spotClearinghouseState / clearinghouseState を照会し USDC 残高の所在を確認。
    spot に十分あれば "spot"、無ければ perp("")。判定根拠も返す。"""
    user = ARB_SENDER_ADDRESS.lower()
    spot_avail = Decimal("0"); perp_withdrawable = Decimal("0")
    spot_raw = perp_raw = None
    try:
        r = requests.post(cfg.HL_INFO_URL, json={"type": "spotClearinghouseState", "user": user}, timeout=10)
        spot_raw = r.json()
        for b in spot_raw.get("balances", []) or []:
            if b.get("coin") == "USDC":
                total = Decimal(str(b.get("total") or "0"))
                hold = Decimal(str(b.get("hold") or "0"))
                spot_avail = total - hold
    except Exception as e:
        print(f"[sourceDex] spot 照会失敗: {e}")
    try:
        r = requests.post(cfg.HL_INFO_URL, json={"type": "clearinghouseState", "user": user}, timeout=10)
        perp_raw = r.json()
        perp_withdrawable = Decimal(str(perp_raw.get("withdrawable") or "0"))
    except Exception as e:
        print(f"[sourceDex] perp 照会失敗: {e}")

    if spot_avail >= amount:
        source_dex, reason = "spot", f"spot available {spot_avail} >= {amount}"
    elif perp_withdrawable >= amount:
        source_dex, reason = "", f"perp withdrawable {perp_withdrawable} >= {amount}（spot不足）"
    else:
        source_dex, reason = cfg.DEFAULT_SOURCE_DEX, (
            f"spot avail {spot_avail} / perp wd {perp_withdrawable} ともに不足→既定 '{cfg.DEFAULT_SOURCE_DEX}'")
    return {"source_dex": source_dex, "reason": reason,
            "spot_available": spot_avail, "perp_withdrawable": perp_withdrawable}

# =====================================================================
# sendToEvmWithData action 構築 + EIP-712 署名（Circle仕様・signatureChainId=0xa4b1）
# =====================================================================
def build_send_to_evm_action(amount_str: str, source_dex: str, nonce_ms: int) -> dict:
    """Circle "Withdraw USDC from HyperCore to EVM chains" の sendToEvmWithData action。
    destinationRecipient は hex（小文字）で自分の Arbitrum アドレス。"""
    return {
        "type": "sendToEvmWithData",
        "hyperliquidChain": cfg.HYPERLIQUID_CHAIN,            # "Mainnet"
        "signatureChainId": cfg.SIGNATURE_CHAIN_ID,           # "0xa4b1"（=42161, 署名 domain 用）
        "token": cfg.WITHDRAW_TOKEN,                          # "USDC"
        "amount": amount_str,                                 # "1" / "5"（人間可読 USDC 文字列）
        "sourceDex": source_dex,                              # "spot" / ""(perp)
        "destinationRecipient": ARB_SENDER_ADDRESS.lower(),   # 自分の Arbitrum アドレス(hex,小文字)
        "addressEncoding": cfg.ADDRESS_ENCODING,              # "hex"
        "destinationChainId": cfg.DEST_CHAIN_ID_CCTP_DOMAIN,  # 3 (Arbitrum CCTP domain)
        "gasLimit": cfg.GAS_LIMIT,                            # 200000
        "data": cfg.DATA_HEX,                                 # "0x"（自動 forwarding 有効）
        "nonce": nonce_ms,                                    # 現在時刻 ms
    }

def sign_send_to_evm(action: dict) -> dict:
    """sendToEvmWithData を EIP-712 署名し {r,s,v} を返す。
    domain={name:HyperliquidSignTransaction, version:1, chainId:int(signatureChainId,16)=42161,
            verifyingContract:0x0}, primaryType=HyperliquidTransaction:SendToEvmWithData。
    SDK の user_signed_payload/sign_inner を流用（types に signatureChainId は含めない）。"""
    data = user_signed_payload(cfg.SEND_TO_EVM_PRIMARY_TYPE, cfg.SEND_TO_EVM_SIGN_TYPES, action)
    return sign_inner(_WALLET, data)

def verify_signature_self(action: dict, signature: dict) -> str:
    """署名から署名者アドレスを復元（自己検証）。ARB_SENDER_ADDRESS と一致するはず。"""
    recovered = recover_user_from_user_signed_action(
        dict(action), signature, cfg.SEND_TO_EVM_SIGN_TYPES, cfg.SEND_TO_EVM_PRIMARY_TYPE, is_mainnet=True)
    return Web3.to_checksum_address(recovered)

@dataclass
class SendResult:
    ok: bool = False
    local_send_ns: int = 0
    nonce: int = 0
    action: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)
    http_status: int = 0

def post_send_to_evm(action: dict, signature: dict) -> SendResult:
    """{action, nonce, signature} を /exchange に POST。t0(local_send_ns) を post 直前に記録。"""
    payload = {
        "action": action,
        "nonce": action["nonce"],
        "signature": signature,
        "vaultAddress": None,
        "expiresAfter": None,
    }
    local_send_ns = time.time_ns()
    resp = requests.post(cfg.HL_EXCHANGE_URL, json=payload, timeout=15)
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    ok = (resp.status_code == 200 and isinstance(body, dict) and body.get("status") == "ok")
    return SendResult(ok=ok, local_send_ns=local_send_ns, nonce=action["nonce"],
                      action=action, response=body, http_status=resp.status_code)

# =====================================================================
# 修正1: HyperCore WS 生ログ（独立スレッド・全件記録）。delta.type 未知のため全件残す。
# =====================================================================
class HcRawWsLogger:
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
# HyperCore 引落検知（t_hc_debit）。delta.type 未知のため「金額が引出額に近い最初の更新」を best-effort 検知。
# 取れなくても CSV 空欄で続行（raw ログから後追い可能）。
# =====================================================================
@dataclass
class DebitResult:
    found: bool = False
    ledger_time_ms: Optional[int] = None
    amount: Optional[Decimal] = None
    delta_type: Optional[str] = None
    raw_event: Optional[dict] = None

class HcDebitListener:
    """userNonFundingLedgerUpdates を購読し、experiment_start_ms 以降で
    引出額(amount)の概ね [0.5x, 2.0x] の金額を持つ最初の delta を引落イベントとして拾う。
    CCTP withdraw の delta.type は未知なので type ではなく金額帯でマッチ（type は記録のみ）。"""

    def __init__(self, user_address: str, experiment_start_ms: int, amount: Decimal):
        self.user_address = user_address.lower()
        self.experiment_start_ms = experiment_start_ms
        self.amount = amount
        self.lo = amount * Decimal("0.5")
        self.hi = amount * Decimal("2.0")
        self.result = DebitResult()
        self._event = threading.Event()
        self.ws = None
        self._stop = False

    def on_open(self, ws):
        ws.send(json.dumps({"method": "subscribe", "subscription": {
            "type": "userNonFundingLedgerUpdates", "user": self.user_address}}))
        print("[hc-debit] listener connected & subscribed")

    @staticmethod
    def _extract_amount(delta: dict) -> Optional[Decimal]:
        for k in ("amount", "usdc", "usdcValue", "usdcAmount"):
            v = delta.get(k)
            if v is not None:
                try:
                    return abs(Decimal(str(v)))
                except Exception:
                    continue
        return None

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
                amt = self._extract_amount(d)
                if amt is None:
                    continue
                if not (self.lo <= amt <= self.hi):
                    continue
                print(f"\n[hc-debit] 引落候補検知: type={d.get('type')} amount={amt} delta={d}")
                self.result.found = True
                self.result.ledger_time_ms = t_ms
                self.result.amount = amt
                self.result.delta_type = d.get("type")
                self.result.raw_event = upd
                self._event.set()
                try:
                    ws.close()
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[hc-debit] msg error: {e}")

    def on_error(self, ws, error):
        print(f"[hc-debit] WS error: {error}")

    def _run(self):
        import websocket
        while not self._stop and not self.result.found:
            try:
                self.ws = websocket.WebSocketApp(cfg.HL_WS_URL, on_open=self.on_open,
                                                 on_message=self.on_message, on_error=self.on_error)
                self.ws.run_forever()
            except Exception as e:
                print(f"[hc-debit] setup error: {e}")
            if not self._stop and not self.result.found:
                time.sleep(3)

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def wait(self, timeout_sec: int) -> DebitResult:
        start = time.time(); last_hb = start
        while time.time() < start + timeout_sec:
            if self._event.wait(timeout=1.0):
                return self.result
            now = time.time()
            if now - last_hb >= 10:
                print(f"[heartbeat] still waiting hc-debit ({int(now-start)}秒経過)")
                last_hb = now
        print(f"[hc-debit] TIMEOUT after {timeout_sec}s（raw ログから後追い可）")
        return self.result

    def stop(self):
        self._stop = True
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

# =====================================================================
# HyperEVM CCTP burn 検知（t1）: MessageTransmitterV2.MessageSent(bytes)
# =====================================================================
@dataclass
class BurnResult:
    found: bool = False
    block_number: int = 0
    block_ts_ms: int = 0
    tx_hash: str = "N/A"
    message_hex: Optional[str] = None
    cctp_nonce_hex: Optional[str] = None
    dest_domain: Optional[int] = None
    body_amount: Optional[int] = None

def _decode_message_sent(data_hex: str) -> bytes:
    """MessageSent(bytes) の data から message バイト列を取り出す（offset+length+payload）。"""
    raw = bytes.fromhex(data_hex[2:] if data_hex.startswith("0x") else data_hex)
    if len(raw) < 64:
        return b""
    length = int.from_bytes(raw[32:64], "big")
    return raw[64:64 + length]

def _parse_cctp_v2_message(msg: bytes) -> dict:
    """CCTP V2 message header + BurnMessageV2 body の必要フィールドを抽出。"""
    out = {}
    if len(msg) >= 148:
        out["version"] = int.from_bytes(msg[0:4], "big")
        out["source_domain"] = int.from_bytes(msg[4:8], "big")
        out["dest_domain"] = int.from_bytes(msg[8:12], "big")
        out["nonce_hex"] = "0x" + msg[12:44].hex()
        body = msg[148:]
        if len(body) >= 100:
            out["body_mint_recipient"] = "0x" + body[36:68].hex()
            out["body_amount"] = int.from_bytes(body[68:100], "big")
    return out

def watch_hevm_burn(hl_w3: Web3, start_block: int, timeout_sec: int) -> BurnResult:
    """HyperEVM MessageTransmitterV2 の MessageSent を start_block 以降 polling。
    destinationDomain==3(Arbitrum) かつ message 内に自分の Arbitrum 宛先(20byte) を含むものを burn tx とする。"""
    mt = Web3.to_checksum_address(cfg.HEVM_MESSAGE_TRANSMITTER_V2)
    my_hex = ARB_SENDER_ADDRESS[2:].lower()
    start = time.time(); deadline = start + timeout_sec; last_hb = start
    frm = start_block
    while time.time() < deadline:
        try:
            latest = hl_w3.eth.block_number
            if latest >= frm:
                logs = hl_w3.eth.get_logs({
                    "fromBlock": frm, "toBlock": latest, "address": mt,
                    "topics": [MESSAGE_SENT_TOPIC],
                })
                for lg in logs:
                    data_hex = lg["data"].hex() if isinstance(lg["data"], (bytes, bytearray)) else lg["data"]
                    msg = _decode_message_sent(data_hex)
                    if not msg:
                        continue
                    info = _parse_cctp_v2_message(msg)
                    if info.get("dest_domain") != cfg.ARB_DOMAIN_ID:
                        continue
                    if my_hex not in msg.hex().lower():
                        continue
                    blk = hl_w3.eth.get_block(lg["blockNumber"])
                    th = lg["transactionHash"]
                    return BurnResult(
                        found=True, block_number=lg["blockNumber"], block_ts_ms=blk.timestamp * 1000,
                        tx_hash=(th.hex() if hasattr(th, "hex") else th),
                        message_hex="0x" + msg.hex(), cctp_nonce_hex=info.get("nonce_hex"),
                        dest_domain=info.get("dest_domain"), body_amount=info.get("body_amount"))
                frm = latest + 1
        except Exception as e:
            print(f"[hevm-burn] watch error: {e}")
        now = time.time()
        if now - last_hb >= 10:
            print(f"[heartbeat] still waiting hevm burn ({int(now-start)}秒経過)")
            last_hb = now
        time.sleep(2)
    print(f"[hevm-burn] TIMEOUT after {timeout_sec}s")
    return BurnResult(False)

# =====================================================================
# 第2層: HyperEVM burn ライブ捕捉ワーカー（WS優先 + HTTP小範囲pollフォールバック）
#   公開RPCの非アーカイブ/100req/min/WS非対応が主因だったため、アーカイブRPC(cfg.HL_EVM_RPC_ARCHIVE)を使用。
#   捕捉後ただちに Iris を burn_tx で poll し iris_live(t2 のライブ時刻) を得る。
# =====================================================================
def make_hevm_archive_web3() -> Web3:
    provider = Web3.HTTPProvider(
        cfg.HL_EVM_RPC_ARCHIVE,
        request_kwargs={"timeout": 10, "headers": {"User-Agent": "Mozilla/5.0"}})
    w3 = Web3(provider)
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return w3

def _match_my_burn(msg: bytes, my_hex: str, gross_raw: int) -> bool:
    """destinationDomain==3(Arbitrum) かつ（message 内に自分の宛先(20B) を含む OR body_amount==送金gross）。"""
    info = _parse_cctp_v2_message(msg)
    if info.get("dest_domain") != cfg.ARB_DOMAIN_ID:
        return False
    if my_hex in msg.hex().lower():
        return True
    if gross_raw and info.get("body_amount") == gross_raw:
        return True
    return False

class HevmBurnWatcher(threading.Thread):
    """送信前に start。HyperEVM の MessageTransmitterV2(0x81D4) MessageSent をライブ捕捉して
    burn(t1) を得る。HL_EVM_WS_URL があれば eth_subscribe(logs)、無ければ HTTP 小範囲 poll。
    捕捉後ただちに Iris を burn_tx で poll し iris_live(t2) を取得する。"""

    def __init__(self, hevm_start_block: int, gross_raw: int, capture_timeout: int):
        super().__init__(daemon=True)
        self.hevm_start_block = hevm_start_block
        self.gross_raw = gross_raw
        self.capture_timeout = capture_timeout
        self.my_hex = ARB_SENDER_ADDRESS[2:].lower()
        self.burn = BurnResult(False)
        self.iris_live: Optional[IrisResult] = None
        self.hevm_height_at_attest = 0  # attestation 完了(t2)時点の HyperEVM 最新ブロック番号
        self._stopev = threading.Event()
        self._ws = None
        self._lock = threading.Lock()
        self.captured_via = None
        try:
            self.w3 = make_hevm_archive_web3()
        except Exception as e:
            print(f"[hevm-burn] archive web3 init error: {e}")
            self.w3 = None

    def stop(self):
        self._stopev.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    # archive getLogs は広範囲だと 413（QuickNode 制限）。小範囲チャンクで回避。
    HTTP_CHUNK = 4

    def _finalize_burn(self, block_number: int, tx_hash: str, msg: bytes, via: str) -> bool:
        """最初に捕捉した経路のみ採用（スレッドセーフ）。採用したら True。"""
        with self._lock:
            if self.burn.found:
                return False
            info = _parse_cctp_v2_message(msg)
            ts_ms = 0
            try:
                ts_ms = self.w3.eth.get_block(block_number).timestamp * 1000
            except Exception as e:
                print(f"[hevm-burn] block ts 取得失敗: {e}")
            self.captured_via = via
            self.burn = BurnResult(
                found=True, block_number=block_number, block_ts_ms=ts_ms, tx_hash=tx_hash,
                message_hex="0x" + msg.hex(), cctp_nonce_hex=info.get("nonce_hex"),
                dest_domain=info.get("dest_domain"), body_amount=info.get("body_amount"))
            print(f"[hevm-burn] ★捕捉 ({via}) tx={tx_hash} block={block_number} ts_ms={ts_ms}")
            return True

    def _consider_log(self, data_hex, block_number, tx_hash, via: str):
        msg = _decode_message_sent(data_hex) if data_hex else b""
        if msg and _match_my_burn(msg, self.my_hex, self.gross_raw):
            bn = int(block_number, 16) if isinstance(block_number, str) else block_number
            th = tx_hash.hex() if hasattr(tx_hash, "hex") else tx_hash
            self._finalize_burn(bn, th, msg, via)

    def _run_http(self):
        """小範囲(<=HTTP_CHUNK)に区切って getLogs（413回避）。アーカイブRPC使用。WSのバックアップとして併走。"""
        mt = Web3.to_checksum_address(cfg.HEVM_MESSAGE_TRANSMITTER_V2)
        frm = self.hevm_start_block
        start = time.time(); last_hb = start
        while not self._stopev.is_set() and not self.burn.found and time.time() < start + self.capture_timeout:
            try:
                latest = self.w3.eth.block_number
                while frm <= latest and not self.burn.found and not self._stopev.is_set():
                    to = min(latest, frm + self.HTTP_CHUNK - 1)
                    logs = self.w3.eth.get_logs({
                        "fromBlock": frm, "toBlock": to, "address": mt,
                        "topics": [MESSAGE_SENT_TOPIC]})
                    for lg in logs:
                        dh = lg["data"].hex() if isinstance(lg["data"], (bytes, bytearray)) else lg["data"]
                        self._consider_log(dh, lg["blockNumber"], lg["transactionHash"], "HTTP")
                    frm = to + 1
            except Exception as e:
                print(f"[hevm-burn] http poll error: {e}")
            now = time.time()
            if now - last_hb >= 10:
                print(f"[heartbeat] capturing hevm burn ({int(now-start)}s, HTTP frm={frm})")
                last_hb = now
            self._stopev.wait(2.0)

    def _run_ws(self):
        """eth_subscribe(logs) で MessageSent をライブ受信（主経路・低レート）。"""
        import websocket
        mt = Web3.to_checksum_address(cfg.HEVM_MESSAGE_TRANSMITTER_V2)
        sub = {"jsonrpc": "2.0", "id": 1, "method": "eth_subscribe",
               "params": ["logs", {"address": mt, "topics": [MESSAGE_SENT_TOPIC]}]}
        start = time.time()

        def on_open(ws):
            ws.send(json.dumps(sub))
            print("[hevm-burn] WS subscribed eth_subscribe(logs)")

        def on_message(ws, message):
            try:
                mm = json.loads(message)
                res = (mm.get("params") or {}).get("result")
                if isinstance(res, dict):
                    self._consider_log(res.get("data"), res.get("blockNumber"),
                                       res.get("transactionHash"), "WS")
                    if self.burn.found:
                        ws.close()
            except Exception as e:
                print(f"[hevm-burn] ws msg error: {e}")

        def on_error(ws, error):
            print(f"[hevm-burn] ws error: {error}")

        while not self._stopev.is_set() and not self.burn.found and time.time() < start + self.capture_timeout:
            try:
                self._ws = websocket.WebSocketApp(
                    cfg.HL_EVM_WS_URL, on_open=on_open, on_message=on_message, on_error=on_error)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"[hevm-burn] ws setup error: {e}")
            if not self._stopev.is_set() and not self.burn.found:
                time.sleep(2)

    def run(self):
        if self.w3 is None:
            print("[hevm-burn] archive web3 未初期化のため捕捉スキップ")
            return
        # WS（あれば）と HTTP 小範囲poll を併走。先に捕捉した方を採用。
        threads = []
        if cfg.HL_EVM_WS_URL:
            tws = threading.Thread(target=self._run_ws, daemon=True); tws.start(); threads.append(tws)
        thttp = threading.Thread(target=self._run_http, daemon=True); thttp.start(); threads.append(thttp)
        # 捕捉 or タイムアウトまで待機
        start = time.time()
        while time.time() < start + self.capture_timeout and not self.burn.found and not self._stopev.is_set():
            time.sleep(0.5)
        if not self.burn.found:
            print(f"[hevm-burn] capture TIMEOUT after {self.capture_timeout}s（WS/HTTP両経路）")
        self._stopev.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        # 捕捉できたら即 Iris を burn_tx で poll（t2 ライブ）。
        if self.burn.found:
            print("[iris] (live) attestation 待ち by burn_tx ...")
            # poll 自体は stop に関係なく実行したいので一時的にクリア不要（_stop は capture 用）
            self.iris_live = poll_iris_attestation(self.burn.tx_hash, cfg.IRIS_TIMEOUT_SEC)
            if self.iris_live:
                # attestation 完了(t2)時点の HyperEVM 最新ブロック番号を記録
                #   hevm_confirmations_at_attestation = この値 − t1_hevm_burn_block_number
                try:
                    self.hevm_height_at_attest = self.w3.eth.block_number
                except Exception:
                    pass
                print(f"[iris] (live) complete t2={self.iris_live.complete_local_ms} "
                      f"finalityThresholdExecuted={self.iris_live.finality_threshold_executed} "
                      f"hevm_height_at_attest={self.hevm_height_at_attest}")

# =====================================================================
# Iris attestation（t2）: source domain=19, GET /v2/messages/19?transactionHash={hevm_burn_tx}
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
    max_fee: Optional[int] = None
    mint_recipient: Optional[str] = None
    body_amount: Optional[int] = None
    forward_state: Optional[str] = None
    raw: dict = field(default_factory=dict)

def _to_int(v):
    try:
        return None if v is None else int(v)
    except Exception:
        return None

def _build_iris_result(m: dict) -> IrisResult:
    """Iris の messages[0] dict から IrisResult を構築（poll/nonce逆引き 共通）。
    complete_local_ms はこの瞬間のローカル時刻（poll 経路では status==complete 検知時刻＝t2 になる）。"""
    msg_hex = m.get("message")
    dec = m.get("decodedMessage") or {}
    body = dec.get("decodedMessageBody") or {}
    return IrisResult(
        complete_local_ms=int(time.time() * 1000),
        message_hex=msg_hex,
        attestation_hex=m.get("attestation"),
        event_nonce=str(m.get("eventNonce") or dec.get("nonce") or ""),
        message_hash=(Web3.keccak(hexstr=msg_hex).hex() if msg_hex else None),
        finality_threshold_executed=_to_int(
            dec.get("finalityThresholdExecuted") or m.get("finalityThresholdExecuted")),
        fee_executed=_to_int(body.get("feeExecuted") or m.get("feeExecuted")),
        max_fee=_to_int(body.get("maxFee") or m.get("maxFee")),
        mint_recipient=(body.get("mintRecipient") or dec.get("mintRecipient")),
        body_amount=_to_int(body.get("amount")),
        forward_state=m.get("forwardState"),
        raw=m)

def poll_iris_attestation(burn_tx: str, timeout_sec: int) -> Optional[IrisResult]:
    """burn tx hash で Iris を 0.25s poll し status=='complete' 検知時刻を t2 として返す（第2層・ライブ）。"""
    url = (cfg.IRIS_API_HOST + cfg.IRIS_MESSAGES_PATH.format(source_domain=cfg.HYPEREVM_DOMAIN_ID)
           + f"?transactionHash={burn_tx}")
    start = time.time(); deadline = start + timeout_sec; last_hb = start
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                msgs = (resp.json() or {}).get("messages") or []
                if msgs and msgs[0].get("status") == "complete":
                    return _build_iris_result(msgs[0])
        except Exception as e:
            print(f"[iris] poll error: {e}")
        now = time.time()
        if now - last_hb >= 10:
            print(f"[heartbeat] still waiting iris ({int(now-start)}秒経過)")
            last_hb = now
        time.sleep(cfg.IRIS_POLL_INTERVAL_SEC)
    print(f"[iris] TIMEOUT after {timeout_sec}s")
    return None

# =====================================================================
# 第1層: Arbitrum mint tx の MessageReceived 解析 + Iris nonce 逆引き
#   burn 観測の成否に依存せず nonce/finalityThresholdExecuted/fees/message_hash を確実に充填する。
# =====================================================================
@dataclass
class MintReceiptInfo:
    found: bool = False
    cctp_nonce: Optional[str] = None
    finality_threshold_executed: Optional[int] = None
    source_domain: Optional[int] = None

def parse_mint_receipt(w3: Web3, mint_tx_hash: str) -> MintReceiptInfo:
    """Arbitrum mint tx receipt の MessageReceived(emitter=ARB MessageTransmitterV2) から
    cctp_nonce=topic[2], finalityThresholdExecuted=topic[3], sourceDomain=data[0:32] を抽出。"""
    try:
        rcpt = w3.eth.get_transaction_receipt(mint_tx_hash)
    except Exception as e:
        print(f"[mint-receipt] receipt 取得失敗: {e}")
        return MintReceiptInfo()
    mt = Web3.to_checksum_address(cfg.ARB_MESSAGE_TRANSMITTER_V2).lower()
    for lg in rcpt.logs:
        try:
            if lg["address"].lower() != mt:
                continue
            topics = lg["topics"]
            if not topics or topics[0].hex().lower() != MESSAGE_RECEIVED_TOPIC.lower():
                continue
            nonce = "0x" + topics[2].hex()[-64:]
            fte = int(topics[3].hex(), 16)
            data = lg["data"]; dhex = data.hex() if hasattr(data, "hex") else data
            raw = bytes.fromhex(dhex[2:] if dhex.startswith("0x") else dhex)
            src = int.from_bytes(raw[0:32], "big") if len(raw) >= 32 else None
            return MintReceiptInfo(found=True, cctp_nonce=nonce,
                                   finality_threshold_executed=fte, source_domain=src)
        except Exception as e:
            print(f"[mint-receipt] log parse error: {e}")
    print("[mint-receipt] MessageReceived ログが見つかりません")
    return MintReceiptInfo()

def fetch_iris_by_nonce(nonce: str, timeout_sec: int = 60) -> Optional[IrisResult]:
    """nonce で Iris 逆引き GET /v2/messages/19?nonce=<nonce>。status=='complete' を待ち
    message_hash/attestation/maxFee/feeExecuted/mintRecipient/body_amount/forwardState を補完。
    ※mint 後に叩くため attestation '完了時刻' のライブ計測には使えない（メタデータ補完専用）。"""
    url = (cfg.IRIS_API_HOST + cfg.IRIS_MESSAGES_PATH.format(source_domain=cfg.HYPEREVM_DOMAIN_ID)
           + f"?nonce={nonce}")
    start = time.time(); deadline = start + timeout_sec; last = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                msgs = (resp.json() or {}).get("messages") or []
                if msgs:
                    last = msgs[0]
                    if last.get("status") == "complete":
                        return _build_iris_result(last)
        except Exception as e:
            print(f"[iris-nonce] error: {e}")
        time.sleep(cfg.IRIS_POLL_INTERVAL_SEC)
    if last is not None:
        print("[iris-nonce] complete 未到達。取得済み分を返す。")
        return _build_iris_result(last)
    print(f"[iris-nonce] TIMEOUT after {timeout_sec}s（messages 空）")
    return None

def fetch_forward_state(burn_tx: str) -> Optional[str]:
    try:
        url = (cfg.IRIS_API_HOST + cfg.IRIS_MESSAGES_PATH.format(source_domain=cfg.HYPEREVM_DOMAIN_ID)
               + f"?transactionHash={burn_tx}")
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            msgs = (resp.json() or {}).get("messages") or []
            if msgs:
                return msgs[0].get("forwardState")
    except Exception as e:
        print(f"[iris] forwardState 取得失敗: {e}")
    return None

# =====================================================================
# Arbitrum mint 検知（t3）: USDC Transfer(from=ZERO, to=ARB_SENDER_ADDRESS)
# =====================================================================
@dataclass
class MintResult:
    found: bool = False
    block_number: int = 0
    block_ts_ms: int = 0
    tx_hash: str = "N/A"
    amount_raw: Optional[int] = None

def watch_arb_mint(w3: Web3, start_block: int, timeout_sec: int) -> MintResult:
    usdc = Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS)
    to_topic = "0x" + ARB_SENDER_ADDRESS[2:].lower().rjust(64, "0")
    from_topic = "0x" + ZERO_ADDRESS[2:].rjust(64, "0")
    start = time.time(); deadline = start + timeout_sec; last_hb = start
    frm = start_block
    while time.time() < deadline:
        try:
            latest = w3.eth.block_number
            if latest >= frm:
                logs = w3.eth.get_logs({
                    "fromBlock": frm, "toBlock": latest, "address": usdc,
                    "topics": [TRANSFER_TOPIC, from_topic, to_topic],
                })
                if logs:
                    lg = logs[0]
                    val = int(lg["data"], 16) if isinstance(lg["data"], str) else int.from_bytes(lg["data"], "big")
                    blk = w3.eth.get_block(lg["blockNumber"])
                    th = lg["transactionHash"]
                    return MintResult(
                        found=True, block_number=lg["blockNumber"], block_ts_ms=blk.timestamp * 1000,
                        tx_hash=(th.hex() if hasattr(th, "hex") else th), amount_raw=val)
                frm = latest + 1
        except Exception as e:
            print(f"[arb-mint] watch error: {e}")
        now = time.time()
        if now - last_hb >= 10:
            print(f"[heartbeat] still waiting arb mint ({int(now-start)}秒経過)")
            last_hb = now
        time.sleep(2)
    print(f"[arb-mint] TIMEOUT after {timeout_sec}s")
    return MintResult(False)

# =====================================================================
# CSV（§2' 確定スキーマ・この順・改変禁止）
# =====================================================================
CSV_HEADER = [
    "experiment_id", "direction", "amount_usdc(usdc)", "amount_received(usdc)", "cctp_nonce", "message_hash",
    "t_hc_debit_ledger_time(ms)",
    "t1_hevm_burn_block_ts(ms)", "t1_hevm_burn_block_number", "hevm_burn_tx_hash",
    "t2_iris_attestation_complete(ms)", "t2_iris_complete_local(ms)",
    "t3_arb_mint_block_ts(ms)", "t3_arb_mint_block_number", "arb_mint_tx_hash",
    "finalityThresholdExecuted", "forward_state",
    "maxFee(usdc_atomic)", "feeExecuted(usdc_atomic)", "forwarding_fee(usdc_atomic)",
    "t0_local_send(ns)", "hc_debit_offset(ms)", "hevm_confirmations_at_attestation(blocks)",
    "hypercore_to_burn(ms)", "iris_wait(ms)", "attestation_to_mint(ms)",
    "E2E_wit_onchain(ms)", "E2E_wit_hypercore(ms)", "E2E_wit_wallclock(ms)",
]

def save_csv(row: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)

def build_row(exp_id, amount_str, send, debit, burn, iris_live, iris_meta, mint, mint_info,
              hevm_height_at_attest, forward_state="") -> dict:
    """iris_live=burn_txでのライブpoll（t2時刻が有効）。iris_meta=nonce逆引き（メタデータのみ・t2不可）。
    mint_info=Arbitrum mint tx の MessageReceived 由来（nonce/finalityThresholdExecuted・第1層確実コア）。"""
    iris = iris_live or iris_meta  # メタデータ用の最良ソース
    t0_ns = send.local_send_ns if send else 0
    t0_ms = (t0_ns / 1e6) if t0_ns else 0
    t_hc = debit.ledger_time_ms if (debit and debit.found and debit.ledger_time_ms) else 0
    t1 = burn.block_ts_ms if (burn and burn.found) else 0
    # ★t2 は burn_tx でのライブ poll でのみ有効。nonce 逆引き(iris_meta)は mint 後のため t2 に使わない。
    t2_raw = iris_live.complete_local_ms if iris_live else 0
    t3 = mint.block_ts_ms if (mint and mint.found) else 0
    # t2 クランプ: relayer が poll 検知より先に mint した場合 t2=min(t2_raw, t3)
    t2 = min(t2_raw, t3) if (t2_raw and t3) else t2_raw

    # 内訳（右端・単位付き）
    hypercore_to_burn = (t1 - t_hc) if (t_hc and t1) else ""
    iris_wait = (t2 - t1) if (t1 and t2) else ""
    attestation_to_mint = (t3 - t2) if (t2 and t3) else ""
    e2e_onchain = (t3 - t1) if (t1 and t3) else ""
    e2e_hypercore = (t3 - t_hc) if (t_hc and t3) else ""
    e2e_wallclock = round(t3 - t0_ms, 3) if (t3 and t0_ms) else ""
    hc_debit_offset = round(t_hc - t0_ms, 3) if (t_hc and t0_ms) else ""

    amount_received = (Decimal(mint.amount_raw) / Decimal(10**6)) if (mint and mint.found and mint.amount_raw is not None) else ""
    # cctp_nonce/finalityThresholdExecuted は mint_info（Arbitrum 受信側・最も確実）を優先
    cctp_nonce = ((mint_info.cctp_nonce if (mint_info and mint_info.cctp_nonce) else None)
                  or (iris.event_nonce if (iris and iris.event_nonce) else None)
                  or (burn.cctp_nonce_hex if (burn and burn.found) else None) or "")
    fte = ((mint_info.finality_threshold_executed if (mint_info and mint_info.finality_threshold_executed is not None) else None)
           if mint_info else None)
    if fte is None and iris:
        fte = iris.finality_threshold_executed
    msg_hash = (iris.message_hash if iris else "") or ""
    fwd_state = forward_state or (iris.forward_state if iris else "") or ""
    # forwarding_fee: 実測 feeExecuted があればそれを採用（withdraw は feeExecuted=0.2USDC が forwarding 相当）
    forwarding_fee_atomic = (iris.fee_executed if (iris and iris.fee_executed is not None)
                             else cfg.FORWARDING_FEE_FALLBACK_ATOMIC)

    return {
        "experiment_id": exp_id, "direction": "withdraw",
        "amount_usdc(usdc)": amount_str,
        "amount_received(usdc)": (str(amount_received) if amount_received != "" else ""),
        "cctp_nonce": cctp_nonce, "message_hash": msg_hash,
        "t_hc_debit_ledger_time(ms)": t_hc or "",
        "t1_hevm_burn_block_ts(ms)": t1 or "",
        "t1_hevm_burn_block_number": (burn.block_number if (burn and burn.found) else ""),
        "hevm_burn_tx_hash": (burn.tx_hash if (burn and burn.found) else ""),
        "t2_iris_attestation_complete(ms)": t2_raw or "",
        "t2_iris_complete_local(ms)": t2 or "",
        "t3_arb_mint_block_ts(ms)": t3 or "",
        "t3_arb_mint_block_number": (mint.block_number if (mint and mint.found) else ""),
        "arb_mint_tx_hash": (mint.tx_hash if (mint and mint.found) else ""),
        "finalityThresholdExecuted": (fte if fte is not None else ""),
        "forward_state": fwd_state,
        "maxFee(usdc_atomic)": (iris.max_fee if (iris and iris.max_fee is not None) else ""),
        "feeExecuted(usdc_atomic)": (iris.fee_executed if (iris and iris.fee_executed is not None) else ""),
        "forwarding_fee(usdc_atomic)": forwarding_fee_atomic,
        "t0_local_send(ns)": t0_ns or "",
        "hc_debit_offset(ms)": (f"{hc_debit_offset:.3f}" if hc_debit_offset != "" else ""),
        "hevm_confirmations_at_attestation(blocks)": (
            (hevm_height_at_attest - burn.block_number) if (hevm_height_at_attest and burn and burn.found) else ""),
        "hypercore_to_burn(ms)": hypercore_to_burn,
        "iris_wait(ms)": iris_wait,
        "attestation_to_mint(ms)": attestation_to_mint,
        "E2E_wit_onchain(ms)": e2e_onchain,
        "E2E_wit_hypercore(ms)": e2e_hypercore,
        "E2E_wit_wallclock(ms)": (f"{e2e_wallclock:.3f}" if e2e_wallclock != "" else ""),
    }

def write_send_checkpoint(exp_id: int, send: SendResult, out_dir: Path) -> Path:
    path = out_dir / f"send_checkpoint_{exp_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "exp_id": exp_id, "nonce": send.nonce, "local_send_ns": send.local_send_ns,
        "http_status": send.http_status, "ok": send.ok, "response": send.response,
        "action": {k: v for k, v in send.action.items()},  # 鍵は含まれない
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.flush()
    print(f"[checkpoint] send saved -> {path}")
    return path

# =====================================================================
# ドライラン
# =====================================================================
def test_hc_ws(timeout: int = 5) -> bool:
    import websocket
    ok = {"v": False}; ev = threading.Event()
    def on_open(ws):
        ws.send(json.dumps({"method": "subscribe", "subscription": {
            "type": "userNonFundingLedgerUpdates", "user": ARB_SENDER_ADDRESS.lower()}}))
    def on_message(ws, message):
        ok["v"] = True; ev.set(); ws.close()
    def on_error(ws, error):
        ev.set()
    def run():
        try:
            websocket.WebSocketApp(cfg.HL_WS_URL, on_open=on_open,
                                   on_message=on_message, on_error=on_error).run_forever()
        except Exception:
            ev.set()
    threading.Thread(target=run, daemon=True).start()
    ev.wait(timeout=timeout)
    return ok["v"]

def run_dry_run(prod: bool = False):
    amount_str = fmt_amount(cfg.WITHDRAW_AMOUNT_PROD if prod else cfg.WITHDRAW_AMOUNT_TEST)
    amount = Decimal(amount_str)
    mode = "PROD" if prod else "TEST"
    print("=" * 70)
    print(" T4 Withdraw (HyperCore → Arbitrum, CCTP V2 / 実測 finalityThresholdExecuted=2000) — DRY RUN")
    print(" ※ 送信しません（--broadcast 未指定）")
    print("=" * 70)
    print(f" mode           : {mode}")
    print(f" amount         : {amount_str} USDC  ← {'★PROD 5 USDC' if prod else 'test 1 USDC'}")
    print(f" recipient(Arb) : {ARB_SENDER_ADDRESS.lower()}")
    print()

    # 1. sourceDex 確定（実残高所在）
    print("[1/4] sourceDex 確定（info API 実残高所在）...")
    sd = determine_source_dex(amount)
    print(f"   spot available : {sd['spot_available']} USDC")
    print(f"   perp withdrawable: {sd['perp_withdrawable']} USDC")
    print(f"   => sourceDex   : '{sd['source_dex']}'  ({sd['reason']})")
    print()

    # 2. action 構築 + 署名 + 自己 recover 検証
    print("[2/4] sendToEvmWithData 構築・EIP-712署名・自己recover検証 ...")
    nonce_ms = int(time.time() * 1000)
    action = build_send_to_evm_action(amount_str, sd["source_dex"], nonce_ms)
    sig = sign_send_to_evm(action)
    recovered = verify_signature_self(action, sig)
    ok_sig = (recovered == ARB_SENDER_ADDRESS)
    print(f"   署名 r={sig['r'][:14]}... v={sig['v']}")
    print(f"   recover        : {recovered}  {'OK(=sender)' if ok_sig else '✗ NG 不一致！'}")
    if not ok_sig:
        print("   [ABORT判定] 署名の自己検証に失敗。実装を確認するまで送信不可。")
    print()

    # 3. RPC 接続性
    print("[3/4] RPC/WS 接続性 ...")
    try:
        w3 = make_arb_web3()
        usdc = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), abi=ERC20_ABI)
        bal = usdc.functions.balanceOf(ARB_SENDER_ADDRESS).call()
        print(f"   Arbitrum       : connected={w3.is_connected()} chainId={w3.eth.chain_id} "
              f"USDC残高={Decimal(bal)/Decimal(10**6)}")
    except Exception as e:
        print(f"   [WARN] Arbitrum 接続失敗: {e}")
    try:
        hl = make_hl_web3()
        print(f"   HyperEVM(公開)  : connected={hl.is_connected()} block={hl.eth.block_number}")
    except Exception as e:
        print(f"   [WARN] HyperEVM 接続失敗: {e}")
    # 第2層 burn 捕捉用のアーカイブ/WS RPC の状態
    archive_set = (cfg.HL_EVM_RPC_ARCHIVE != cfg.HL_EVM_RPC_URL)
    try:
        aw = make_hevm_archive_web3()
        print(f"   HyperEVM(捕捉)  : archive_RPC_set={archive_set} connected={aw.is_connected()} block={aw.eth.block_number}")
    except Exception as e:
        print(f"   [WARN] 捕捉用RPC 接続失敗: {e}")
    print(f"   burn捕捉方式    : {'WS(eth_subscribe)' if cfg.HL_EVM_WS_URL else 'HTTP小範囲poll'}"
          f"{'' if (archive_set or cfg.HL_EVM_WS_URL) else '  ⚠第三者RPC未設定→第2層(iris_wait)成功率低'}")
    print(f"   HyperCore WS   : {'OK' if test_hc_ws(5) else 'NG/timeout'}")
    print()

    # 4. 送信予定 action 要約
    print("[4/4] 送信予定 action 要約")
    print("-" * 70)
    print(f"   POST           : {cfg.HL_EXCHANGE_URL}")
    print(f"   primaryType    : {cfg.SEND_TO_EVM_PRIMARY_TYPE}")
    print(f"   domain         : name={cfg.EIP712_DOMAIN_NAME} version={cfg.EIP712_DOMAIN_VERSION} "
          f"chainId={cfg.SIGNATURE_CHAIN_ID_INT} verifyingContract={cfg.EIP712_VERIFYING_CONTRACT}")
    for k in ("type", "hyperliquidChain", "signatureChainId", "token", "amount", "sourceDex",
              "destinationRecipient", "addressEncoding", "destinationChainId", "gasLimit", "data", "nonce"):
        print(f"   action.{k:<20}: {action[k]}")
    print("-" * 70)
    print(f" mode={mode}  amount={amount_str} USDC  sourceDex='{sd['source_dex']}'  署名自己検証={'OK' if ok_sig else 'NG'}")
    print(" DRY RUN 完了。送信は行っていません。")
    print("=" * 70)

# =====================================================================
# 本番計測（--broadcast 時のみ）
# =====================================================================
def run_broadcast(exp_id_override: Optional[int] = None, prod: bool = False):
    csv_path = cfg.PROD_CSV if prod else cfg.TEST_CSV
    out_dir = cfg.PROD_DIR if prod else cfg.TEST_DIR
    amount_str = fmt_amount(cfg.WITHDRAW_AMOUNT_PROD if prod else cfg.WITHDRAW_AMOUNT_TEST)
    amount = Decimal(amount_str)
    exp_id = exp_id_override if exp_id_override is not None else get_next_experiment_id(csv_path)
    exp_start_ms = int(time.time() * 1000)
    mode = "PROD" if prod else "TEST"
    print(f"===== T4 Withdraw CCTP Fast (BROADCAST/{mode}) exp_id={exp_id} =====")
    print(f"[out] csv={csv_path}  intermediate_dir={out_dir}")
    print(f"[amount] {amount_str} USDC  ({'★PROD' if prod else 'test'})")

    # 0. sourceDex 確定 + action 構築 + 署名 + 自己 recover（送信前の最終ガード）
    sd = determine_source_dex(amount)
    print(f"[sourceDex] '{sd['source_dex']}'  ({sd['reason']})")
    nonce_ms = int(time.time() * 1000)
    action = build_send_to_evm_action(amount_str, sd["source_dex"], nonce_ms)
    assert action["destinationRecipient"] == ARB_SENDER_ADDRESS.lower(), "[ABORT] recipient != self"
    sig = sign_send_to_evm(action)
    recovered = verify_signature_self(action, sig)
    if recovered != ARB_SENDER_ADDRESS:
        print(f"[ABORT] 署名自己検証失敗 recovered={recovered} != {ARB_SENDER_ADDRESS}。送信中止。")
        sys.exit(1)
    print(f"[sign] 自己recover OK (= {recovered})")

    # 1. 生 WS ログ + 引落 listener を送信前に起動
    raw_path = out_dir / f"hc_ws_raw_{exp_id}.jsonl"
    raw_logger = HcRawWsLogger(ARB_SENDER_ADDRESS, raw_path)
    raw_logger.start()
    debit_listener = HcDebitListener(ARB_SENDER_ADDRESS, exp_start_ms, amount)
    debit_listener.start()
    time.sleep(2)

    # 1b. 送信直前の起点ブロックを記録 + 第2層 burn ライブ捕捉ワーカーを起動
    arb_w3 = make_arb_web3()
    arb_start_block = max(0, arb_w3.eth.block_number - 2)
    gross_raw = int(amount * Decimal(10**6))
    try:
        hevm_start_block = max(0, make_hevm_archive_web3().eth.block_number - 2)
    except Exception as e:
        print(f"[hevm-burn] 起点ブロック取得失敗: {e}")
        hevm_start_block = 0
    if cfg.HL_EVM_RPC_ARCHIVE == cfg.HL_EVM_RPC_URL and not cfg.HL_EVM_WS_URL:
        print("[hevm-burn] ⚠ 第三者アーカイブ/WS RPC 未設定（公開RPCで捕捉試行・成功率低）。"
              "第2層(iris_wait)安定取得には .env に HL_EVM_RPC_ARCHIVE / HL_EVM_WS_URL を推奨。")
    burn_worker = HevmBurnWatcher(hevm_start_block, gross_raw, cfg.HEVM_BURN_TIMEOUT_SEC)
    burn_worker.start()  # 送信前から捕捉開始

    # 2. 送信（t0 = local_send_ns）
    print(f"[send] POST sendToEvmWithData nonce={nonce_ms} ...")
    send = post_send_to_evm(action, sig)
    print(f"[send] http={send.http_status} ok={send.ok} response={send.response}")
    write_send_checkpoint(exp_id, send, out_dir)
    if not send.ok:
        print("[ERROR] 送信が status=ok になりませんでした。検知フェーズに進まず終了。")
        burn_worker.stop()
        row = build_row(exp_id, amount_str, send, debit_listener.result, BurnResult(False),
                        None, None, MintResult(False), MintReceiptInfo(), 0, "")
        save_csv(row, csv_path)
        raw_logger.stop(); debit_listener.stop()
        sys.exit(1)

    # 3. 検知フェーズ（crash-safe: finally で取得済み値だけ必ず保存）
    debit = debit_listener.result
    burn = BurnResult(False); iris_live = None; iris_meta = None
    mint = MintResult(False); mint_info = MintReceiptInfo()
    hevm_height_at_attest = 0; forward_state = ""
    try:
        # 3a. HyperCore 引落（t_hc_debit）
        debit = debit_listener.wait(timeout_sec=cfg.HC_DEBIT_TIMEOUT_SEC)
        if debit.found:
            print(f"[hc-debit] t={debit.ledger_time_ms} type={debit.delta_type} amount={debit.amount}")

        # 3b. Arbitrum mint（t3）— burn 捕捉はワーカーが並行実行中
        print("[arb-mint] mint 検知 ...")
        mint = watch_arb_mint(arb_w3, arb_start_block, cfg.ARB_MINT_TIMEOUT_SEC)
        if mint.found:
            print(f"[arb-mint] tx={mint.tx_hash} block={mint.block_number} "
                  f"received={Decimal(mint.amount_raw)/Decimal(10**6)} USDC")
            # 3c. 第1層: mint tx の MessageReceived から nonce / finalityThresholdExecuted（確実コア）
            mint_info = parse_mint_receipt(arb_w3, mint.tx_hash)
            print(f"[mint-receipt] nonce={mint_info.cctp_nonce} "
                  f"finalityThresholdExecuted={mint_info.finality_threshold_executed} "
                  f"sourceDomain={mint_info.source_domain}")

        # 3d. 第2層 burn ワーカーの完了待ち（捕捉済みなら iris_live も入る）
        print("[hevm-burn] ワーカー join 待ち ...")
        burn_worker.join(timeout=cfg.HEVM_BURN_TIMEOUT_SEC)
        if burn_worker.is_alive():
            burn_worker.stop(); burn_worker.join(timeout=10)
        burn = burn_worker.burn; iris_live = burn_worker.iris_live
        hevm_height_at_attest = burn_worker.hevm_height_at_attest
        if burn.found:
            print(f"[hevm-burn] 確定 t1={burn.block_ts_ms} block={burn.block_number} tx={burn.tx_hash} "
                  f"hevm_conf@attest={(hevm_height_at_attest-burn.block_number) if hevm_height_at_attest else '?'}")
        else:
            print("[hevm-burn] ライブ捕捉できず（第2層=iris_wait欠落・第1層で継続）")

        # 3e. 第1層メタデータ補完（iris_live が無いとき nonce で Iris 逆引き）
        if iris_live is None and mint_info.cctp_nonce:
            print("[iris-nonce] nonce逆引きでメタデータ補完 ...")
            iris_meta = fetch_iris_by_nonce(mint_info.cctp_nonce, timeout_sec=60)

        if iris_live and iris_live.forward_state:
            forward_state = iris_live.forward_state
        elif iris_meta and iris_meta.forward_state:
            forward_state = iris_meta.forward_state
        elif burn.found:
            forward_state = fetch_forward_state(burn.tx_hash) or ""
        print(f"[iris] forwardState = {forward_state}")
    finally:
        forward_state = locals().get("forward_state", "") or ""
        row = build_row(exp_id, amount_str, send, debit, burn, iris_live, iris_meta, mint,
                        mint_info, hevm_height_at_attest, forward_state)
        save_csv(row, csv_path)
        print("===== 計測結果（finally 保存）=====")
        for k in CSV_HEADER:
            print(f"  {k} = {row[k]}")
        # 検算（第2層成立時）: hypercore_to_burn + iris_wait + attestation_to_mint == E2E_wit_hypercore
        h2b, iw, a2m, e2ehc = (row["hypercore_to_burn(ms)"], row["iris_wait(ms)"],
                               row["attestation_to_mint(ms)"], row["E2E_wit_hypercore(ms)"])
        if all(isinstance(x, int) for x in (h2b, iw, a2m, e2ehc)):
            s = h2b + iw + a2m
            print(f"[検算] {h2b}+{iw}+{a2m}={s} vs E2E_wit_hypercore={e2ehc} "
                  f"=> {'一致' if s == e2ehc else '不一致'}")
        print(f"Saved to {csv_path}")
        try:
            burn_worker.stop()
        except Exception:
            pass
        raw_logger.stop()
        debit_listener.stop()
        print(f"[hc-raw] logger stopped. messages logged = {raw_logger.msg_count}, file = {raw_path}")

# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="T4 Withdraw CCTP Fast measurement (HyperCore → Arbitrum)")
    ap.add_argument("--broadcast", action="store_true", help="実送信して計測する（未指定はドライラン）")
    ap.add_argument("--prod", action="store_true", help="本番モード（5 USDC・cfg.PROD_CSV 連番追記）。未指定は test 1 USDC。")
    ap.add_argument("--exp-id", type=int, default=None, help="experiment_id を明示指定（未指定は自動採番）")
    args = ap.parse_args()
    if args.broadcast:
        run_broadcast(exp_id_override=args.exp_id, prod=args.prod)
    else:
        run_dry_run(prod=args.prod)

if __name__ == "__main__":
    main()
