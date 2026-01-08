from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3
from utils import now_ns, ns_to_iso
import os
import sys
import csv
from decimal import Decimal

# ==========================
# env 読み込み
# ==========================

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

# RPC
ARBITRUM_HTTP_RPC = os.getenv("ARBITRUM_HTTP_RPC", "").strip()

# 送信元アドレス・秘密鍵（ここが今回のポイント）
ARB_SENDER_ADDRESS = (os.getenv("ARB_SENDER_ADDRESS") or os.getenv("HL_USER_ADDRESS") or "").strip()
ARB_SENDER_PRIVATE_KEY = os.getenv("ARB_SENDER_PRIVATE_KEY", "").strip()

# CCTP 関連
ARB_CCTP_DEST_DOMAIN_STR = os.getenv("ARB_CCTP_DEST_DOMAIN", "0").strip()        # ここは 19 にしておく
ARB_CCTP_AMOUNT_USDC_STR = os.getenv("ARB_CCTP_AMOUNT_USDC", "5.0").strip()
ARB_CCTP_DRY_RUN = os.getenv("ARB_CCTP_DRY_RUN", "true").lower() in ("1", "true", "yes")

# アドレス
ARB_USDC_ADDRESS = Web3.to_checksum_address(
    os.getenv("ARB_USDC_ADDRESS", "0xaf88d065e77c8C2239327C5EdcEaF4507BBcE01")
)
ARB_TOKEN_MESSENGER_ADDRESS = Web3.to_checksum_address(
    os.getenv("ARB_TOKEN_MESSENGER", "0xec546b6B5d7BC121d0875c847E0a93b0aE3bEc5e")
)

# HyperEVM 側の mint recipient
MINT_RECIPIENT_ADDRESS = (
    os.getenv("ARB_CCTP_MINT_RECIPIENT")
    or os.getenv("HL_USER_ADDRESS")
    or ARB_SENDER_ADDRESS
    or ""
).strip()

USDC_DECIMALS = int(os.getenv("ARB_USDC_DECIMALS", "6"))

# env フラグを使った DRY_RUN
DRY_RUN = ARB_CCTP_DRY_RUN

# ==========================
# 必要最小限の ABI
# ==========================

USDC_ABI = [
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

TOKEN_MESSENGER_ABI = [
    {
        "name": "depositForBurn",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amount", "type": "uint256"},
            {"name": "destinationDomain", "type": "uint32"},
            {"name": "mintRecipient", "type": "bytes32"},
            {"name": "burnToken", "type": "address"},
        ],
        "outputs": [],
    }
]


# ==========================
# Utility
# ==========================

def to_checksum_or_die(label: str, addr: str) -> str:
    if not addr:
        print(f"[ERROR] {label} が空です。env を確認してください。", file=sys.stderr)
        sys.exit(1)
    try:
        return Web3.to_checksum_address(addr)
    except Exception:
        print(f"[ERROR] {label} の形式が不正です: {addr}", file=sys.stderr)
        sys.exit(1)


def address_to_bytes32(addr: str) -> bytes:
    """Solidity の address → bytes32 (左ゼロ詰め) と同じ変換。"""
    caddr = Web3.to_checksum_address(addr)
    return b"\x00" * 12 + bytes.fromhex(caddr[2:])


def append_csv_row(csv_path: Path, row: list, header: list):
    csv_path.parent.mkdir(exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


# ==========================
# Main
# ==========================

def main():
    # --- env チェック ---
    if not ARBITRUM_HTTP_RPC:
        print("[ERROR] ARBITRUM_HTTP_RPC が .env に設定されていません。", file=sys.stderr)
        sys.exit(1)
    if not ARB_SENDER_ADDRESS:
        print("[ERROR] ARB_SENDER_ADDRESS または HL_USER_ADDRESS が .env に設定されていません。", file=sys.stderr)
        sys.exit(1)
    if not ARB_SENDER_PRIVATE_KEY:
        print("[ERROR] ARB_SENDER_PRIVATE_KEY が .env に設定されていません。", file=sys.stderr)
        sys.exit(1)

    sender = to_checksum_or_die("ARB_SENDER_ADDRESS", ARB_SENDER_ADDRESS)
    mint_recipient_addr = to_checksum_or_die("MINT_RECIPIENT_ADDRESS", MINT_RECIPIENT_ADDRESS)

    # destDomain
    try:
        dest_domain = int(ARB_CCTP_DEST_DOMAIN_STR)
    except ValueError:
        print(f"[ERROR] ARB_CCTP_DEST_DOMAIN が整数として解釈できません: {ARB_CCTP_DEST_DOMAIN_STR}", file=sys.stderr)
        sys.exit(1)

    # amount
    try:
        amount_usdc = Decimal(ARB_CCTP_AMOUNT_USDC_STR)
    except Exception:
        print(f"[ERROR] ARB_CCTP_AMOUNT_USDC が数値として解釈できません: {ARB_CCTP_AMOUNT_USDC_STR}", file=sys.stderr)
        sys.exit(1)

    amount_raw = int(amount_usdc * (10 ** USDC_DECIMALS))

    # Web3 接続
    w3 = Web3(Web3.HTTPProvider(ARBITRUM_HTTP_RPC))
    if not w3.is_connected():
        print(f"[ERROR] Arbitrum RPC に接続できません: {ARBITRUM_HTTP_RPC}", file=sys.stderr)
        sys.exit(1)

    chain_id = w3.eth.chain_id
    print(f"Connecting to Arbitrum RPC: {ARBITRUM_HTTP_RPC}")
    print(f"Connected. chainId = {chain_id}")
    print()
    print("===== 環境確認 =====")
    print(f"  sender_address      : {sender}")
    print(f"  dest_domain (env)   : {dest_domain}")
    print(f"  mint_recipient_addr : {mint_recipient_addr}")
    print(f"  DRY_RUN             : {DRY_RUN}")
    print()

    if dest_domain == 0:
        print("[WARNING] ARB_CCTP_DEST_DOMAIN が 0 のままです。HyperEVM に送る場合は 19 にしてください。")
        print("          (.env の ARB_CCTP_DEST_DOMAIN=19 を必ず設定してから実送信してください。)")
        print()

    # ローカル送信時刻（t_send 計測用）
    t_ns = now_ns()
    t_iso = ns_to_iso(t_ns)
    print("===== CCTP burn 計画 =====")
    print(f"  local_send_time_ns : {t_ns}")
    print(f"  local_send_time_iso: {t_iso}")
    print(f"  sender_address     : {sender}")
    print(f"  dest_domain        : {dest_domain}")
    print(f"  amount_usdc        : {amount_usdc}")
    print(f"  amount_raw         : {amount_raw}")
    print(f"  usdc_address       : {ARB_USDC_ADDRESS}")
    print(f"  token_messenger    : {ARB_TOKEN_MESSENGER_ADDRESS}")
    print(f"  mint_recipient     : {mint_recipient_addr}")
    print()

    if DRY_RUN:
        print("[INFO] 現在 DRY_RUN = True のため、ここから先は on-chain 送信しません。")
        print("       内容を確認して問題なければ、.env の ARB_CCTP_DRY_RUN を false に変更して再実行してください。")
        return

    # --- コントラクトインスタンス ---
    usdc = w3.eth.contract(address=ARB_USDC_ADDRESS, abi=USDC_ABI)
    messenger = w3.eth.contract(address=ARB_TOKEN_MESSENGER_ADDRESS, abi=TOKEN_MESSENGER_ABI)

    # 残高チェック
    balance = usdc.functions.balanceOf(sender).call()
    if balance < amount_raw:
        print(f"[ERROR] USDC 残高不足: balance={balance}, required={amount_raw}", file=sys.stderr)
        sys.exit(1)

    # allowance 確認
    allowance = usdc.functions.allowance(sender, ARB_TOKEN_MESSENGER_ADDRESS).call()
    nonce = w3.eth.get_transaction_count(sender)
    gas_price = w3.eth.gas_price

    approve_tx_hash_hex = ""
    approve_gas_used = 0

    if allowance < amount_raw:
        print(f"[INFO] allowance 不足: {allowance} < {amount_raw} -> approve を送信します。")
        approve_tx = usdc.functions.approve(
            ARB_TOKEN_MESSENGER_ADDRESS, amount_raw
        ).build_transaction(
            {
                "from": sender,
                "nonce": nonce,
                "gasPrice": gas_price,
                "chainId": chain_id,
            }
        )
        approve_tx["gas"] = w3.eth.estimate_gas(approve_tx)

        signed_approve = w3.eth.account.sign_transaction(approve_tx, private_key=ARB_SENDER_PRIVATE_KEY)
        approve_tx_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
        approve_tx_hash_hex = approve_tx_hash.hex()
        print(f"[approve] tx_hash = {approve_tx_hash_hex}")
        receipt = w3.eth.wait_for_transaction_receipt(approve_tx_hash)
        approve_gas_used = receipt.gasUsed
        print(f"[approve] gasUsed = {approve_gas_used}")
        nonce += 1
    else:
        print(f"[INFO] allowance OK: {allowance} >= {amount_raw} -> approve スキップ")

    # depositForBurn
    mint_recipient_bytes32 = address_to_bytes32(mint_recipient_addr)

    print("===== depositForBurn トランザクション作成 =====")
    burn_tx = messenger.functions.depositForBurn(
        amount_raw,
        dest_domain,
        mint_recipient_bytes32,
        ARB_USDC_ADDRESS,
    ).build_transaction(
        {
            "from": sender,
            "nonce": nonce,
            "gasPrice": gas_price,
            "chainId": chain_id,
        }
    )
    burn_tx["gas"] = w3.eth.estimate_gas(burn_tx)

    signed_burn = w3.eth.account.sign_transaction(burn_tx, private_key=ARB_SENDER_PRIVATE_KEY)
    burn_tx_hash = w3.eth.send_raw_transaction(signed_burn.rawTransaction)
    burn_tx_hash_hex = burn_tx_hash.hex()
    print(f"[burn] tx_hash = {burn_tx_hash_hex}")

    receipt = w3.eth.wait_for_transaction_receipt(burn_tx_hash)
    burn_gas_used = receipt.gasUsed
    burn_block_number = receipt.blockNumber
    block = w3.eth.get_block(burn_block_number)
    burn_block_ts = int(block["timestamp"])
    burn_block_ts_iso = ns_to_iso(burn_block_ts * 10**9)

    print(f"[burn] blockNumber = {burn_block_number}")
    print(f"[burn] gasUsed     = {burn_gas_used}")
    print(f"[burn] gasPriceWei = {gas_price}")
    print(f"[burn] blockTime   = {burn_block_ts} ({burn_block_ts_iso})")
    print()

    # CSV ログ
    log_path = BASE_DIR / "results" / "arb_cctp_burn_log.csv"
    header = [
        "local_send_time_ns",
        "local_send_time_iso",
        "sender_address",
        "amount_usdc",
        "amount_raw",
        "dest_domain",
        "mint_recipient",
        "approve_tx_hash",
        "approve_gas_used",
        "burn_tx_hash",
        "burn_block_number",
        "burn_gas_used",
        "burn_gas_price_wei",
        "burn_block_timestamp",
        "burn_block_timestamp_iso",
    ]
    row = [
        t_ns,
        t_iso,
        sender,
        str(amount_usdc),
        amount_raw,
        dest_domain,
        mint_recipient_addr,
        approve_tx_hash_hex,
        approve_gas_used,
        burn_tx_hash_hex,
        burn_block_number,
        burn_gas_used,
        int(gas_price),
        burn_block_ts,
        burn_block_ts_iso,
    ]
    append_csv_row(log_path, row, header)
    print(f"[INFO] 1 row appended to {log_path}")


if __name__ == "__main__":
    main()
