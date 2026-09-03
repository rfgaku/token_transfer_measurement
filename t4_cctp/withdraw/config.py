"""
t4_cctp/withdraw/config.py — T4（CCTP実測）withdraw 側（HyperCore → Arbitrum）の定数・パス定義。

共通定数（アドレス/domain/Iris/RPC）は deposit 側 config をそのまま流用し、
withdraw 固有（金額・出力先・送金 action 定数）のみ追加する。

■ withdraw 方式（Circle公式 "Withdraw USDC from HyperCore to EVM chains" で確定）
  HyperCore へ単一の EIP-712 署名アクション sendToEvmWithData を POST すると、
  HyperCore 引落 → HyperEVM での CCTP burn → Iris attestation → Arbitrum で自動 mint
  （data="0x" による自動 forwarding）まで完結する。
  ■実測事実（test exp_id=2 / 2026-06-04）: finalityThresholdExecuted=2000（Standard/Finalized）で
    実行される（公式表現の "Fast default" とは異なる）。HyperBFT の高速 finality により Finalized でも
    数秒で完結（実測 E2E≈8s, iris_wait≈7.2s）。maxFee≈0.2 USDC（=forwarding 満額徴収・feeExecuted=200000）、
    着金 = amount − 0.2 USDC（額に非比例の固定 forwarding 手数料）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- deposit 側 config の共通定数を流用（アドレス/domain/Iris/RPC/finality threshold） ---
from t4_cctp.deposit import config as dep

load_dotenv()

# =====================================================================
# パス（このファイルは t4_cctp/withdraw/config.py に置かれる前提）
#   __file__.parents[0] = t4_cctp/withdraw
#   __file__.parents[1] = t4_cctp
#   __file__.parents[2] = リポジトリルート
# =====================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "result" / "T4_cctp"

TEST_DIR = RESULT_DIR / "test" / "withdraw"               # test 中間(raw/checkpoint/log)＋test CSV
PROD_DIR = RESULT_DIR / "prod"                            # 本番中間(raw/checkpoint/log)
TEST_CSV = TEST_DIR / "withdraw_cctp_test.csv"            # test 用 CSV
PROD_CSV = RESULT_DIR / "withdraw_cctp_latency.csv"       # 本番 最終 CSV（1ファイル連番追記）

# =====================================================================
# 共通定数（deposit config から流用・再エクスポート）
# =====================================================================
# CCTP domain ID
ARB_DOMAIN_ID = dep.ARB_DOMAIN_ID                 # 3  (Arbitrum)
HYPEREVM_DOMAIN_ID = dep.HYPEREVM_DOMAIN_ID        # 19 (HyperEVM)

# アドレス（CCTP V2 は MessageTransmitterV2 / TokenMessengerV2 が全 EVM チェーン共通アドレス）
ARB_USDC_ADDRESS = dep.ARB_USDC_ADDRESS                       # 0xaf88...5831
HYPEREVM_USDC_ADDRESS = dep.HYPEREVM_USDC_ADDRESS             # 0xb883...630f
# MessageTransmitterV2 = MessageSent(bytes) の発行元。withdraw では HyperEVM 上の burn を検知する。
HEVM_MESSAGE_TRANSMITTER_V2 = dep.ARB_MESSAGE_TRANSMITTER_V2  # 0x81D4...4B64（全 EVM 共通）
ARB_MESSAGE_TRANSMITTER_V2 = dep.ARB_MESSAGE_TRANSMITTER_V2   # 同上（Arbitrum 側の receiveMessage 用・回収時）
ARB_TOKEN_MESSENGER_V2 = dep.ARB_TOKEN_MESSENGER_V2          # 0x28b5...cf5d（全 EVM 共通）
# CoreDepositWallet = HyperEVM 上で withdraw 時の burn 実行主体
CORE_DEPOSIT_WALLET = dep.CORE_DEPOSIT_WALLET                 # 0x6B9E...0A24

# CCTP finality threshold
FINALITY_THRESHOLD_FAST = dep.FINALITY_THRESHOLD_FAST          # 1000
FINALITY_THRESHOLD_STANDARD = dep.FINALITY_THRESHOLD_STANDARD  # 2000

# --- API / RPC（deposit config 流用） ---
IRIS_API_HOST = dep.IRIS_API_HOST                  # https://iris-api.circle.com
IRIS_MESSAGES_PATH = dep.IRIS_MESSAGES_PATH        # /v2/messages/{source_domain}?transactionHash=...
IRIS_POLL_INTERVAL_SEC = dep.IRIS_POLL_INTERVAL_SEC  # 0.25s

ARB_RPC_URL = dep.ARB_RPC_URL
HL_EVM_RPC_URL = dep.HL_EVM_RPC_URL                # https://rpc.hyperliquid.xyz/evm（公開・非アーカイブ・100req/min・WS非対応）
HL_WS_URL = dep.HL_WS_URL                          # wss://api.hyperliquid.xyz/ws（HyperCore WS）

# ★第2層(iris_wait)用: HyperEVM の CCTP burn をライブ捕捉するための第三者 HyperEVM RPC/WS。
#   公開 RPC は非アーカイブ・100req/min・WS非対応で burn 捕捉に失敗するため、
#   .env に高レート/アーカイブ/WS対応の第三者エンドポイント（Quicknode nanoreth / Alchemy /
#   Dwellir / Chainstack 等の無料枠）を設定することを強く推奨。未設定なら公開 RPC にフォールバック。
HL_EVM_RPC_ARCHIVE = os.getenv("HL_EVM_RPC_ARCHIVE", dep.HL_EVM_RPC_URL)
HL_EVM_WS_URL = os.getenv("HL_EVM_WS_URL", "")     # 例: wss://...（eth_subscribe(logs) 対応）。空なら HTTP poll のみ

# =====================================================================
# withdraw 固有定数
# =====================================================================
HL_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# 金額（★安全策: test は必ず 1 USDC、prod のみ 5 USDC）
WITHDRAW_AMOUNT_TEST = os.getenv("HL_CCTP_WITHDRAW_AMOUNT_TEST", "1.0")
WITHDRAW_AMOUNT_PROD = os.getenv("HL_CCTP_WITHDRAW_AMOUNT_PROD", "5.0")

# sendToEvmWithData action 定数（Circle公式 "Withdraw USDC from HyperCore to EVM"）
HYPERLIQUID_CHAIN = "Mainnet"
SIGNATURE_CHAIN_ID = "0xa4b1"        # Arbitrum EVM chainId(hex)=42161。署名 domain.chainId に使用
SIGNATURE_CHAIN_ID_INT = 42161
WITHDRAW_TOKEN = "USDC"
ADDRESS_ENCODING = "hex"
DEST_CHAIN_ID_CCTP_DOMAIN = ARB_DOMAIN_ID            # 3 = Arbitrum CCTP domain（action.destinationChainId）
GAS_LIMIT = 200000
DATA_HEX = "0x"                                      # "0x" で Arbitrum 宛 自動 forwarding 有効
# sourceDex は送信前に info API で実残高所在を確認して設定（spot→"spot" / perp→""）。
# md 記載では spot 蓄積のはずだが実値で確認する（既定値はあくまでフォールバック）。
DEFAULT_SOURCE_DEX = "spot"

# EIP-712 署名（domain / primaryType / types）— Circle公式 + hyperliquid SDK 規約で確定
EIP712_DOMAIN_NAME = "HyperliquidSignTransaction"
EIP712_DOMAIN_VERSION = "1"
EIP712_VERIFYING_CONTRACT = "0x0000000000000000000000000000000000000000"
SEND_TO_EVM_PRIMARY_TYPE = "HyperliquidTransaction:SendToEvmWithData"
# types（順序厳守。signatureChainId は action のみで、署名 types には含めない）
SEND_TO_EVM_SIGN_TYPES = [
    {"name": "hyperliquidChain", "type": "string"},
    {"name": "token", "type": "string"},
    {"name": "amount", "type": "string"},
    {"name": "sourceDex", "type": "string"},
    {"name": "destinationRecipient", "type": "string"},
    {"name": "addressEncoding", "type": "string"},
    {"name": "destinationChainId", "type": "uint32"},
    {"name": "gasLimit", "type": "uint64"},
    {"name": "data", "type": "bytes"},
    {"name": "nonce", "type": "uint64"},
]

# Forwarding 手数料（Circle forwarding-service: Arbitrum 宛 0.2 USDC 固定）。
# 自動 forwarding 時は amount から差し引かれて着金する。Iris/実差分で確定できればそちらを優先。
FORWARDING_FEE_FALLBACK_ATOMIC = 200_000  # 0.2 USDC (6 decimals)

# タイムアウト（§4）
IRIS_TIMEOUT_SEC = 300
HEVM_BURN_TIMEOUT_SEC = 180
ARB_MINT_TIMEOUT_SEC = 600
HC_DEBIT_TIMEOUT_SEC = 120
