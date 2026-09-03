"""
t4_cctp/config.py — T4（CCTP実測）用の定数・アドレス定義。

Phase A（一次ソース調査）で確定した値を反映。確証レベルをコメントで明示する。
未確定/要最終確認の項目は実送信(Phase B)の前に必ず再確認すること。
設計の正本は T4_CCTP_measurement_spec_v2.md（別途共有）を参照。
"""

import os

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- パス ---
# このファイルは t4_cctp/deposit/config.py に置かれる前提。
#   __file__.parent           = t4_cctp/deposit
#   __file__.parent.parent    = t4_cctp
#   __file__.parent.parent.parent = リポジトリルート
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = REPO_ROOT / "result" / "T4_cctp"

# 中間/出力の置き場（D 整理後）
TEST_DIR = RESULT_DIR / "test" / "deposit"           # test の中間(raw/checkpoint/log)＋test CSV
PROD_DIR = RESULT_DIR / "prod"                        # 本番の中間(raw/checkpoint/log)
TEST_CSV = TEST_DIR / "deposit_cctp_test.csv"         # test 用 CSV
PROD_CSV = RESULT_DIR / "deposit_cctp_latency.csv"    # 本番 最終 CSV（1ファイルに追記・T1運用踏襲）

# --- CCTP domain ID（Circle公式 "CCTP Supported Blockchains" で CONFIRMED） ---
ARB_DOMAIN_ID = 3
HYPEREVM_DOMAIN_ID = 19

# =====================================================================
# Arbitrum 側（送信＝burn 側）
# =====================================================================

# ネイティブ USDC（CONFIRMED: Circle stablecoins doc + Arbiscan）
ARB_USDC_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

# TokenMessengerV2 = depositForBurn(WithHook) を呼ぶ「送信(burn)側」コントラクト
# CONFIRMED: Circle CCTP contract-addresses + Arbiscan "Circle CCTP: Token Messenger V2"
# （CCTP V2 は全EVMチェーンで同一アドレス。.env の ARB_CCTP_TOKEN_MESSENGER と一致）
ARB_TOKEN_MESSENGER_V2 = "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d"

# MessageTransmitterV2 = 受信(mint)側。送信には使わない（参考保持）
# CONFIRMED: Arbiscan "Circle CCTP: Message Transmitter V2"
ARB_MESSAGE_TRANSMITTER_V2 = "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64"

# =====================================================================
# HyperEVM 側（受信＝mint → forward 側）
# =====================================================================

# HyperEVM 上の USDC（CONFIRMED: Circle stablecoins doc + HyperEVMScan "Circle: USDC Token"）
HYPEREVM_USDC_ADDRESS = "0xb88339CB7199b77E23DB6E890353E22632Ba630f"

# CctpForwarder（mintRecipient に指定する HyperEVM 上の Forwarder プロキシ）
# CONFIRMED: HyperEVMScan で Circle Deployer 製・proxy 実装名="CctpForwarder"・
# mintAndForward(bytes,bytes) メソッド・196k+ txn を独立検証。Candidate(USDCと取り違え)を排除済み。
CCTP_FORWARDER_HEVM = "0xb21d281dEDB17ae5B501f6Aa8256Fe38c4e45757"  # proxy
CCTP_FORWARDER_HEVM_IMPL = "0x335828b6d3777ccDed12d626b01bF12c3C7CCb58"  # 参考: 実装

# CoreDepositWallet（Forwarder が mint 後に USDC を預けて HyperCore へ credit する先）
# CONFIRMED: Chainstack doc + HyperEVMScan（Circle Deployer 製・約$2B USDC 保有・deposit メソッド確認）
CORE_DEPOSIT_WALLET = "0x6b9e773128f453f5C2c60935Ee2De2cBC5390A24"

# 旧 Native Bridge（CCTP では使わない。誤用防止のため明示）
HL_DEPOSIT_BRIDGE_ADDRESS_LEGACY = "0x2df1c51e09aecf9cacb7bc98cb1742757f163df7"

# CctpExtension（Arbitrum 側の別経路。今回は depositForBurnWithHook 直呼びを採用するため未使用）
CCTP_EXTENSION_ARB = None  # 採用しない（理由は deposit_cctp_measure.py 冒頭 docstring 参照）

# =====================================================================
# CCTP V2 定数（evm-cctp-contracts FinalityThresholds.sol で CONFIRMED）
# =====================================================================
FINALITY_THRESHOLD_FAST = 1000      # FINALITY_THRESHOLD_CONFIRMED（Fast）
FINALITY_THRESHOLD_STANDARD = 2000  # FINALITY_THRESHOLD_FINALIZED（Standard）

# Deposit は Fast。minFinalityThreshold<=1000 を設定 → finalityThresholdExecuted=1000 を期待。
DEPOSIT_MIN_FINALITY_THRESHOLD = 1000

# Arbitrum→HyperCore Fast は CCTP 手数料ゼロ（Circle公式）。maxFee=0 / feeExecuted=0 を期待。
DEPOSIT_MAX_FEE = 0

# CctpForwarder hookData の destinationId（CoreDepositWallet.sol 定数）
HC_DEST_PERP = 0           # perp 残高
HC_DEST_SPOT = 0xFFFFFFFF  # spot 残高
DEPOSIT_HC_DESTINATION_ID = HC_DEST_PERP  # 既定は perp

# CctpForwarder hookData マジック（CctpForwarderHookData.sol。生バイト連結=encodePacked）
HOOK_MAGIC = b"cctp-forward"  # 12 bytes → 右0詰めで 24 bytes に拡張して使用

# --- API ---
IRIS_API_HOST = "https://iris-api.circle.com"
IRIS_MESSAGES_PATH = "/v2/messages/{source_domain}"  # ?transactionHash={burnTx}
# fee API（CONFIRMED: src=3/dst=19 で HTTP200。developers.circle.com cctp-finality-and-fees）
# レスポンス例: [{"finalityThreshold":1000,"minimumFee":1.3},{"finalityThreshold":2000,"minimumFee":0}]
# minimumFee の単位は basis points(bps)。maxFee(atomic) = ceil(amount_atomic * bps / 10000)。
IRIS_FEE_PATH = "/v2/burn/USDC/fees/{source_domain}/{dest_domain}"
# attestation 完了を物理時刻(±poll間隔精度)で捉えるため最大レートで poll する。
# 0.25s = 4 req/s（Iris 制限 2〜4 req/s の上限）。これが t2(信頼層) 測定の精度を決める。
IRIS_POLL_INTERVAL_SEC = 0.25  # 4 req/s（上限）

# --- RPC（.env を参照、既定値は T1 と同じ） ---
ARB_RPC_URL = os.getenv("ARB_RPC_URL", os.getenv("ARBITRUM_HTTP_RPC", "https://arb1.arbitrum.io/rpc"))
HL_EVM_RPC_URL = os.getenv("HL_EVM_RPC_URL", "https://rpc.hyperliquid.xyz/evm")
HL_WS_URL = "wss://api.hyperliquid.xyz/ws"

# --- 送金額（既定 5.0 USDC） ---
DEPOSIT_AMOUNT_USDC = os.getenv("ARB_CCTP_AMOUNT_USDC", "5.0")
