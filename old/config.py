from pathlib import Path
import os

from dotenv import load_dotenv

# このファイルがあるディレクトリ = プロジェクトルート想定
BASE_DIR = Path(__file__).resolve().parent

# 同じディレクトリの .env を読む
load_dotenv(BASE_DIR / ".env")

# Arbitrum RPC / アカウント
ARBITRUM_HTTP_RPC = os.getenv("ARBITRUM_HTTP_RPC", "")
ARBITRUM_WS_RPC = os.getenv("ARBITRUM_WS_RPC", "")
ARB_EOA_ADDRESS = os.getenv("ARB_EOA_ADDRESS", "")
ARB_PRIVATE_KEY = os.getenv("ARB_PRIVATE_KEY", "")

# Hyperliquid 側で使うウォレットアドレス
HL_USER_ADDRESS = os.getenv("HL_USER_ADDRESS", "")

# 結果保存用ディレクトリ
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
