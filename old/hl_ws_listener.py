# hl_ws_listener.py
#
# Hyperliquid userNonFundingLedgerUpdates を購読して、
# deposit / withdraw / internalTransfer を CSV にロギングするスクリプト

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from hyperliquid.info import Info
from hyperliquid.utils import constants

from config import BASE_DIR, HL_USER_ADDRESS
from utils import now_ns, ns_to_iso

# ログファイルのパス
LOG_PATH = BASE_DIR / "results" / "hl_nonfunding_log.csv"


def _ensure_logfile() -> None:
    """CSV がなければヘッダー付きで作成しておく。"""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "local_ns",
                    "local_iso",
                    "ledger_time_ms",
                    "ledger_time_iso",
                    "type",
                    "usdc",
                    "fee",
                    "vault",
                    "user",
                    "destination",
                    "tx_hash",
                ]
            )


def _ms_to_iso(ms: int) -> str:
    """ミリ秒 UNIX time を UTC ISO 文字列に変換。"""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def handle_msg(msg: Any) -> None:
    """
    SDK から受け取ったメッセージを処理して、
    deposit / withdraw / internalTransfer だけ CSV に追記する。
    """
    t_local_ns = now_ns()
    t_local_iso = ns_to_iso(t_local_ns)

    # 安全のため、構造をチェック
    if not isinstance(msg, dict) or msg.get("channel") != "userNonFundingLedgerUpdates":
        print(f"\n[WARN] unexpected message: {msg!r}")
        return

    data: Dict[str, Any] = msg.get("data", {})
    is_snapshot: bool = bool(data.get("isSnapshot", False))
    updates: List[Dict[str, Any]] = data.get("nonFundingLedgerUpdates", [])

    if is_snapshot:
        # 初回スナップショット → 件数だけ表示して終了（CSV には書かない）
        print(
            f"\n===== snapshot received =====\n"
            f" local_time (PC): {t_local_iso}\n"
            f" num_updates: {len(updates)}\n"
            f" (測定には使わないので CSV には記録しません)\n"
        )
        return

    if not updates:
        print(f"\n[INFO] empty updates at {t_local_iso}")
        return

    rows: List[List[Any]] = []

    for item in updates:
        ts_ms = int(item.get("time", 0))
        tx_hash = item.get("hash", "")
        delta: Dict[str, Any] = item.get("delta", {})

        etype: str = delta.get("type", "")
        # 測定対象の種類だけ拾う
        if etype not in ("deposit", "withdraw", "internalTransfer"):
            continue

        usdc = delta.get("usdc", "")
        fee = delta.get("fee", "")
        vault = delta.get("vault", "")
        user = delta.get("user", "")
        destination = delta.get("destination", "")

        ledger_iso = _ms_to_iso(ts_ms)

        row = [
            t_local_ns,
            t_local_iso,
            ts_ms,
            ledger_iso,
            etype,
            usdc,
            fee,
            vault,
            user,
            destination,
            tx_hash,
        ]
        rows.append(row)

        # 画面にも要約を出す
        print("\n===== nonFundingLedgerUpdate =====")
        print(f" local_time (PC): {t_local_iso}")
        print(f" ledger_time  (HL): {ledger_iso} ({ts_ms} ms)")
        print(
            f" type={etype}, usdc={usdc}, fee={fee}, "
            f"vault={vault}, user={user}, dest={destination}"
        )
        print(f" tx_hash={tx_hash}")

    if rows:
        try:
            with LOG_PATH.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"\n[INFO] {len(rows)} row(s) appended to {LOG_PATH}")
        except Exception as e:
            print(f"\n[ERROR] failed to write CSV: {e}")


def main() -> None:
    _ensure_logfile()

    print("Connecting Hyperliquid Info WebSocket via SDK (mainnet)...")
    print(f"User address: {HL_USER_ADDRESS}")

    info = Info(base_url=constants.MAINNET_API_URL)

    subscription = {
        "type": "userNonFundingLedgerUpdates",
        "user": HL_USER_ADDRESS,
    }

    print(f"Subscribing with: {subscription}")
    print("※ Ctrl+C で終了できます。")

    # ここでブロックして、以後 handle_msg がメッセージごとに呼ばれる
    info.subscribe(subscription, handle_msg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n手動で停止しました。")
