# t4_cctp/ — T4（CCTP実測）スクリプト置き場

T4（Circle CCTP による Arbitrum⇄HyperCore のトークン転送実測）のスクリプト。
**Deposit（Arbitrum→HyperCore, Fast+Forwarding）は実装・実測済み**（exp_id=3 で E2E≈7秒を取得）。

## 構成（一目で分かる版）
```
t4_cctp/
  deposit/                         ← ★Deposit本番で実行するファイルはすべてここ
    deposit_cctp_measure.py        ← 本体（burn→iris→mint→credit測定）。実行: python3 -u t4_cctp/deposit/deposit_cctp_measure.py [--broadcast] [--prod] [--exp-id N]
    config.py                      ← 定数・アドレス・パス（本番/test CSV パスもここ）
    check_fee.py                   ← maxFee を fee API 実値で確定（送金しない・事前確認用）
    check_balance.py               ← 残高/allowance/nonce 確認（送金しない）
    recover_cctp.py                ← 配送失敗時の自力回収（mintAndForward。元本不変・HyperEVMガスのみ）
    inspect_real_deposits.py       ← 実例 deposit のパラメータ突合（read-only 調査）
    T4_CCTP_deposit_spec_v3.md     ← ★確定仕様（列定義・maxFee式・検知器・出力先）
  _archive/                        ← 使い捨て/一度きりスクリプト（_verify_*, migrate_csv_v3 等）
  README.md / __init__.py
```

## 実行コマンド
- 事前確認（送金なし）: `python3 -u t4_cctp/deposit/check_balance.py` / `python3 -u t4_cctp/deposit/check_fee.py`
- ドライラン: `python3 -u t4_cctp/deposit/deposit_cctp_measure.py`
- test 実送信: `python3 -u t4_cctp/deposit/deposit_cctp_measure.py --broadcast`
- **本番**（200件・1ファイル追記）: `python3 -u t4_cctp/deposit/deposit_cctp_measure.py --broadcast --prod`

## 出力先
- 本番 最終CSV: `result/T4_cctp/deposit_cctp_latency.csv`（experiment_id を連番追記）／中間: `result/T4_cctp/prod/`
- test CSV: `result/T4_cctp/test/deposit/deposit_cctp_test.csv`／中間: 同 `test/deposit/`

## 確定仕様
- 正本は **`t4_cctp/deposit/T4_CCTP_deposit_spec_v3.md`**（v2 から更新。maxFee二層・credit=send/CoreDepositWallet・E2E二起点を反映）。

## 重要原則（要点）
- Deposit = Fast（minFinalityThreshold=1000）／domain: Arbitrum=3 / HyperEVM=19
- maxFee = ceil((base + forwardFee[med]) × 1.05)、安全弁 maxFee < forwardFee[med] で中止
- credit 検知は `delta.type=="send"` ∧ from=CoreDepositWallet ∧ USDC ∧ spot
