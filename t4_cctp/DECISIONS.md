# T4 CCTP 測定 — 決定記録（DECISIONS）

日付入りで方針決定を時系列に記録する。コードの正本は各スクリプト、設計の正本は
`t4_cctp/deposit/T4_CCTP_deposit_spec_v3.md`。本ファイルは「なぜそう決めたか」を残す。

---

## 【決定 2026-06-04】t2クランプ / 打ち切り（censoring）方針の統一 — 全件保持

**deposit / withdraw 両方向とも「全件保持」とする。**

- クランプ発動行（`t2_raw > mint時刻`：公開Iris の COMPLETE 反映が forwarder の実取得より遅れた回）も
  **破棄・再試行せず、そのまま CSV に残す**（現コードの挙動どおり）。
- 分析時、クランプ行は **打ち切り（censored）データ**として扱う：
  - `iris_wait` は真の CCTP 信頼層レイテンシの **上界値**（true ≤ 記録値）。
  - `attestation_to_mint = 0` は relay≈0 の上界。
  - 生存時間解析 S(t) / 打ち切り対応推定（Kaplan–Meier 等）と整合する扱い。
- **旧方針「本番は全件クランプ非発動で揃える」は撤回。** 理由：
  1. クランプ発動回だけ捨てると「速い attestation」が過剰排除され **選択バイアス** が入る。
  2. 非発動になるまで再試行すると **送金コスト** が増える。
  3. 全件保持＋打ち切り扱いが、バイアス無く統計的に正しい。
- **根拠の実測**：公開Iris の COMPLETE 反映は forwarder の実 attestation 取得より遅れることがある。
  - wit prod `exp_id=1`: `t2_raw` が `t3`(mint) の **約4秒後** → クランプ発動（`attestation_to_mint=0`, `iris_wait=5000`=上界）。
  - wit test `exp_id=2`: 非発動（`iris_wait=7187`, `attestation_to_mint=813` の物理分解が成立）。
- **将来のスケジューラに「クランプ発動による再試行」ロジックを入れないこと。**
- CSV は raw/clamped 両値を記録済み（dep: `t2_iris_attestation_complete` / `t2_iris_complete_local`、
  wit: 同名列）。よって **コード変更は一切不要**。
- 反映先: `T4_CCTP_deposit_spec_v3.md` §8 に追記済み。

---

## 【決定 2026-06-04】E2E 指標の役割分担と CSV 列順（前回「列入れ替え」指示は撤回）

**CSV 列順は据え置き**：両 CSV とも末尾は `..., E2E_*_onchain(ms), E2E_*_hypercore|onchain(ms), E2E_*_wallclock(ms)`
で **wallclock を最右翼（headline）** とする。前回の「E2E_wit_hypercore を最終列へ入れ替え」指示は研究設計上の
誤りとして撤回（検証の結果、入れ替えは未実施だったため revert 不要）。

E2E の 2 軸の役割：
- **`E2E_*_wallclock(ms) = t3 − t0/1e6`**（PC送信→着金チェーン時刻・**無補正**）。
  T1 の `latency(ms)`（withdraw は no_offset 版）と**同一区間・同一構成** → **T1 vs T4 比較の headline 指標**。CSV 最終列。
- **`E2E_wit_hypercore(ms)`（dep 側は `E2E_dep_onchain(ms)`）= チェーン時刻系 E2E**。
  3 区間（wit: hypercore_to_burn / iris_wait / attestation_to_mint、dep: iris_wait / attestation_to_mint / credit_wait）の
  **和と厳密一致する構造分解の主軸**。PC 時計 skew が乗らない。
- **`hc_debit_offset(ms)` / `rtt_offset(ms)`** = wallclock に含まれる **PC 時計 skew の診断列**。
  **どの指標の計算にも不使用**（T1 で offset 補正を本計算に混ぜた事故の再発防止）。

根拠：T1 の `latency(ms)` は両方向とも「PC送信(t0) → 着金チェーン時刻(t3)」の無補正生値であり、T4 の
`E2E_*_wallclock` と同一。T4 は T1 との比較論文のため、比較 headline は wallclock に固定する。
詳細な列対応は `t4_cctp/SCHEMA.md` を正本とする。

---

## 【決定 2026-06-11】両側目標を 210 件へ変更 ＋ deposit id=108 を事後削除

**背景**: deposit id=108 は時系列矛盾を含む測定失敗行（除外対象）。
- t3_hc_credit(1780923690059) が t2_iris_attestation(1780923697425) より **約7.37秒「前」**（着金が attestation 完了より前＝不整合）。
- iris_wait=13425ms > E2E_dep_onchain=6059ms（部分区間が全体を超過）。
- t2.5 mint block=0/0・hevm_mint_tx=N/A、attestation_to_mint/credit_wait/src_inclusion 空欄。隣接 107/109 は健全（telescoping 一致）。
- 事後に値を埋めても矛盾は解消しないため **物理削除**する。

**決定**:
1. 研究の比較設計上、両方向の**有効データを各210件**に揃える。id=108 を後で削除するため
   **deposit は実取得を211件**にし、削除後に210へ着地させる（**target_dep=211 / target_wit=210**）。
2. スケジューラを**方向別目標対応**に改修（後方互換維持）:
   - `plan.json` に `target_dep` / `target_wit` を追加。runtime ゲート（`_run_one_direction`）と
     `_status_line` / `print_summary` / `_verify_plan.py` は `_target_for(plan, direction)` で方向別目標を参照。
   - 単一 `target` しか無い旧 plan は従来どおり両方向同一でフォールバック。**measure スクリプトは無改変**。
3. 完了期限は据え置き（**6/12 夕刻**）。残数増（dep_rem=38/wit_rem=36）を本日14:30〜6/12 17:00 の5セルに
   再配置（20分間隔・dep→wit 2〜5分は厳守）。最終実イベント 6/12 16:48・SPARE は target ゲートで非発火。
4. **id=108 の物理削除＋リナンバリングは全測定完了・スケジューラ停止後にのみ実施**（Phase 2）。
   生データは `deposit_cctp_latency.raw_backup.csv` に完全保存してから削除。dep/wit の id 対応関係は
   SCHEMA を確認の上で扱う（往復ペアリングの有無を要確認）。

---

## 【追補 2026-06-12】deposit id=189 も削除対象 → target_dep を 212 へ

deposit id=189 も測定失敗行（id=108 と同様に除外）:
- t2_iris_attestation=0 / t2.5_hevm_mint=0 / t3_hc_credit=0 / hevm_mint_tx=N/A、cctp_nonce・message_hash 空、
  iris_wait〜E2E・feeExecuted・forward_state すべて空欄。burn(t1/arb_burn_tx)のみ成立し downstream を一切捕捉できず。
  隣接 188/190 は健全。

**決定変更**: deposit は **id=108 と id=189 の2行**を削除するため、実取得を **212件** にして削除後 210 着地。
- **target_dep=212 / target_wit=210**（target_wit 据置）。Phase1 計画を再生成（残数 dep9/wit8、本日17:30までに完了）。
- Phase2 の削除対象は **id=108 と id=189 の2行**。両方とも raw_backup 退避後に削除 → 1..210 連番振り直し。

## 【追補 2026-06-12】PC ロック画面（蓋開け時のPIN）の復帰について

「蓋を閉じて開けてもロック画面（PIN入力）が出ず、閉じる前の状態が続く」現象は **専用の設定変更ではなく、
T4 計測のための2つの状態の副作用**:
- keep-awake の `ES_DISPLAY_REQUIRED`（画面を常時ONに保持＝ロック画面/サインインが発生しない）
- LIDACTION=0（蓋閉じで何もしない）

→ いずれも既存の復元機構で自動的に元へ戻る:
1. **本日 測定完了時（STATUS=DONE）**: watchdog が keep-awake を解除 → 画面オフ可能 → アイドルでロック復活。
2. **2026-06-13 09:00（`T4_RestoreLid` SYSTEMタスク）**: LIDACTION=1（スリープ）復元 → 蓋閉じ→スリープ→ロック復活、
   隠し属性復元、自タスク削除。
- サインイン要求（CONSOLELOCK 等）自体は未変更＝ユーザー元設定のまま。蓋閉じロックを本日中に戻したい場合のみ
  管理者で `C:\Users\Public\t4_restore_lid.ps1` を手動実行（任意・既定は 6/13 自動復元）。

## 【完了 2026-06-12】Phase2 実行: id=108, 189 削除 ＋ リナンバリング 完了

全測定完了（取得 deposit 212 / withdraw 210）後、スケジューラ停止状態で実施。
- **全行監査**（`scheduler/tools/audit_csv.py`）: deposit 異常= **{108, 189} のみ**（telescoping 210/212・正常クランプ20行は誤検知せず）、
  withdraw **異常ゼロ**（telescoping 210/210・クランプ32行健全）。108/189 以外に欠損・異常なしを確認。
- **削除＋連番**（`scheduler/tools/drop_bad_rows.py`, 多重実行ガード付）: 原本を
  `result/T4_cctp/deposit_cctp_latency.raw_backup.csv` に完全保存 → id=108,189 を物理削除 → 残210行を 1..210 連番に振り直し。
  **experiment_id 列以外は一切改変せず**（行順保持・物理行は arb_burn_tx_hash で対応追跡）。
- **最終検証**（`scheduler/tools/final_verify.py`）: backup−{108,189} と新CSV が非id列で**完全一致（不一致0セル）**、
  deposit 210 / withdraw 210・id 1..210 連番・重複なし・telescoping 全行一致・両CSV異常ゼロ。
- **dep/wit の experiment_id は方向別独立採番**（withdraw は deposit id を非参照・件数も元から非整列）＝ペアリング無し。
  よって deposit リナンバリングは withdraw に影響なし（未改変）。

**最終データセット: deposit 210 / withdraw 210（各 id 1..210・全行健全）。** raw_backup は監査・再現性のため保持。
