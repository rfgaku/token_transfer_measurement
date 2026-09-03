# Hyperliquid ネイティブブリッジ 到着率・同時負荷 追加調査 作業ログ

ACM DLT 投稿論文 付録用の追加調査。**読み取り専用**（送金・ブロードキャストは一切行っていない）。
既存ファイルの変更・移動・削除なし。新規ファイルは `experiments/` 配下（スクリプト・本 NOTES）と
`result/native_bridge_survey/` 配下（出力 CSV、および中間物用の `_tmp/`）のみ。コミットは行っていない。

- スクリプト: `experiments/native_bridge_arrival_survey.py`（単一ファイル、argparse）
- 実施日: 2026-09-02（JST）／ログ中の時刻表記は UTC

---

## 1. 前提の確認

### 1.1 Bridge2 コントラクトアドレス

| 出所 | アドレス |
|---|---|
| `deposit_latency_measure.py:51-53`（deposit 送信先 `HL_DEPOSIT_BRIDGE_ADDRESS` の既定値） | `0x2df1c51e09aecf9cacb7bc98cb1742757f163df7` |
| `withdraw_latency_measure.py:73-75`（withdraw 受領元フィルタ） | 同上 |
| `.env` の `HL_DEPOSIT_BRIDGE_ADDRESS` | 同上 |
| Arbiscan のコントラクトラベル | `Hyperliquid: Deposit Bridge 2`（`0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7`） |
| Hyperliquid 公式 GitHub `hyperliquid-dex/contracts/Bridge2.sol` | 同コントラクトのソース（イベント定義を本調査で利用） |

**照合結果: 一致。**

補足: Hyperliquid 公式 GitBook は現在アドレスを本文に直書きしていない
（`hypercore/usdc.md` は「Arbitrum の旧ブリッジは HyperCore の USDC 供給の 10% 未満」と記載するのみ、
`architecture/bridge` は 404）。そのため公式リポジトリのソース + Arbiscan の公式ラベル + 実測スクリプトの
3 点照合とし、さらに以下のオンチェーン検証を実施した（`verify` サブコマンド）。

```
Bridge2 code size   : 19394 bytes        （コントラクトとして実在）
Bridge2 USDC balance: 349,477,886.91 USDC （エスクロー残高として妥当）
```

### 1.2 Arbitrum One の USDC（ネイティブ USDC）

| 出所 | アドレス |
|---|---|
| `deposit_latency_measure.py:46-48` / `withdraw_latency_measure.py:76-78` の既定値 | `0xaf88d065e77c8cC2239327C5EDb3A432268e5831` |
| `.env` の `ARB_USDC_ADDRESS` | 同上 |
| オンチェーン `symbol()` 呼び出し結果 | `USDC` |

**照合結果: 一致。**（`0xff970a61...` は Bridged USDC.e で別物。本調査では使用しない）

### 1.3 RPC エンドポイント

- 既存設定を再利用: `.env` の `ARBITRUM_HTTP_RPC = https://arb1.arbitrum.io/rpc`
  （スクリプトは `ARB_RPC_URL` → `ARBITRUM_HTTP_RPC` の順に読む）。**新しい鍵は作成していない。**
- `eth_getLogs` は上記エンドポイントのみを使用。
- **block timestamp の一括解決についてのみ**、鍵不要の公開エンドポイントを併用した（下記 4.3）。

---

## 2. 付録 E の 7 窓の復元

### 2.1 経路 (a): DepositForBurn 収集スクリプト

`t4_cctp/` 配下を全走査したが、**DepositForBurn を収集して
`result/T4_cctp/cctp_fast_standard_events.csv` を生成したスクリプトはリポジトリに存在しない**。

```
$ grep -rn "cctp_fast_standard" --include=*.py .
t4_cctp/queueing_sim/queueing_sim.py:57   EVENTS_CSV = ... （読む側のみ）
```

`t4_cctp/deposit/deposit_cctp_measure.py` は自分の送金の DepositForBurn を扱うだけで、
悉皆収集の窓定義は持っていない。よって経路 (a) は**取得不可**。

### 2.2 経路 (b): events CSV からの復元（採用）

`result/T4_cctp/cctp_fast_standard_events.csv` の `chain == "Arbitrum One"` 行について、
`window_id` ごとの `block_time_utc` / `block_number` の min/max を窓境界とした
（`t4_cctp/queueing_sim/queueing_sim.py` の `estimate_lambdas()` と同じ扱い）。

**7 窓すべて復元できた。**

| window_id | start_block | end_block | start_utc | end_utc | duration_s | CCTP events |
|---|---|---|---|---|---|---|
| 1 | 477127340 | 477177151 | 2026-06-25T07:54:13Z | 2026-06-25T11:21:34Z | 12441.0 | 571 |
| 2 | 476450039 | 476500037 | 2026-06-23T08:54:06Z | 2026-06-23T12:22:47Z | 12521.0 | 845 |
| 3 | 475787707 | 475837506 | 2026-06-21T10:54:49Z | 2026-06-21T14:22:43Z | 12474.0 | 475 |
| 4 | 475142884 | 475192836 | 2026-06-19T13:54:12Z | 2026-06-19T17:22:35Z | 12503.0 | 752 |
| 5 | 474454614 | 474504559 | 2026-06-17T13:54:15Z | 2026-06-17T17:22:19Z | 12484.0 | 997 |
| 6 | 473811532 | 473861498 | 2026-06-15T16:54:14Z | 2026-06-15T20:23:03Z | 12529.0 | 796 |
| 7 | 473126496 | 473176451 | 2026-06-13T16:54:12Z | 2026-06-13T20:25:16Z | 12664.0 | 403 |

整合性チェック:
- 7 窓の block 数の合計は `end-start+1` の総和で **349,672**。
  `cctp_fast_standard_summary.csv` の Arbitrum One `total_blocks_covered = 350007`（= 7 × 50,001）と
  **335 ブロック差**。これは元調査が「50,001 ブロック固定幅」で窓を切っており、
  窓の端に CCTP イベントが無かった分だけ min/max が内側に寄るためで、矛盾ではない。
  本調査は指示どおり (b) の min/max を窓境界として採用した。
- Arbitrum One の 7 窓合計 `duration_s = 87,616 s`。窓期間は
  `2026-06-13T16:54:12Z .. 2026-06-25T11:21:34Z` に散在（2 日おき・約 3.5 時間/窓）。

---

## 3. 収集した事象

USDC (`0xaf88d065...5831`) の `Transfer(address indexed from, address indexed to, uint256 value)`
（topic0 = `0xddf252ad...b3ef`）を、topic フィルタで 2 系列に分けて取得。

| direction | フィルタ | 意味 |
|---|---|---|
| `deposit` | `topics = [Transfer, *, Bridge2]`（`to == Bridge2`） | ネイティブブリッジへの deposit 到着 |
| `withdraw` | `topics = [Transfer, Bridge2, *]`（`from == Bridge2`） | Bridge2 からの withdraw 放出 |

記録項目: `block_number, block_timestamp_utc, block_timestamp_ms, tx_hash, log_index,
from, to, amount_usdc, direction`（期間 B は `window_id` 列を追加）。

参考系列として、Bridge2 コントラクト自身のイベント（`Deposit` / `RequestedWithdrawal` /
`FinalizedWithdrawal` ほか、topic0 は `Bridge2.sol` の宣言から keccak256 で算出）も
`bridge-events` サブコマンドで取得できるようにした（結果は下記 6）。

---

## 4. 実装メモ

### 4.1 日時 → ブロック番号（二分探索）

`eth_getBlockByNumber` を使った二分探索で「timestamp >= target となる最小ブロック」を求めた。

| 目標時刻 (UTC) | unix ts | 求まったブロック | そのブロックの ts | 探索ステップ |
|---|---|---|---|---|
| 2025-11-27T00:00:00Z | 1764201600 | **404479803** | 1764201600 | 23 |
| 2025-12-09T00:00:00Z | 1765238400 | **408635723** | 1765238400 | 23 |

→ 期間 A の対象ブロック範囲は **404479803 .. 408635722（4,155,920 ブロック）**。
取得後、`block_timestamp_ms` が `[1764201600000, 1765238400000)` に入る行だけを残して端点を厳密化した。

### 4.2 eth_getLogs の分割・リトライ・レート制限対策

- 初期スパン 50,000 ブロック、上限 100,000、下限 500。
- `log query timed out` / `limit` / `429` 等のエラーを検出したらスパンを **半減**、
  成功が続いたら 1.25 倍ずつ戻す（`get_logs_adaptive`）。
- HTTP 429・JSON-RPC 429・タイムアウトは指数バックオフ（最大 30 s）で最大 4〜8 回リトライ。
- 公開 RPC は `User-Agent` 無しだと **403** を返すことがあるため、明示的に付与している。
- 実測では期間 A の 415 万ブロックをスパン 100,000 のまま完走（縮小発動なし）。

### 4.3 block timestamp の解決（性能上の判断）

`eth_getLogs` の戻り値に timestamp が無いため、イベントが載ったブロックの timestamp を別途取得する必要がある。
期間 A では **81,231 ブロック**が対象。

- `arb1.arbitrum.io/rpc` は 1 リクエストあたりのコスト上限が厳しく、
  バッチ 50 件 × 3 並列で 429 が連発、単スレッド・バッチ 25 では **約 15.6 blocks/s**
  （= 期間 A に 約 87 分）だった。
- timestamp は**チェーン上の客観値でどのノードから引いても同一**であるため、
  鍵不要の公開エンドポイントを併用して並列化した:
  - `https://arbitrum-one.public.blastapi.io`
  - `https://arbitrum-one-rpc.publicnode.com`
  - `https://arb1.arbitrum.io/rpc`（既存設定）
- バッチ 200 件、エンドポイント数 × 2 スレッド、欠けた id だけを別エンドポイントに再投入する方式。
- **`eth_getLogs`（＝イベントの悉皆性を左右する部分）は既存設定の `arb1.arbitrum.io/rpc` のみ**で行っており、
  追加エンドポイントは timestamp 解決にしか使っていない。新しい鍵・アカウントは作成していない。

### 4.4 冪等性

`result/native_bridge_survey/_tmp/` に以下の一時ファイルを残す。
途中で落ちても同じコマンドの再実行で続きから取得する。

| ファイル | 役割 |
|---|---|
| `progress.json` | 期間・方向・窓ごとの取得済み最終ブロック、および期間 A のブロック範囲メタ |
| `native_bridge_events_2025-11.partial.csv` / `..._2026-06.partial.csv` | 取得済み生ログ（500,000 ブロックごとに追記コミット） |
| `bridge_2025-11.partial.csv` / `bridge_2026-06.partial.csv` | Bridge2 自身のイベントの生ログ |
| `blockts_cache.json` | block_number → timestamp キャッシュ |

**成果物 CSV は `result/native_bridge_survey/` 直下、中間物は `_tmp/` 配下**に分離しており、
`_tmp/` は削除して差し支えない（再実行で作り直される）。
本調査の実施後、初回実行時の中間物と実行ログは指示により削除済み。

実際に一度、timestamp 解決の途中で 429 により停止したが、再実行で `eth_getLogs` を再取得せずに
継続できることを確認済み（`partial 再利用: ... rows` のログ）。

### 4.5 依存パッケージ

`requests` / `pandas` / `numpy` / `scipy` / `python-dotenv` のみ。**いずれも既存環境に導入済みで、
`requirements` 等には一切手を加えていない**。`web3` は本スクリプトでは使っていない
（生 JSON-RPC を直接叩いてバッチ処理とレート制御を細かく制御するため）。

---

## 5. 結果

### 5.1 出力ファイル（すべて `result/native_bridge_survey/` 配下）

| ファイル | 行数 | 内容 |
|---|---|---|
| `native_bridge_events_2025-11.csv` | 107,341 | 期間 A の全イベント |
| `native_bridge_events_2026-06.csv` | 3,480 | 期間 B の全イベント（`window_id` 列つき） |
| `native_bridge_arrival_rates.csv` | 42 | 期間 A（全体 2＋日別 24）・期間 B（窓別 14＋合算 2）の direction 別集計 |
| `t1_deposit_concurrency.csv` | 117 | T1 deposit 実測 117 件の同時負荷 |
| `t1_withdraw_concurrency.csv` | 117 | T1 withdraw 実測 117 件の同時負荷 |
| `native_bridge_contract_events_2026-06.csv` | 3,023 | （参考）Bridge2 自身のイベント・期間 B |
| `native_bridge_contract_events_2025-11.csv` | 101,295 | （参考）Bridge2 自身のイベント・期間 A |

### 5.2 期間 A（2025-11-27 00:00 – 2025-12-09 00:00 UTC, 1,036,800 s）

| direction | 件数 | 到着率 [件/秒] | 平均額 [USDC] | 中央値額 | p90 額 |
|---|---|---|---|---|---|
| deposit  | 57,195 | **0.055165** | 18,866.29 | 234.80 | 11,000.00 |
| withdraw | 50,146 | **0.048366** | 22,023.70 | 299.00 | 15,891.34 |

日別（1 日 = 86,400 s 固定で除算）:

| 日 (UTC) | deposit 件数 | deposit 率 | withdraw 件数 | withdraw 率 |
|---|---|---|---|---|
| 2025-11-27 | 6,479 | 0.074988 | 5,403 | 0.062535 |
| 2025-11-28 | 4,881 | 0.056493 | 4,297 | 0.049734 |
| 2025-11-29 | 3,226 | 0.037338 | 3,173 | 0.036725 |
| 2025-11-30 | 3,339 | 0.038646 | 3,148 | 0.036435 |
| 2025-12-01 | 8,172 | **0.094583**（最大） | 6,774 | **0.078403**（最大） |
| 2025-12-02 | 6,164 | 0.071343 | 5,012 | 0.058009 |
| 2025-12-03 | 5,445 | 0.063021 | 4,554 | 0.052708 |
| 2025-12-04 | 4,104 | 0.047500 | 4,377 | 0.050660 |
| 2025-12-05 | 4,551 | 0.052674 | 3,991 | 0.046192 |
| 2025-12-06 | 3,360 | 0.038889 | 2,640 | **0.030556**（最小） |
| 2025-12-07 | 3,624 | 0.041944 | 3,227 | 0.037350 |
| 2025-12-08 | 3,850 | 0.044560 | 3,550 | 0.041088 |

日別レンジ: deposit 0.0373–0.0946 件/秒（2.5 倍）、withdraw 0.0306–0.0784 件/秒（2.6 倍）。

### 5.3 期間 B（付録 E と同一の 7 窓, 合計 87,616 s）

| window | 期間 (UTC) | 長さ [s] | deposit 件数 | deposit 率 | withdraw 件数 | withdraw 率 |
|---|---|---|---|---|---|---|
| w1 | 2026-06-25 07:54:13–11:21:34 | 12,441 | 266 | 0.021381 | 169 | 0.013584 |
| w2 | 2026-06-23 08:54:06–12:22:47 | 12,521 | 331 | 0.026436 | 183 | 0.014615 |
| w3 | 2026-06-21 10:54:49–14:22:43 | 12,474 | 215 | 0.017236 | 141 | 0.011304 |
| w4 | 2026-06-19 13:54:12–17:22:35 | 12,503 | 284 | 0.022715 | 326 | 0.026074 |
| w5 | 2026-06-17 13:54:15–17:22:19 | 12,484 | 403 | **0.032281**（最大） | 279 | 0.022349 |
| w6 | 2026-06-15 16:54:14–20:23:03 | 12,529 | 303 | 0.024184 | 254 | 0.020273 |
| w7 | 2026-06-13 16:54:12–20:25:16 | 12,664 | 178 | **0.014056**（最小） | 148 | 0.011687 |
| **合算** | — | **87,616** | **1,980** | **0.022599** | **1,500** | **0.017120** |

額（合算）: deposit 平均 18,120.09 / 中央値 170.94 / p90 15,000.99 USDC、
withdraw 平均 89,399.11 / 中央値 71.08 / p90 10,614.71 USDC。
（withdraw の平均が極端に大きいのは w5 に単発の巨額放出が含まれるため。平均は裾に強く引かれる）

### 5.4 期間 A と期間 B の対比（数値のみ、解釈は行わない）

- deposit 到着率: 期間 A 0.055165 → 期間 B 0.022599（B / A = 0.410）
- withdraw 放出率: 期間 A 0.048366 → 期間 B 0.017120（B / A = 0.354）
- 参考: 同じ 7 窓での CCTP（Arbitrum One, Fast）の到着率は
  `queueing_sim.py` の `estimate_lambdas()` で 4,415 / 87,616 = **0.050390 件/秒**。

### 5.5 T1 実測 117 件の同時負荷

いずれも「他利用者の」件数（自分の 117 deposit tx と 117 withdraw tx の tx_hash は除外）。
窓の定義は直前系が `[t - X, t - 1 ms]`（自分の到着時刻を含まない）、`±60 s` 系が `[t - 60 s, t + 60 s]`。

`t1_deposit_concurrency.csv`:

| 指標 | 平均 | 中央値 | 最小 | 最大 |
|---|---|---|---|---|
| 直前 16 s の他者 deposit 件数 | 0.761 | 1 | 0 | 5 |
| 直前 60 s の他者 deposit 件数 | 2.769 | 3 | 0 | 8 |
| 前後 ±60 s の他者 deposit 件数 | 5.872 | 6 | 0 | 19 |
| latency(ms) | 8,687.05 | 8,523.22 | 2,585.81 | 47,321.02 |

`t1_withdraw_concurrency.csv`:

| 指標 | 平均 | 中央値 | 最小 | 最大 |
|---|---|---|---|---|
| 直前 200 s の他者 withdraw 放出件数 | 7.863 | 7 | 0 | 20 |
| 前後 ±60 s の他者 withdraw 放出件数 | 4.111 | 4 | 0 | 16 |
| latency(ms) | 229,306.00 | 229,774.19 | 152,759.18 | 253,716.81 |

### 5.6 Spearman 順位相関（参考値・解釈は行わない）

| 対 | r | p | n |
|---|---|---|---|
| **直前 16 s の他者 deposit 件数 × latency(ms)** | **0.0987** | **0.2899** | 117 |
| 直前 60 s の他者 deposit 件数 × latency(ms) | -0.0000 | 0.9996 | 117 |
| 直前 200 s の他者 withdraw 件数 × latency(ms) | 0.0201 | 0.8296 | 117 |

---

## 6. 参考: Bridge2 コントラクト自身のイベント

`Bridge2.sol` の event 宣言から keccak256 で topic0 を算出し、`address = Bridge2` の生ログを分類した。

### 6.1 期間 B（7 窓, 3,023 件）

| event | 件数 |
|---|---|
| `FinalizedWithdrawal` | 1,500 |
| `RequestedWithdrawal` | 1,498 |
| `FailedPermitDeposit` | 21 |
| `FailedWithdrawal` | 4 |

**相互検証**: `FinalizedWithdrawal` 1,500 件は、Transfer 基準の withdraw 放出 1,500 件と**完全に一致**。
Transfer 基準の 2 系列が Bridge2 の実挙動を取りこぼしていないことの裏づけになる。

`RequestedWithdrawal`（1,498）と `FinalizedWithdrawal`（1,500）の差は、窓境界をまたいで
request が窓外・finalize が窓内になった分（Bridge2 の dispute period が約 200 s あるため境界で必ず生じる）。

`Bridge2.Deposit` イベントは 0 件。これは deposit が Bridge2 への**素の ERC-20 `transfer`** で行われ、
コントラクト呼び出しを伴わないため（`Deposit` イベントは permit 経由の入金でのみ発火する）。
したがって **deposit 到着の悉皆計測には Transfer 基準が必須**であり、本調査の設計が正しいことを確認した。

### 6.2 期間 A（101,295 件）

| event | 件数 |
|---|---|
| `FinalizedWithdrawal` | 50,146 |
| `RequestedWithdrawal` | 50,141 |
| `FailedWithdrawal` | 805 |
| `FailedPermitDeposit` | 203 |

**相互検証**: `FinalizedWithdrawal` 50,146 件は、Transfer 基準の withdraw 放出 50,146 件と**完全に一致**。
`Bridge2.Deposit` はここでも 0 件（6.1 と同じ理由）。
`FailedWithdrawal` は 12 日間で 805 件（放出 50,146 件に対して 1.58%）。

---

## 7. 取得に失敗した範囲

**なし。**

- 期間 A: ブロック 404,479,803 – 408,635,722（4,155,920 ブロック）を欠落なく走査。
  `eth_getLogs` はスパン 100,000 のまま縮小発動せずに完走。
- 期間 B: 7 窓すべて（合計 349,672 ブロック）を欠落なく走査。
- block timestamp: 期間 A の 81,231 ブロック、期間 B の 3,067 ブロックすべて解決済み
  （未解決が残った場合は `attach_timestamps()` が例外を送出する設計。発生しなかった）。
- 途中で 2 回、公開 RPC の 429 によりプロセスが停止したが、
  いずれも冪等再開により `eth_getLogs` を再取得せずに継続し、最終的な取得範囲に欠落はない。

### 既知の留意点（測定上の制約であり失敗ではない）

1. Arbitrum One のブロック timestamp は**秒精度**のため、`block_timestamp_ms` は常に `*000` になる。
   同時負荷の窓（16 s / 60 s / 200 s）に対しては十分な分解能。
2. 1 つの tx が複数の withdraw Transfer を含む場合がある（Bridge2 のバッチ確定）。
   本調査は **Transfer ログ単位**で 1 件と数えている（`tx_hash` + `log_index` で一意化）。
3. 「他利用者」の除外は、T1 実測の deposit 117 件・withdraw 117 件の `arb_tx_hash` 全体を
   除外集合として用いた（指示の「自分の tx_hash を除外」を保守的に解釈）。
4. 期間 A の日別到着率は、1 日 = 86,400 s 固定で除算している（12 日すべて完全な 1 日）。

### 補足（リポジトリ運用上の指摘）

`.env` に `ARB_SENDER_PRIVATE_KEY` が平文で残っている。本調査では読み取り専用のため一切使用していないが、
論文の成果物公開・リポジトリ共有の前にローテーションと除去を推奨する。
