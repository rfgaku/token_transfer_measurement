# T4 CCTP 測定 CSV スキーマ（解析・論文の基礎資料）

最終更新 2026-06-04。正本：本ファイル。方針決定の経緯は `t4_cctp/DECISIONS.md`、設計は
`t4_cctp/deposit/T4_CCTP_deposit_spec_v3.md`。

対象 CSV:
- deposit: `result/T4_cctp/deposit_cctp_latency.csv`（31列）
- withdraw: `result/T4_cctp/withdraw_cctp_latency.csv`（29列）

---

## 0. 使う時計（clock）の種類

| 記号 | 時計 | 性質 |
|---|---|---|
| **PC** | 計測PCのローカル時計 (`time.time_ns`) | t0 と Iris poll 検知に使用。NTP 程度の絶対精度。チェーン/HyperCoreとの系間skewあり |
| **HC** | HyperCore ledger 時刻（consensus clock） | `userNonFundingLedgerUpdates` の `time` |
| **HEVM** | HyperEVM block.timestamp | 秒粒度（×1000でms化） |
| **ARB** | Arbitrum block.timestamp | 秒粒度（×1000でms化） |
| **Iris(PC)** | Iris の status が complete に変わった瞬間を PC poll(0.25s) で検知した PC時刻 | オフチェーン事象・チェーン刻印なし |

> 重要: 異なる時計をまたぐ差分（例 `E2E_*_wallclock` は ARB/HC − PC）には系間 skew が乗る。
> 純チェーン時刻どうしの差分（E2E_*_onchain 等）は skew が乗らない。詳細は §4。

---

## 1. 5点の時刻定義（両方向の対応）

| 記号 | deposit（Arbitrum→HyperCore） | withdraw（HyperCore→Arbitrum） |
|---|---|---|
| **t0** | PC が Arbitrum `depositForBurnWithHook` を送信 (PC) | PC が HyperCore `sendToEvmWithData` を /exchange へ送信 (PC) |
| **t_hc_debit** | （無し） | HyperCore が残高引落した ledger 時刻 (HC) |
| **t1** | Arbitrum で CCTP **burn** したブロック ts (ARB) | HyperEVM で CCTP **burn** したブロック ts (HEVM) |
| **t2** | Iris attestation 完了の poll 検知 (Iris/PC) | 同左 (Iris/PC) |
| **t2.5** | HyperEVM **mint**（Forwarder宛）ブロック ts (HEVM) | （無し） |
| **t3** | HyperCore **credit** ledger 時刻 (HC) | Arbitrum で自分宛 **mint** したブロック ts (ARB) |

> 非対称の理由: deposit は forwarder が HEVM で mint→CoreDepositWallet 経由で HC credit するため
> t2.5(HEVM mint) と t3(HC credit) が分かれる。withdraw は Arbitrum 宛 自動 forwarding が
> 受信側で直接ユーザーへ mint するため、**mint=t3 が終点**（独立した t2.5 を持たない）。
> source-chain の **burn は両方向とも t1**。

---

## 2. deposit CSV 列対応（31列）

| # | 列名 | 定義 | 計算式 | 時計 | 備考 |
|---|---|---|---|---|---|
|1|experiment_id|連番ID|—|—|`get_next_experiment_id`|
|2|direction|`"deposit"`|—|—||
|3|amount_usdc(usdc)|送金額|—|—|gross|
|4|cctp_nonce|Iris eventNonce|—|Iris|burn後にIris割当|
|5|message_hash|keccak(message)|—|Iris||
|6|t1_arb_burn_block_ts(ms)|**t1**|burn tx の block.timestamp×1000|ARB||
|7|t1_arb_burn_block_number|burn block番号|—|ARB||
|8|arb_burn_tx_hash|burn tx|—|ARB||
|9|t2_iris_attestation_complete(ms)|**t2_raw**|status complete を poll検知した PC時刻|Iris(PC)|生値|
|10|t2_iris_complete_local(ms)|**t2(clamped)**|`min(t2_raw, t2_5)`|Iris(PC)/HEVM|§5クランプ|
|11|t2_5_hevm_mint_block_ts(ms)|**t2.5**|HEVM mint(Forwarder宛0x0→) block ts|HEVM||
|12|t2_5_hevm_mint_block_number|mint block番号|—|HEVM||
|13|hevm_mint_tx_hash|mint tx|—|HEVM||
|14|t3_hc_credit_ledger_time(ms)|**t3**|HyperCore credit ledger time|HC||
|15|minFinalityThreshold_set|設定値|1000(Fast)|—|送信時設定|
|16|finalityThresholdExecuted|実測|Iris返り値|Iris|Fast=1000期待|
|17|forward_state|配送状態|Iris forwardState|Iris|credit後取得|
|18|maxFee(usdc_atomic)|設定maxFee|—|—||
|19|feeExecuted(usdc_atomic)|実徴収|Iris|Iris||
|20|arb_gas_used(gas)|burn tx gas|—|ARB||
|21|arb_gas_price(wei)|—|—|ARB||
|22|arb_tx_fee(eth)|gas×price|—|ARB||
|23|t0_local_send(ns)|**t0**|burn 送信直前の `time.time_ns()`|PC||
|24|rtt_offset(ms)|診断|`t1 − t0/1e6`|ARB−PC|**診断専用・計算不使用**|
|25|arb_confirmations_at_attestation(blocks)|診断|attestation時のArb高 − burn block|ARB||
|26|iris_wait(ms)|区間|`t2 − t1`|—|**CCTP信頼層**|
|27|attestation_to_mint(ms)|区間|`t2_5 − t2`|—|relay+mint|
|28|credit_wait(ms)|区間|`t3 − t2_5`|—|mint→HC credit|
|29|src_inclusion(ms)|参考|`t2_5 − t1`|—|burn→mint合算|
|30|E2E_dep_onchain(ms)|**チェーン系E2E**|`t3 − t1`|HC−ARB|構造分解の主軸(§4)|
|31|E2E_dep_wallclock(ms)|**wallclock E2E**|`t3 − t0/1e6`|HC−PC|**headline**(§4)|

**検算（deposit）**: `iris_wait + attestation_to_mint + credit_wait == E2E_dep_onchain`
（t1→t2→t2.5→t3 の telescoping。t2 が打ち消し合うため厳密一致）。

---

## 3. withdraw CSV 列対応（29列）

| # | 列名 | 定義 | 計算式 | 時計 | 備考 |
|---|---|---|---|---|---|
|1|experiment_id|連番ID|—|—||
|2|direction|`"withdraw"`|—|—||
|3|amount_usdc(usdc)|送金額|—|—|gross|
|4|amount_received(usdc)|純着金|mint Transfer value/1e6|ARB|=amount−0.2|
|5|cctp_nonce|nonce|mint receiptのMessageReceived topic[2] / Iris|ARB/Iris||
|6|message_hash|keccak(message)|—|Iris||
|7|t_hc_debit_ledger_time(ms)|**t_hc_debit**|HyperCore引落 ledger time|HC|delta.type=="send"|
|8|t1_hevm_burn_block_ts(ms)|**t1**|HEVM burn(MessageSent) block ts|HEVM|WSライブ捕捉|
|9|t1_hevm_burn_block_number|burn block番号|—|HEVM||
|10|hevm_burn_tx_hash|burn tx|—|HEVM||
|11|t2_iris_attestation_complete(ms)|**t2_raw**|status complete poll検知 PC時刻|Iris(PC)|生値|
|12|t2_iris_complete_local(ms)|**t2(clamped)**|`min(t2_raw, t3)`|Iris(PC)/ARB|§5クランプ|
|13|t3_arb_mint_block_ts(ms)|**t3**|Arbitrum mint(0x0→自分) block ts|ARB||
|14|t3_arb_mint_block_number|mint block番号|—|ARB||
|15|arb_mint_tx_hash|mint tx|—|ARB||
|16|finalityThresholdExecuted|実測|mint receipt topic[3] / Iris|ARB/Iris|**実測2000(Finalized)**|
|17|forward_state|配送状態|Iris forwardState|Iris|保存時PENDING→後にCONFIRMED|
|18|maxFee(usdc_atomic)|プロトコル設定|Iris|Iris|実測200000(0.2)|
|19|feeExecuted(usdc_atomic)|実徴収|Iris|Iris|実測200000(0.2)|
|20|forwarding_fee(usdc_atomic)|forwarding費|=feeExecuted(あれば)/既定200000|Iris|**額非比例の固定0.2**|
|21|t0_local_send(ns)|**t0**|/exchange 送信直前の `time.time_ns()`|PC||
|22|hc_debit_offset(ms)|診断|`t_hc_debit − t0/1e6`|HC−PC|**診断専用・計算不使用**(実測−96〜−835ms)|
|23|hevm_confirmations_at_attestation(blocks)|診断|attestation時のHEVM高 − t1 burn block|HEVM||
|24|hypercore_to_burn(ms)|区間|`t1 − t_hc_debit`|HEVM−HC|HC引落→HEVM burn ルーティング|
|25|iris_wait(ms)|区間|`t2 − t1`|—|**CCTP信頼層**|
|26|attestation_to_mint(ms)|区間|`t3 − t2`|—|relay+receive|
|27|E2E_wit_onchain(ms)|区間E2E|`t3 − t1`|ARB−HEVM|=iris_wait+attestation_to_mint|
|28|E2E_wit_hypercore(ms)|**チェーン系E2E**|`t3 − t_hc_debit`|ARB−HC|構造分解の主軸(§4)|
|29|E2E_wit_wallclock(ms)|**wallclock E2E**|`t3 − t0/1e6`|ARB−PC|**headline**(§4)|

**検算（withdraw・2本）**:
- `hypercore_to_burn + iris_wait + attestation_to_mint == E2E_wit_hypercore`（t_hc_debit→t1→t2→t3 telescoping）
- `iris_wait + attestation_to_mint == E2E_wit_onchain`（t1→t2→t3 telescoping）
（いずれもクランプ発動有無に関わらず厳密一致＝t2が打ち消し合うため）

---

## 4. E2E の 2 軸と T1 との対応【決定 2026-06-04・DECISIONS.md】

| 軸 | 列（dep / wit） | 式 | 役割 | skew |
|---|---|---|---|---|
| **wallclock** | `E2E_dep_wallclock` / `E2E_wit_wallclock` | `t3 − t0/1e6` | **T1 vs T4 比較の headline**。CSV最終列 | PC↔chain系間skewを含む（無補正） |
| **chain系E2E** | `E2E_dep_onchain` / `E2E_wit_hypercore` | dep:`t3−t1` / wit:`t3−t_hc_debit` | **構造分解の主軸**（3区間の和と厳密一致） | PC skew無し |

**T1 列との対応（同一区間）**:
- T1 deposit `latency(ms)` = `hl_ledger_time − local_send` ↔ **T4 `E2E_dep_wallclock(ms)`**（= `t3_hc_credit − t0`）
- T1 withdraw `no_offset latency(ms)` = `arb_block_timestamp − local_broadcast` ↔ **T4 `E2E_wit_wallclock(ms)`**（= `t3_arb_mint − t0`）

→ T1 の latency は両方向とも「PC送信 → 着金チェーン時刻」の**無補正生値**。T4 の wallclock と
同一構成なので、論文の apple-to-apple 比較は wallclock 軸で行う。

**offset 列（`hc_debit_offset` / `rtt_offset`）は診断専用**。wallclock に含まれる PC 時計 skew の大きさを
可視化するだけで、**iris_wait/E2E などどの指標の計算にも使わない**（T1 で offset を本計算に混ぜた事故の再発防止）。

---

## 5. t2 クランプ機構と測定限界

**2列構成**:
- `t2_iris_attestation_complete(ms)` = **生検知**（status pending→complete を最初に poll で見た PC 時刻）
- `t2_iris_complete_local(ms)` = `min(生検知, mint時刻)`（dep は t2_5、wit は t3）= 区間計算に使う t2

**発動判別**: `t2_iris_complete_local < t2_iris_attestation_complete`（＝生検知 > mint時刻）なら**クランプ発動**。
発動時は `attestation_to_mint == 0`、`iris_wait == (mint − t1)`（= 上界）。

**分析時の扱い（DECISIONS.md / spec §8）**: 発動行も**全件保持**。発動行は**打ち切り（censored）**データ：
`iris_wait` は真の信頼層レイテンシの**上界**、`attestation_to_mint=0` は relay≈0 の上界。
生存時間解析 S(t)・打ち切り対応推定（Kaplan–Meier 等）と整合。**再試行ロジックは入れない**。

**★t2 の測定限界（重要）**: attestation 完了は**オフチェーン事象でチェーン刻印が無い**ため、t2 は
PC 時計での poll 検知（0.25s 粒度）でしか取れない。したがって **iris_wait と attestation_to_mint の
「分割点」には PC 時計 skew ＋ poll 粒度の不確かさが乗る**。一方、両区間の**「和」= t3 − t1 は
純チェーン時刻で正確**（dep 側も `iris_wait+attestation_to_mint+credit_wait = t3−t1` で同構造）。
→ 解析では「分割点」を点推定として過信せず、和（chain系E2E）と、非発動行の iris_wait 分布を主に用いる。

---

## 6. 実測上の既知事実（2026-06-04 時点）

- withdraw の `finalityThresholdExecuted` は **2000（Standard/Finalized）**（公式 "Fast default" 表現と異なる）。
  HyperBFT の高速 finality により Finalized でも数秒で完結。
- withdraw の forwarding 手数料は **額非比例の固定 0.2 USDC**（`maxFee=feeExecuted=forwarding_fee=200000`）。
  着金 = amount − 0.2（test 1→0.8 / prod 5→4.8 で確認）。
- burn(t1) 捕捉は HyperEVM の `MessageSent`@`0x81D4…` を **WS `eth_subscribe` ＋ HTTP小範囲poll 併走**で
  ライブ取得（公開RPCは非アーカイブ・100req/min・WS非対応で不可、第三者 archive/WS RPC が必須）。
- `cctp_nonce` / `finalityThresholdExecuted` は CCTP V2 では burn 時点で空、Iris が attestation 時に埋める
  → mint 後（Arbitrum `MessageReceived` or Iris nonce 逆引き）に確定。
