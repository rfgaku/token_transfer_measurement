# T4 CCTP Deposit 測定 SPEC v3

Arbitrum → HyperCore の USDC deposit（CCTP V2 Fast + Circle Forwarding Service）レイテンシ測定の確定仕様。
v2 からの主な変更（exp_id=1〜3 の実測知見を反映）を本書にまとめる。

## 1. 経路（確定）
- Arbitrum 上で `TokenMessengerV2.depositForBurnWithHook` を直接呼ぶ。
- `mintRecipient = destinationCaller = CctpForwarder(HyperEVM, 0xb21d281d…5757)`。
- `hookData` に HyperCore 受取人（magic "cctp-forward" + ver0 + len24 + recipient20 + destId4, 計56B）。
- Circle Forwarding Service が HyperEVM で `mintAndForward` を実行 → CoreDepositWallet(0x6b9e77…) 経由で HyperCore に credit。
- **実例突合（段3）**: 自動着金している本物 deposit は全件 `destinationCaller=forwarder`（ZEROではない）/ `finalityThresholdExecuted=1000(Fast)`。→ 本実装の設定は実例と一致。

## 2. maxFee（v3で確定・最重要）
CCTP には2層の手数料があり、両方を maxFee が満たさないと Fast 自動着金しない:
- **base（CCTP基本手数料）** = `ceil(amount_atomic × Fast minimumFee(bps) / 10000)`。`GET /v2/burn/USDC/fees/{src}/{dst}` の `minimumFee`（bps, 1bps=0.01%）。
- **forwardFee（Forwarding手数料）** = `GET /v2/burn/USDC/fees/{src}/{dst}?forward=true` の Fast entry `forwardFee{low,med,high}`（atomic, ガス連動で変動。≈0.22–0.24 USDC）。

確定式（C, A案）:
```
maxFee = ceil( (base + forwardFee["med"]) × 1.05 )
```
- 安全弁: `maxFee < forwardFee["med"]` または fee 取得失敗なら **送信中止**（INSUFFICIENT_FEE 配送失敗の再発防止）。
- 実測知見: forwarder は **feeExecuted = maxFee 満額**を徴収（exp_id=3: maxFee=feeExecuted=293589）。よって HyperCore credit = `amount − maxFee`。
- `maxFee=0` は Standard 落ち＋forwarding 失敗（exp_id=1）、`maxFee=716`(base のみ) も forwarding 不足で失敗（exp_id=2）。→ どちらも禁止。

## 3. credit 検知（v3で是正）
CCTP の HyperCore credit は **ネイティブブリッジの `delta.type=="deposit"` ではなく**、以下で届く（exp_id=1〜3 実証）:
- `delta.type == "send"` ∧ `delta.user == CoreDepositWallet(0x6b9e77…)` ∧ `token == "USDC"` ∧ **spot 着金** ∧ `time >= experiment_start_ms`。
- 実装: `HcCctpDepositListener`（旧 `DirectHlListener`(deposit型) では構造的に拾えない）。

## 4. 4点タイミングと CSV 列（A/B 確定）
- t1 = Arbitrum burn block timestamp（`t1_arb_burn_block_ts`）
- t2 = Iris attestation complete（local, `t2_iris_attestation_complete`）
- t2.5 = HyperEVM mint block timestamp（`t2_5_hevm_mint_block_ts`）
- t3 = HyperCore credit ledger time（`t3_hc_credit_ledger_time`）
- t0 = 測定PCの tx 送信時刻（`t0_local_send_ns`）

### E2E は起点別に2列（A・最重要）
| 列 | 定義 | 起点 | 用途 |
|---|---|---|---|
| `E2E_dep_wallclock(ms)` | `t3 − t0_local_send/1e6` | **測定PCの送信時刻** | **T1 の `latency(ms)` と同義。T1とのapple-to-apple比較用** |
| `E2E_dep_onchain(ms)` | `t3 − t1_arb_burn_block_ts` | **Arbブロックの物理TS** | PC時計・RTTを排除した物理測定用 |

### 内訳（attestation を物理時刻化し burn 起点で統一）
Iris 応答にサーバ側タイムスタンプは無い（確認済）。そこで **t2(attestation完了) を burn 確定直後からの
最大レート poll(0.25s=4req/s) で物理時刻化**する。`status` が `pending→complete` に変わった最初の検知時刻を
`t2_iris_attestation_complete`(raw) として記録し、区間計算には次のクランプ後の `t2_iris_complete_local` を使う:

- **t2 クランプ**: relayer が poll 検知より先に mint した場合（`t2_raw > t2_5`）、attestation は遅くとも t2.5 までに
  利用可能だったはずなので `t2 = min(t2_raw, t2_5)`。これは「attestation 即 mint（relay 一瞬）・信頼層が律速」を表す。

| 列 | 定義 | 意味 | 精度 |
|---|---|---|---|
| `iris_wait(ms)` | `t2_iris_complete_local − t1_arb_burn_block_ts` | **burn→attestation完了＝CCTP信頼層レイテンシ** | ±0.25s（poll間隔） |
| `attestation_to_mint(ms)` | `t2_5_hevm_mint_block_ts − t2_iris_complete_local` | attestation→HEVM mint（relay+mint実行） | ±0.25s |
| `credit_wait(ms)` | `t3_hc_credit_ledger − t2_5_hevm_mint_block_ts` | mint→HyperCore credit | CHAIN |
| `src_inclusion(ms)` | `t2_5 − t1` | burn→mint 合算（参考保持） | CHAIN |

**検算**: `iris_wait + attestation_to_mint + credit_wait == E2E_dep_onchain`（t2 が打ち消し合い telescoping で厳密一致）。
- 精度の限界（正直に明記）: t2 は poll 検知のため真値より最大 +0.25s 遅れ得る。クランプが効いた回（`t2_raw>t2_5`）は
  `iris_wait=src_inclusion` / `attestation_to_mint=0` となり、その回は「信頼層 ≤ src_inclusion・relay≈0」の上界としてのみ解釈する。
  クランプ無しの回は ±0.25s 精度の物理分解として信頼できる。t1(Arb seq clock) と t2(PC clock) の系間スキューは
  数十〜数百ms（rtt_offset 相当）あり得る点も留意。

### 列順（B）・単位
列名は T1 同様 `()` で単位付き。識別子 → 時刻/ブロック → 方式(`minFinalityThreshold_set, finalityThresholdExecuted, forward_state`) → コスト(`maxFee(usdc_atomic), feeExecuted(usdc_atomic), arb_gas_*, arb_tx_fee(eth)`) → クロスチェック(`t0_local_send(ns), rtt_offset(ms), arb_confirmations_at_attestation(blocks)`) → **【右端】内訳3列 + `E2E_dep_onchain(ms), E2E_dep_wallclock(ms)`**。
`forward_state` は credit 検知直後に Iris から取得（PENDING→SENT→COMPLETE と遷移するため、検知直後は SENT のことがある。credit 着金＝配送成功の確証）。

## 5. 出力（D3）
- **本番**: `result/T4_cctp/deposit_cctp_latency.csv`（1ファイルに `experiment_id` を連番追記。`--prod`）。中間は `result/T4_cctp/prod/`。
- **test**: `result/T4_cctp/test/deposit/deposit_cctp_test.csv`。中間は同 `test/deposit/`。
- `experiment_id` は対象 CSV を読んで連番継続（`get_next_experiment_id`、T1運用踏襲）。`--exp-id` で明示指定可。

## 6. 安全・堅牢（維持）
recipient assert / 生WSログ(hc_ws_raw) / burn即時 checkpoint / 検知 phase の crash-safe save / 10秒ハートビート / タイムアウト(Iris300s, mint180s, credit300s)。
配送が失敗しても `recover_cctp.py`（`CctpForwarder.mintAndForward` を自力実行）で全額回収可能（元本不変・HyperEVMガスのみ）。

## 7. 参照アドレス
- TokenMessengerV2(Arb): `0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d`
- CctpForwarder(HyperEVM): `0xb21d281dEDB17ae5B501f6Aa8256Fe38c4e45757`
- CoreDepositWallet: `0x6b9e773128f453f5C2c60935Ee2De2cBC5390A24`
- MessageTransmitterV2(全EVM共通): `0x81D40F21F12A8F0E3252Bccb954D722d4c464B64`
- domain: Arbitrum=3 / HyperEVM=19, finalityThreshold: Fast=1000 / Standard=2000

## 8. t2クランプ / 打ち切り（censoring）方針【決定 2026-06-04】

**deposit / withdraw 両方向とも「全件保持」とする。**

- クランプ発動行（`t2_raw > mint時刻`、すなわち公開Irisの COMPLETE 反映が forwarder の実取得より遅れた回）も
  **破棄・再試行せず、そのまま CSV に残す**（現コードの挙動どおり。CSV は `t2_iris_attestation_complete`(raw) と
  `t2_iris_complete_local`(clamped) の両値を記録済みのため、**コード変更は一切不要**）。
- 分析時、クランプ行は **打ち切り（censored）データ**として扱う：`iris_wait` は真の信頼層レイテンシの**上界値**、
  `attestation_to_mint=0` は relay≈0 の上界。生存時間解析 S(t) / 打ち切り対応推定（Kaplan–Meier 等）と整合する。
- **旧方針「本番は全件クランプ非発動で揃える」は撤回**。理由：(a) クランプ発動回だけ捨てると速い attestation が
  過剰排除され**選択バイアス**が入る、(b) 再試行は送金コスト増。全件保持＋打ち切り扱いがバイアス無く正しい。
- 根拠の実測：公開Iris の COMPLETE 反映は forwarder の実 attestation 取得より遅れることがある
  （wit prod exp_id=1 で `t2_raw` が `t3`(mint) の約4秒後＝クランプ発動。wit test exp_id=2 は非発動）。
- **将来のスケジューラに「クランプ発動による再試行」ロジックを入れないこと。**
- withdraw 版の対応列：`t2_iris_attestation_complete(ms)`(raw) / `t2_iris_complete_local(ms)`(=min(raw,t3)) /
  内訳 `iris_wait = t2_clamped − t1` / `attestation_to_mint = t3 − t2_clamped`。検算
  `hypercore_to_burn + iris_wait + attestation_to_mint == E2E_wit_hypercore` および
  `iris_wait + attestation_to_mint == E2E_wit_onchain`（telescoping で厳密一致、クランプ有無に関わらず成立）。
