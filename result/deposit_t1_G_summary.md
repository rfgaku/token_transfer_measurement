# T1 (Native Bridge) deposit — 露出窓 G の再構成

生成: `t1_deposit_finality_gap.py` / 2026-07-30T14:10:29

## 方法 (T4 CCTP deposit と同一)

T4 の `t4_cctp/analysis/enrich_l1.py` + `finality_gap.py` の突合ロジックを流用:

1. `anchor_l1_block` = ロック TX の L2 ブロック時刻 t_1 を含む**実 L1 ブロック**
   (T4 の `burn_rt_l1_block` と同一定義。L1 block timestamp の二分探索で逆引き)
2. `t_safe` = `anchor_l1_block` 以降で最初に `SequencerBatchDelivered` が出た
   L1 ブロックの timestamp (SequencerInbox `0x1c479675ad559DC151F6Ec7ed3FbF8ceE79582B6`)
3. `t_hard` = `t_safe` + 768s (2 epoch = 64 slot x 12s の hard finality 下界)
4. `G` = `t_hard` − t_3 (t_3 = `hl_ledger_time(ms)`/1000)

注: T4 と同じく「anchor 以降の最初のバッチ」を採るため t_safe は**下界**
(バッチ本体の L2 ブロック範囲デコードは行っていない = T4 と同条件)。

## 突合結果

- n = 117
- 突合成功 = **117/117**
- 失敗 0 件

## G の分布 [秒]

| 指標 | n | mean | median | q90 | min | max | std |
|---|---|---|---|---|---|---|---|
| T1 deposit G | 117 | 795.0 | 782.3 | 835.9 | 743.5 | 1092.9 | 56.2 |
| T1 deposit L_wallclock | 117 | 8.7 | 8.5 | 12.5 | 2.6 | 47.3 | 4.8 |
| T1 deposit L_onchain (t_3−t_1) | 117 | 9.2 | 8.9 | 13.1 | 3.7 | 49.1 | 4.9 |
| T4 CCTP deposit G (参考) | 210 | 811.6 | 805.0 | 875.1 | 739.0 | 926.0 | 43.9 |

- 全件 G > 0 か: **YES**

## T4 (CCTP deposit) との比較

| 指標 | T1 Native Bridge | T4 CCTP Fast | 差 (T1−T4) |
|---|---|---|---|
| n | 117 | 210 | — |
| G mean [s] | 795.0 | 811.6 | -16.6 |
| G median [s] | 782.3 | 805.0 | -22.7 |
| G q90 [s] | 835.9 | 875.1 | -39.2 |
| G min [s] | 743.5 | 739.0 | +4.5 |
| G max [s] | 1092.9 | 926.0 | +166.9 |
| G std [s] | 56.2 | 43.9 | +12.3 |

### 差の要因分解

定義から恒等的に

```
G = t_hard − t_3 = 768 + (t_safe − t_1) − (t_3 − t_1)
               = 768 + バッチ投稿遅延 − L_onchain
```

(全 117 件で残差 max = 0.000s → 恒等式の数値検算 OK)

| 項 | T1 median [s] | T4 median [s] |
|---|---|---|
| 定数 (2 epoch) | 768.0 | 768.0 |
| バッチ投稿遅延 t_safe−t_1 | +19.0 | +45.0 |
| −L_onchain (t_3−t_1) | -8.9 | -7.0 |
| **G** | **782.3** | **805.0** |

→ T1 の G が T4 より約 23s 小さい主因は **当時 (2025-11〜12) の Sequencer バッチ投稿間隔が T4 計測時 (2026-06) より短かったこと** (t_safe−t_1 の中央値 19s vs 45s)。
L_onchain の差 (T1 は Native Bridge の validator 署名で t_3 が数秒遅い) は逆方向に効くが、バッチ投稿遅延の差の方が支配的。いずれも 768s 定数に対して数 % の摂動で、**G ≈ 780〜810s という水準は両方式で同一**。

### 裾 (T4 の G max = 926s を超える件)

- 件数: 6 / 117

| id | G [s] | t_safe−t_1 [s] | anchor_l1 | batch_l1 |
|---|---|---|---|---|
| 81 | 1092.9 | 343 | 23940265 | 23940292 |
| 54 | 1009.5 | 251 | 23931209 | 23931230 |
| 55 | 993.1 | 234 | 23931210 | 23931230 |
| 68 | 989.3 | 229 | 23936906 | 23936925 |
| 69 | 969.4 | 213 | 23936908 | 23936925 |
| 70 | 950.6 | 193 | 23936909 | 23936925 |

いずれもバッチ投稿間隔が一時的に伸びた区間 (anchor→batch の L1 ブロック差が大きい) に該当し、G の裾はバッチ投稿の待ち時間そのもの。突合ミスではない。

## t_1 → t_safe 間隔 (バッチ投稿遅延)

| 指標 | n | mean | median | q90 | min | max | std |
|---|---|---|---|---|---|---|---|
| T1 t_1→t_safe [s] | 117 | 36.2 | 19.0 | 80.4 | -15.0 | 343.0 | 56.9 |
| T4 t_1→t_safe [s] (参考) | 210 | 52.2 | 45.0 | 116.1 | -11.0 | 164.0 | 44.1 |

- T1 中央値 = **19.0s** / T4 中央値 = 45.0s
- t_safe < t_1 (arb_block_timestamp) の件数: **20**
  **これは突合ミスではない**: anchor は「t_1 を含む L1 ブロック」なので、
  その anchor ブロック自身がバッチを載せていた場合 (下表の `anchor==safe_blk` が True)、
  t_safe = ブロック開始時刻 ≤ t_1 となり差は必ず負になる。
  下限は当該 L1 ブロックの生成間隔 (通常 12s、missed slot があれば 24s 以上)。
  T4 でも同様に `safe_lag_s` の最小値は負 (-11s)。
  ※ anchor より前のバッチを採ってしまった件 (anchor != safe_blk かつ diff<0) は **0 件** = 真の突合ミスなし。
  - id=8 t1=1764335794 t_safe=1764335783 diff=-11s (anchor=23897410 == safe_blk? True)
  - id=22 t1=1764489724 t_safe=1764489719 diff=-5s (anchor=23910153 == safe_blk? True)
  - id=35 t1=1764651177 t_safe=1764651167 diff=-10s (anchor=23923519 == safe_blk? True)
  - id=38 t1=1764651224 t_safe=1764651215 diff=-9s (anchor=23923523 == safe_blk? True)
  - id=40 t1=1764651269 t_safe=1764651263 diff=-6s (anchor=23923527 == safe_blk? True)
  - id=43 t1=1764737024 t_safe=1764737015 diff=-9s (anchor=23930612 == safe_blk? True)
  - id=53 t1=1764744246 t_safe=1764744239 diff=-7s (anchor=23931207 == safe_blk? True)
  - id=59 t1=1764746189 t_safe=1764746183 diff=-6s (anchor=23931369 == safe_blk? True)
  - id=72 t1=1764823742 t_safe=1764823727 diff=-15s (anchor=23937557 == safe_blk? True)
  - id=75 t1=1764823799 t_safe=1764823787 diff=-12s (anchor=23937560 == safe_blk? True)

## 出力

- `result/deposit_t1_l1_enriched.csv` (117 行)
- `result/deposit_t1_G_hist.png` (T1/T4 の G ヒストグラム重ね描き, bin=12s)
- 実行ログ: `result/deposit_t1_G_run.log`

## 実行統計 (RPC)

```
{
  "rpc_calls": 116,
  "retries": 1,
  "rate_limited": 0,
  "endpoint_failover": 0,
  "chunk_shrinks": 0,
  "getlogs_calls": 48,
  "getblock_calls": 67
}
```

