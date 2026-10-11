# 投票者4アドレスの身元確認と着金後の票（2026-10-10）

## 1. validators_now.csv との突合
照合先: validators_now.csv の全11列（validator, signer, name, description, nRecentBlocks, stake, isJailed, unjailableAfter, isActive, commission, stake_hype）×35行。加えて、取得したままの validatorSummaries JSON 全体（description と stats を含む）も文字列で検索した。大文字と小文字は区別せず、0x を除いた40桁で照合した。

| アドレス | 一致した列（validators_now.csv） | 生JSON内での出現 |
|---|---|---|
| `0x263294039413b96d25e4173a5f7599f8b3801504` | なし | なし |
| `0xda6816df552c3f9e0fb64979fb357800d690d79b` | なし | なし |
| `0x58e1b0e63c905d5982324fcd9108582623b8132e` | なし | なし |
| `0xef2364db5db6f5539aa0bc111771a94ee47637fc` | なし | なし |

→ 4アドレスはどれも、現在（2026-10-10）のバリデータの validator 列・signer 列のどちらにも含まれない。

## 2. 着金ブロックに4票目がなかった6件：着金後15ブロックの票

取得したブロック数: 90 / 90（全件取得）
件ごとに取得したブロック数: {'23': 15, '24': 15, '32': 15, '64': 15, '101': 15, '102': 15}

### 当該送金への票（着金ブロック以前の3票と、着金後の票）

| exp | 着金前の投票者（3票） | 着金後の票: アドレス, +ブロック, +ms, error |
|---|---|---|
| 23 | `0x58e1…132e`, `0x2632…1504`, `0xda68…d79b` | `0xef23…37fc` +4 blk, +261 ms, Deposit has already been credited. |
| 24 | `0xda68…d79b`, `0x58e1…132e`, `0x2632…1504` | `0xef23…37fc` +10 blk, +928 ms, Deposit has already been credited. |
| 32 | `0xda68…d79b`, `0x2632…1504`, `0x58e1…132e` | `0xef23…37fc` +4 blk, +277 ms, Deposit has already been credited. |
| 64 | `0xda68…d79b`, `0x58e1…132e`, `0x2632…1504` | `0xef23…37fc` +4 blk, +250 ms, Deposit has already been credited. |
| 101 | `0xda68…d79b`, `0x58e1…132e`, `0x2632…1504` | `0xef23…37fc` +3 blk, +221 ms, Deposit has already been credited. |
| 102 | `0xef23…37fc`, `0x2632…1504`, `0xda68…d79b` | `0x58e1…132e` +4 blk, +405 ms, Deposit has already been credited. |

- 当該送金への着金後の票: 6 票。そのうち4アドレス以外からの票: 0 票
- 同じ範囲に含まれていた、他の送金への VoteEthDepositAction: 3 票（投票者: ['0x58e1b0e63c905d5982324fcd9108582623b8132e', '0xef2364db5db6f5539aa0bc111771a94ee47637fc']）
- 取得した範囲で見つかった、4アドレス以外の投票者: なし
- 着金後15ブロックをすべて取得した件のうち、4アドレスすべての票がそろわない件: なし

詳細: `post_credit_votes.csv`（全行。is_target_deposit, in_known_4, tx_error の列あり）

## 3. 直近の取引（explorer RPC userDetails、2026-10-10 に各アドレス1要求）

userDetails は直近300件しか返さない。生の応答は `cache/userDetails/<addr>.json` に保存してある。

| アドレス | 件数 | 期間 (UTC) | 取引の種類（件数） | VoteEthDepositAction |
|---|---|---|---|---|
| `0x263294039413b96d25e4173a5f7599f8b3801504` | 300 | 2026-10-10 09:47:40〜2026-10-10 11:19:24 | VoteEthFinalizedWithdrawalAction 72, ValidatorSignWithdrawalAction 69, voteAbciDigest 64, VoteEthDepositAction 59, voteAppHash 28, voteL1Hash 8 | 59票（送金59件、error 0票）、2026-10-10 09:47:54〜2026-10-10 11:18:04 |
| `0xda6816df552c3f9e0fb64979fb357800d690d79b` | 300 | 2026-10-10 09:46:43〜2026-10-10 11:19:24 | VoteEthFinalizedWithdrawalAction 73, ValidatorSignWithdrawalAction 71, voteAbciDigest 59, VoteEthDepositAction 59, voteAppHash 30, voteL1Hash 8 | 59票（送金59件、error 0票）、2026-10-10 09:47:54〜2026-10-10 11:18:04 |
| `0x58e1b0e63c905d5982324fcd9108582623b8132e` | 300 | 2026-10-10 11:19:28〜2026-10-10 11:20:12 | evmRawTx 285, SetGlobalAction 15 | なし（この範囲には含まれない） |
| `0xef2364db5db6f5539aa0bc111771a94ee47637fc` | 300 | 2026-10-10 09:46:52〜2026-10-10 11:20:11 | VoteEthFinalizedWithdrawalAction 73, ValidatorSignWithdrawalAction 70, VoteEthDepositAction 59, voteAbciDigest 56, voteAppHash 34, voteL1Hash 8 | 59票（送金59件、error 0票）、2026-10-10 09:47:54〜2026-10-10 11:18:04 |

- 預金投票をしている3アドレスについて、投票先の送金の和集合は59件、共通部分は59件だった。

**(a) 今も VoteEthDepositAction を出しているか**
- `0x2632…1504`、`0xda68…d79b`、`0xef23…37fc` の3つは、**今も出している**。2026-10-10 の約1.5時間で各59票。3者とも同じ59件の送金に投票していて、error はない。
- `0x58e1…132e` の直近300件は約44秒分しかなく、中身は SetGlobalAction（オラクル価格の設定に見える）15件と evmRawTx 285件だった。この範囲に預金投票はないが、件数の窓が狭すぎるので「今は投票していない」とまでは言えない。

**(b) 現在のバリデータ集合と無関係に続いているか**
- 上の3アドレスは、現在の validatorSummaries（35件）の validator 列・signer 列のどちらにも含まれない（1節）。それでも、2025-11〜12 と同じ形の預金投票を今も続けている。
- 同じ3アドレスは、出金の投票（VoteEthFinalizedWithdrawalAction、ValidatorSignWithdrawalAction）と、合意のチェックポイント投票（voteAbciDigest、voteAppHash、voteL1Hash）も出している。voteAbciDigest と voteL1Hash は、一覧にあるバリデータの signer（例：ValiDAO の `0x0000…8fde`）も出している種類の投票である。
- したがって、この3つ（と `0x58e1…132e`）は、validatorSummaries に載っている委任型バリデータ集合とは別の固定の署名者集合として、ブリッジの投票（と一部のチェックポイント投票）を継続的に担っているように見える。
- ただし、現在の送金に4者以外が投票しているかどうかは、userDetails だけでは判定できない（該当ブロックの取得が必要）。
