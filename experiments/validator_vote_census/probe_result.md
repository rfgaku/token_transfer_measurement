# 手順1：探索の結果（exp 1、2026-10-10）

## 取得手段
- (a) explorer RPC `POST https://rpc.hyperliquid.xyz/explorer` に `{"type":"blockDetails","height":h}` を送る方法：**採用した。** 応答の `blockDetails.txs[]` に各取引の `time`, `user`, `action`, `block`, `hash`, `error` が含まれる。
- (b) S3 `hl-mainnet-node-data`：`InvalidAccessKeyId` で使えなかった（`probe_s3.md` 参照）。

## 着金ブロックの特定
`deposit_latency.csv` の `hl_block_number`（exp 1 では 20306133）は HyperEVM のブロック番号で、L1 の高さではない。
L1 のこの高さを引くと、2023-04 のブロックが返る。
そこで、L1 で `blockTime == hl_ledger_time`（t3 = 1764226680841）となる高さを補間探索した。結果は **809913323** で、時刻は ms 単位まで一致した。

## 見つかった投票取引（着金ブロック 809913323 内）
```json
{"time": 1764226680841, "user": "0xef2364db5db6f5539aa0bc111771a94ee47637fc",
 "action": {"type": "VoteEthDepositAction", "user": "0x322f51d8191b7cb463c06113c527c28afde70321", "usd": 5000000,
            "ethId": {"b": 404580282, "t": 1, "e": "Deposit", "i": 0,
                      "h": "0x863bf4306897c2b46ea4e46149348d01c92bdb96eeb85d81653d06df96043eb5"}},
 "block": 809913323, "hash": "0x3f375a43e96f6db140b10430464beb010700722984628c83e3000596a863479b", "error": null}
```
同じブロックに、`0x58e1b0e63c905d5982324fcd9108582623b8132e` の票も `"error": "Deposit has already been credited."` として記録されていた。
また、809913320（t = 1764226680554）にも `0xda6816df…d79b` と `0x263294…1504` の有効票が1票ずつあった。

## 構造
- 署名者（投票者）：取引の外側の `user`。`action.user` は入金先の HL アカウント
- 時刻：取引の `time`（ms、ブロック時刻と同じ）
- `ethId`：`b` = Arbitrum のブロック番号、`h` = arb_tx_hash、`i` = ログの順番、`t`、`e` = "Deposit"
- 着金後に届いた票は取引として記録され、`error` に理由が入る
