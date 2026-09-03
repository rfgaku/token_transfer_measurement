# T4 CCTP withdraw（HyperCore → Arbitrum）実測 — 状況報告と論点

作成: 2026-06-03 / 対象: `t4_cctp/withdraw/`

別チャット（Claude Opus 4.8）で対応方針を検討するための自己完結サマリ。

---

## 1. 目的とこれまでの経緯

- T4研究は CCTP V2 Fast Transfer のレイテンシ実測。deposit側（Arbitrum→HyperCore）は前日に本番計測完了。
- 本セッションは **withdraw側（HyperCore→Arbitrum）** の実測ファイルを新規開発。
- withdrawはCircle公式どおり、HyperCoreへ単一のEIP-712署名アクション **`sendToEvmWithData`** を `/exchange` にPOSTするだけで、HyperCore引落→HyperEVMでのCCTP burn→Iris attestation→Arbitrumで自動mint（`data="0x"`の自動forwarding）まで完結する方式。

## 2. 実装済み成果物（すべて動作確認済み）

`t4_cctp/withdraw/`:
- `config.py` … deposit config流用＋withdraw固有定数（金額/出力先/sendToEvm定数/EIP-712 types）
- `withdraw_cctp_measure.py` … 本体（署名・送信・5点計測・CSV・crash-safe・ドライラン）
- `check_balance.py` … Arb/HEVM/HC spot+perp 残高確認
- `recover_cctp.py` … forwarding失敗時の手動回収（receiveMessage）
- `__init__.py`

### 署名（確立済み・正しく動作）
- hyperliquid-python-sdk の `user_signed_payload` + `sign_inner` を再利用。
- SDKの`sign_user_signed_action`は`signatureChainId`を`0x66eee`に強制上書きするため**不使用**。Circle仕様の `signatureChainId="0xa4b1"`(=42161, Arbitrum) を自前設定。
- domain: name=`HyperliquidSignTransaction`, version=`1`, chainId=42161, verifyingContract=0x0。primaryType=`HyperliquidTransaction:SendToEvmWithData`。
- ドライランで**自己recover検証**（署名→署名者復元）が sender に一致することを確認済み。
- 本送信で **HTTP200 / `{"status":"ok"}`** を確認済み（署名・action構築は完全に正しい）。

## 3. test実送信の結果（exp_id=1, 1 USDC, 実際に送金）

送信は成功し、**約7.2秒でArbitrumに着金（0.8 USDC, net）まで完走**。CSV 1行記録済み。

| 指標 | 値 | 取得 |
|---|---|---|
| 送信 | `status:ok` | ✅ |
| **t_hc_debit**（HyperCore引落 ledger） | 1780457026926 ms | ✅ |
| **t3 Arbitrum mint** block ts | 1780457034000 ms（block 469487302） | ✅ |
| arb_mint_tx | `0xa80e57…9831eda` | ✅ |
| **amount_received** | **0.8 USDC**（1 − 0.2 forwarding − ~0 maxFee） | ✅ |
| **E2E_wit_hypercore**（t3 − t_hc_debit） | **7074 ms** | ✅ |
| **E2E_wit_wallclock**（t3 − t0） | **7184 ms** | ✅ |
| hc_debit_offset（t_hc_debit − t0） | 110.4 ms | ✅ |
| **t1 HEVM burn** | （空欄） | ❌ |
| t2 Iris attestation | （空欄） | ❌ |
| cctp_nonce / message_hash / finalityThresholdExecuted / fees | （空欄） | ❌ |
| iris_wait / attestation_to_mint / hypercore_to_burn / E2E_wit_onchain | （空欄） | ❌ |

### 重要な副次発見
- **HyperCore引落の delta.type は `"send"`**（native bridgeの`"withdraw"`ではない）。
  実delta例: `{type:"send", user:<自分>, destination:"0x2000…0000", sourceDex:"spot", token:"USDC", amount:"1.0", usdcValue:"1.0", fee:"0.0", nativeTokenFee:"0.00015167", nonce:<送信nonce>}`。
  → 金額帯マッチで正しく捕捉できた（depositのcreditも`"send"`だったのと整合）。
- Arbitrum着金は ZERO→自分へ直接mint 0.8 + ZERO→フォワーダ(0x6efa3205…)へ0.2(=forwarding手数料)。

## 4. 中段（t1/t2/nonce）が欠損した根本原因 — exhaustiveに検証

**唯一の原因＝「HyperEVM上のCCTP burn tx が公開HyperEVM RPC(`rpc.hyperliquid.xyz/evm`)で観測できない」**。
（depは自分でArbitrum burnを送るのでtx hashを保持でき、それを使ってIris pollしてt1/t2を得ていた。withdrawはburnをHyperCoreが行うためtx hashが手元に無い＝非対称。）

検証（mint時刻1780457034付近、HyperEVMブロックts対応は確認済み: blk36805165=ts1780457026≈引落、blk36805175≈mint）:

| 検証内容 | 走査範囲 | 結果 |
|---|---|---|
| `MessageSent`@MessageTransmitterV2(0x81D4) | blk 36805140–36805279 | MessageSent 2件のみ、いずれも他人（destDom=0のEthereum宛バッチ等） |
| `MessageSent` topicのみ（全コントラクト） | blk 36805160–36805199 | **0件** |
| 全ログから自アドレス(20byte)検索 | blk 36805100–36805419（約5分） | **0件** |
| USDC burn (Transfer→ZERO) | blk 36805140–36805260 | 1 USDCのburn**存在せず**（11.17/2700 USDCの他人分のみ） |
| Iris リスト/フィルタ取得 | `/v2/messages/19`（無param/recipient/destDomain） | すべて**400**（txHashかnonce必須） |

→ withdrawのCCTP burnは**HyperCore/バリデータ系の処理**で行われ、公開HyperEVM RPCの`eth_getLogs`には現れない（自分のUSDC 1枚のERC20 burnすら出ない）。したがって：
- t1（HEVM burn block ts）が取れない。
- burn txが無い → deposit同様の「burn_txでIris poll→status=complete検知」によるt2（attestation完了）の**ライブ時刻化が不可**。

## 5. 取得可能な「確実データ」（burn観測に依存しない経路）

burnを観測できなくても、以下は確実に取得できる（一部は未実装＝下記スクリプト改修で追加可能）:
1. t0（送信）、t_hc_debit（HyperCore ledger）、t3（Arbitrum mint block ts）、amount_received … **実装済み・取得済み**
2. **Arbitrum mint tx の `MessageReceived` イベント**（emitter=0x81D4, topic0=`0xff48c13e…`）から:
   - sourceDomain=19、**finalityThresholdExecuted（topic[3]）= 2000（=Standard/Finalized! Fast期待と異なる重要実測値）**
   - cctp_nonce（topic[2] = `0xdfa370b6…`）
   → **未実装**（mint tx receiptをパースすれば取れる）
3. **Iris を nonce で逆引き** `GET /v2/messages/19?nonce=<nonce>`:
   - message / attestation / message_hash / feeExecuted / maxFee / mintRecipient / amount / forwardState=COMPLETE / forwardTxHash
   → **未実装**（mint後に1回叩けば取れる。ただしattestation完了の“時刻”はライブでは取れず、mint後なので≈t3）

## 6. 論点（方針決定が必要）

研究の★核心メトリクスは **iris_wait = t2 − t1（burn→attestation = CCTP信頼層レイテンシ）** で、depositと対比したいもの。これには t1（burn）と t2（attestation完了時刻）の**ライブ計測**が必要で、両方とも「HEVM burn txの取得」に帰着する。現状その取得ができない。

### 選択肢
- **A（推奨）確実データで完結**: t_hc_debit→t3 を軸にし、Arbitrum mint tx の MessageReceived（nonce, finalityThresholdExecuted）＋ Iris nonce逆引き（attestation, fees, message_hash）で記録を充実。t1/iris_wait は「公開RPCで観測不可」と明記して空欄。スクリプトをこの方針で改修し再計測。
  - 長所: 確実・正直・即完成。E2E（hypercore起点/wallclock起点）とfinality種別・手数料・着金netは揃う。
  - 短所: burn→attestationの内訳（iris_wait）はwithdrawでは得られない（depositとの対称比較が片側のみ）。
- **B: burn tx取得をさらに追う**: 別のHyperEVM RPC/インデクサ（hyperevmscan/purrsec等のAPI）やHyperCoreの別info endpointで burn tx を引けないか調査。引ければ deposit同様にIris pollでt1/t2/iris_waitを復活できる。
  - 長所: 成功すれば対称比較が完成。
  - 短所: 成否不確実・追加調査コスト。HyperCore→burn txを結ぶ公開手段が存在するか不明。
- **C: t_hc_debit を t1 代理に**: burnはHyperCore引落とほぼ同時のシステム処理とみなし t1:=t_hc_debit とする。ただし t2（attestation完了）のライブ時刻化は依然不可のため iris_wait は近似/欠損のまま。中途半端になりやすい。

### 私の見解
- 最小実装でも **finalityThresholdExecuted=2000（Finalized）** は重要な実測（withdrawは公式の“Fast default”表現と異なり、実際にはFinalized閾値で実行され、しかもHyperEVM finalityが速いため約7秒で完結）。これは Arbitrum mint tx から確実に取れるので、**選択肢AでもこのキーデータはCSVに入る**。
- まず **A** で確実な記録を完成させ（本番200件を回せる状態にする）、**B** は並行/後追いの調査タスクとして切り出すのが現実的。

## 7. 参考アーティファクト（このリポジトリ内）
- CSV: `result/T4_cctp/test/withdraw/withdraw_cctp_test.csv`
- 生WSログ: `result/T4_cctp/test/withdraw/hc_ws_raw_1.jsonl`（HyperCore引落の生deltaを含む）
- 送信チェックポイント: `result/T4_cctp/test/withdraw/send_checkpoint_1.json`
- Arbitrum mint tx: `0xa80e578942903a0772279d8369987340768f0cb5d7833968cfa69874b9831eda`
- Iris逆引きnonce: `0xdfa370b616f8e5f05e5db4b1387abd13d950f3508796642467b15a46c089964a`（status=complete, forwardState=COMPLETE）
