# T4 計測インフラ構成記録（再現性・Methods 用）

CCTP V2（Arbitrum ⇄ HyperCore, USDC）の deposit / withdraw 遅延計測に用いた
RPC・WebSocket・API エンドポイントと測定点の対応を集約した再現性記録。
論文（ACM DLT 投稿）Methods 節および再現手順の一次参照とする。

> **重要（EP 失効）**: HyperEVM の live burn 捕捉に用いた **QuickNode 専用エンドポイント
> （Hyperliquid mainnet /nanoreth 系, アーカイブ + WebSocket 対応）は無料トライアルにつき
> 2026-07 に失効済み**。実測データは取得完了しており再取得は不要。再現する場合は、
> 同等の **/nanoreth 対応（アーカイブ + `eth_subscribe(logs)` 対応）の第三者 HyperEVM EP**
> （QuickNode / Alchemy / Dwellir / Chainstack 等の nanoreth 系）を `.env` の
> `HL_EVM_RPC_ARCHIVE` / `HL_EVM_WS_URL` に設定すれば同一手順で再現可能。

---

## 1. 測定点（t0–t3）とデータ取得経路

CCTP 転送の各フェーズ時刻を、下記の経路で物理時刻として取得する。
（詳細な列定義は [`SCHEMA.md`](SCHEMA.md)、送信ロジックは
`deposit/deposit_cctp_measure.py` / `withdraw/withdraw_cctp_measure.py` 参照）

### deposit（source=Arbitrum → dest=HyperCore, Fast）

| 測定点 | 内容 | 取得経路 |
|---|---|---|
| **t0** | PC 送信（broadcast）ローカル時刻 | `time.time_ns()`（`t0_local_send(ns)`, ns 精度の壁時計） |
| **t1** | Arbitrum burn ブロック時刻 + block number | Arbitrum 公開 RPC `eth_getTransactionReceipt` → `eth_getBlockByNumber` |
| **t2** | Iris attestation 完了時刻 | Iris API を burn_tx で 0.25s poll、`status=="complete"` 検知ローカル時刻（±0.25s） |
| **t2.5** | HyperEVM mint ブロック時刻 + block number | HyperEVM RPC で USDC `Transfer(0x0→Forwarder)` を観測 |
| **t3** | HyperCore credit ledger 時刻 | HyperCore WS `userNonFundingLedgerUpdates` 購読 |

### withdraw（source=HyperCore → dest=Arbitrum, Standard/Finalized）

| 測定点 | 内容 | 取得経路 |
|---|---|---|
| **t1** | HyperEVM burn ブロック時刻（`MessageSent(bytes)`） | **第2層 live 捕捉**: QuickNode nanoreth EP で `eth_subscribe(logs)`（主）/ アーカイブ RPC `eth_getLogs` 小範囲 poll（副） |
| **t2** | Iris attestation 完了時刻 | 捕捉した burn_tx で Iris を 0.25s poll、`status=="complete"` 検知（±0.25s） |
| **t3** | Arbitrum mint ブロック時刻 | Arbitrum 公開 RPC でイベント観測 |

> **第2層 burn 捕捉に QuickNode（/nanoreth）が必須だった理由**:
> withdraw 側の HyperEVM burn（`MessageTransmitterV2.MessageSent`）は、HyperCore→HyperEVM の
> **システム transaction（`CoreDepositWallet` 実行・gas=0）** として発火する。この種の tx は
> **公開 HyperEVM RPC（非アーカイブ・100 req/min・WS 非対応）では確実に取得できず**、
> 通常経路では欠測となる。そのため **アーカイブ + WebSocket 対応の nanoreth 系専用 EP** を用いて
> 送信前から `eth_subscribe(logs)` で live 捕捉した（未設定時は公開 RPC にフォールバックするが
> 第2層 iris_wait の成功率が著しく低下する）。実装: `withdraw/withdraw_cctp_measure.py`
> の burn 捕捉ワーカー（L495–665）、設定注記: `withdraw/config.py` L71–76。

> **実 tx による裏取り（2026-07-03 取得、QuickNode nanoreth アーカイブ EP 失効前）**:
> withdraw burn tx 1 件の receipt / transaction を実取得し、**gas=0 のシステム transaction**
> であることを確認した。
> - tx hash: `0xf68852312f526fc2c0e2966e8cc173a52e584d429af0e081b0ca643ca0c3c64a`
> - `from` = `0x2000000000000000000000000000000000000000`（**HyperCore システムアドレス**。EOA ではない）
> - `to` = `0x6B9E773128f453f5c2C60935Ee2De2Cbc5390A24`（= `CoreDepositWallet`）
> - `gasPrice` = `0x0`、**`effectiveGasPrice` = `0x0`**、`gasUsed` = `0x0`、`type` = `0x0`、`status` = `0x1`
> - block = `0x232fd74`（36962164, ts=1780546486）
>
> → burn は**手数料ゼロ・システムアドレス発火のシステム tx** であり、非アーカイブ公開 RPC で
> 確実に取得できない事実を実データで裏付ける。receipt は失効前の nanoreth アーカイブ EP で取得
> （このクエリは EP 失効後は再現不可）。

---

## 2. 使用エンドポイント一覧

URL のキー/トークン部分は伏せ字。専用 EP は `.env`（**git 追跡外**, `.gitignore` L33）から
`os.getenv` で読み込み、平文でリポジトリにコミットしていない。

| プロバイダ | チェーン / サービス | 役割 | 参照（既定値 / 環境変数） |
|---|---|---|---|
| 公開 RPC | Arbitrum One | deposit burn / withdraw mint のブロック・レシート取得 | `https://arb1.arbitrum.io/rpc`（`ARB_RPC_URL` / `ARBITRUM_HTTP_RPC`） |
| 公開 RPC | HyperEVM | deposit mint（t2.5）観測・残高確認 | `https://rpc.hyperliquid.xyz/evm`（`HL_EVM_RPC_URL`, 非アーカイブ・100 req/min・WS 非対応） |
| **QuickNode（失効）** | **HyperEVM /nanoreth** | **withdraw burn の live 捕捉（アーカイブ + WS）** | `.env`: `HL_EVM_RPC_ARCHIVE`（`https://<subdomain>.quiknode.pro/<KEY>/…` 伏せ字）/ `HL_EVM_WS_URL`（`wss://…`） |
| Hyperliquid API | HyperCore | 出金 action 送信・情報取得・credit ledger 購読 | REST `https://api.hyperliquid.xyz`（`/info`, `/exchange`）、WS `wss://api.hyperliquid.xyz/ws` |
| Circle Iris API | CCTP attestation | attestation 完了時刻（t2）取得 | `https://iris-api.circle.com`（`IRIS_API_HOST`） |
| 公開 RPC | Ethereum L1 | L1 finality エンリッチ（`analysis/enrich_l1.py`, `finality_gap.py`） | `https://ethereum-rpc.publicnode.com` |

主要コントラクト / ドメイン（`deposit/config.py`）:
- Arbitrum domain = **3**、HyperEVM domain = **19**（Iris path `/v2/messages/{source_domain}`）
- `TokenMessengerV2`（burn 送信側）= `0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d`
- `MessageTransmitterV2`（mint/`MessageSent` 発火側）= `0x81D40F21F12A8F0E3252Bccb954D722d4c464B64`

---

## 3. RPC メソッド / WebSocket チャンネル一覧

### JSON-RPC メソッド（HTTP）
- `eth_getBlockByNumber` — ブロック timestamp / number
- `eth_getTransactionReceipt` — burn/mint tx のブロック確定
- `eth_getLogs` — `MessageSent(bytes)` / USDC `Transfer` イベント（小範囲チャンク。archive の広範囲 getLogs は 413 制限回避のため分割）
- `eth_getTransactionCount` / `eth_getBalance` — nonce・残高確認

### WebSocket 購読
- **HyperEVM burn 捕捉（QuickNode nanoreth）**:
  `eth_subscribe(["logs", {"address": <MessageTransmitterV2>, "topics": [<MessageSent(bytes) topic>]}])`
  — `MessageSent(bytes) = keccak("MessageSent(bytes)")`。主経路（低レート）。
  実装: `withdraw/withdraw_cctp_measure.py` L605–609。
- **HyperCore credit ledger（Hyperliquid 公開 WS）**:
  `wss://api.hyperliquid.xyz/ws` へ `{"method":"subscribe","subscription":{"type":"userNonFundingLedgerUpdates","user":<address>}}`
  — deposit t3 / withdraw の残高変動検知。実装: `deposit/deposit_cctp_measure.py` L670–711。

---

## 4. Iris API ポーリング仕様

- **エンドポイント**: `GET https://iris-api.circle.com/v2/messages/{source_domain}?transactionHash={burnTx}`
  （deposit: source_domain=3 / withdraw: source_domain=19）
- **ポーリング間隔**: `IRIS_POLL_INTERVAL_SEC = 0.25s`（`deposit/config.py`。attestation 完了を
  最大レートで poll し物理時刻化）
- **時刻精度**: `status=="complete"` 検知時刻を t2 とし、**±0.25s（= poll 間隔）精度**の物理時刻
  として扱う（`deposit/deposit_cctp_measure.py` L550, `config.py` L101）
- 詳細は既存記述（[`SCHEMA.md`](SCHEMA.md) / 各 measure スクリプト docstring）を参照。

---

## 5. シークレット取り扱い

- API キー・専用 EP URL は **すべて `.env` 経由**（`dotenv` / `os.getenv`）で読み込み、
  ソース・ドキュメント・コミット履歴に平文で含まない。
- `.env` は `.gitignore`（L33–34: `.env`, `.env.local`）で **git 追跡外**。リポジトリ内の
  平文シークレット走査（quiknode.pro / alchemy キー / Bearer / 32hex 等）で**該当なし**を確認済み。
- 本ドキュメント内の URL もキー部分は伏せ字表記とする。

---

## 参照

- 測定スクリプト: `deposit/deposit_cctp_measure.py`, `withdraw/withdraw_cctp_measure.py`
- 設定: `deposit/config.py`, `withdraw/config.py`
- L1 finality エンリッチ / 混雑プローブ: `analysis/enrich_l1.py`, `analysis/finality_gap.py`, `analysis/congestion_probe.py`
- 列スキーマ / 設計判断: [`SCHEMA.md`](SCHEMA.md), [`DECISIONS.md`](DECISIONS.md), [`README.md`](README.md)
