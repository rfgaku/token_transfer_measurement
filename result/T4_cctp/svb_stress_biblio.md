# SVB / USDC デペッグ 文献リスト（T4 補強サーベイ）

2023年3月 SVB 破綻に伴う USDC デペッグを扱う学術・準学術文献と、
定量値の一次ソース（規制当局レポート・ニュース）。本論文の Intro / Discussion /
§8 リスク議論での引用候補を各エントリに付す。収集日 2026-07-07。

> 注: SSRN・MDPI 本文は取得時 403（要ログイン）。書誌は検索メタデータ・
> DOI・当局/プレプリントの公開ページから確定。取得可否を各行に明記。

## A. 学術論文（査読付き・プレプリント）

| # | 書誌 | 種別 | 要旨（2–3行） | 定量データ | 引用位置候補 |
|---|---|---|---|---|---|
| 1 | **Watsky, C., Allen, J., Daud, H., Demuth, J., Little, D., Rodden, M., & Seira, A. (2024).** "Primary and Secondary Markets for Stablecoins." *FEDS Notes*, Board of Governors of the Federal Reserve System, Feb 23, 2024. | 規制当局リサーチノート（本文取得済） | SVB 破綻時のステーブルコイン一次/二次市場の挙動を比較。価格変動だけでは市場帰結を説明できず、USDC は大幅デペッグ＋時価総額$10B減の一方 DAI は同様にデペッグしつつシェア増、と示す。 | **DEX 二次市場出来高が 2023-03-11 に $20B 超（平時 $1–3B）＝約7–20×**／USDC 時価総額 −$10B（3月）／安値<90¢／3-10 に約20億USDCが二次市場から除去／Circle が SVB で$3.3B出金不能 | **§8.5 後（臨界線の歴史的文脈の主引用）／(b)ストレス倍率の権威ある出典** |
| 2 | **Ahmed, R., Aldasoro, I., & Duley, C. (2024).** "Public information and stablecoin runs." *BIS Working Papers* No. 1164, Bank for International Settlements, Jan 2024. | 国際機関ワーキングペーパー（PDF取得・本文数値は未抽出） | Circle の SVB 預託$3.3B 開示を外生的な公開情報ショックと解釈し、USDC ラン（取り付け）を分析。情報開示がステーブルコイン・ランを誘発する機構をモデル化。 | 本文数値は未抽出（要 PDF 再取得）。$3.3B 開示ショックの枠組みが中核。 | **Intro（動機＝情報ショックでの取り付け実例）／§9（TTP 信頼前提のリスク議論）** |
| 3 | **Sankaewtong, K., Kitzler, S., Haslhofer, B., & Ikeda, Y. (2026).** "Tracing Stablecoin Contagion during the USDC Depeg after the Silicon Valley Bank Collapse." *arXiv:2606.07442* [cs.CE]. | プレプリント（要旨取得・本文数値は未掲載ページ） | トランザクション水準の高粒度データでショックの伝播を追跡。USDC 関連資産は即時の価格インパクト、他は流動性チャネルとして機能する二分的な伝播、危機時に単一コインから分散ポートフォリオへの再配分を実証。 | 要旨に「surging transaction counts / larger trade volumes」の定性言及（数値は本文図表）。**トランザクション水準＝本研究のオンチェーン計測と最も近い視座** | **Intro / Discussion（オンチェーン粒度での先行研究として。二峰性・待ち行列の別角度）** |
| 4 | **Diop, P. O., Chevallier, J., & Sanhaji, B. (2024).** "Collapse of Silicon Valley Bank and USDC Depegging: A Machine Learning Experiment." *FinTech*, 3(4), 569–590. MDPI. DOI:10.3390/fintech3040030. | 査読付き（MDPI・本文403） | SVB破綻が USDC/DAI/FRAX/USDD の安定性と BTC/USDT との関係に及ぼす影響を、2022-10〜2023-11 の日次データで機械学習分析。SVB が連鎖的デペッグを誘発、影響はコイン間で不均一。 | 日次データ期間 2022-10〜2023-11。倍率の直接記載は本文図表（未抽出）。 | **§9（複数ステーブルコインの連鎖デペッグ＝TTP の裏付け資産リスク）** |
| 5 | **Kakebayashi, M. (2023).** "Potential Points of Failure for Stablecoins — Did the Silicon Valley Bank Collapse Lead to DeFi Instability?" *SSRN* Working Paper No. 4533835. | プレプリント（SSRN・本文403） | SVB 破綻が DeFi エコシステム全体に及ぼした不安定化を分析。ステーブルコインの障害点（point of failure）を体系化。 | 本文数値は未抽出（要 SSRN アクセス）。 | **§9（TTP の障害点分類の参照枠）** |

## B. 定量値の一次ソース（当局・データ・ニュース）

| # | 出典 | 定量データ（出典明記用） | 用途 |
|---|---|---|---|
| 6 | **Circle (2023).** "Circle Delivers USDC Interoperability Across Ecosystems with Mainnet Launch of Cross-Chain Transfer Protocol." Press release / PR Newswire, **2023-04-26**（PRN release 301807557）。 | **CCTP V1 メインネット稼働 = 2023-04-26**（Ethereum・Avalanche）。SVB 危機（3-10〜13）より約6週間後＝**危機当時 CCTP は不在**。 | **CCTPが当時未稼働である事実の一次確認** |
| 7 | **CoinDesk (2023-03-15).** "Decentralized Exchanges Posted Record $25B Daily Volume as USDC Depegged." | DEX 全体で **過去最高 $25B/日（2023-03-11 土）**、前高 $24.3B（2021-05）。 | (b) DEX 全体倍率の裏取り |
| 8 | **CoinDesk (2023-03-13).** "USDC Trading Dominates Record Day for DeFi Exchanges Uniswap, Curve." | **Uniswap ~$12B/24h**（週末）、**Curve ~$8B**。Uniswap 3月累計 $70B > Coinbase $49.2B。 | (b) venue 別倍率 |
| 9 | **Cointelegraph / Decrypt (2023-03-12).** "Curve Finance trading volume reaches $7B historic high after USDC depeg." | **Curve $6.03B（2023-03-11）**、うち stablecoin プールが約80%。Curve は 2023-02 時点で DEX 出来高の約9%。 | (b) スワップ特化 venue 倍率 |
| 10 | **Coin Metrics — Community API**（`community-api.coinmetrics.io/v4`, asset=usdc, metric=`TxTfrCnt`, freq=1d）。取得日 2026-07-07。 | Ethereum ERC-20 USDC の**日次転送件数** 2023-02-01〜03-31。ピーク 03-11 = 1,581,640／平時中央値 ≈ 0.63–0.82M。 | **(b) 集計ストレス倍率 X の一次データ（本研究算出）** |

## 引用計画サマリ（本論文への配置）

- **Intro（動機）**: #2（情報ショックによる取り付けの実例）＋#3（オンチェーン粒度の先行研究）。
  「TTP が前提とする裏付け資産の信頼は歴史的に揺らいだ（SVB 2023-03）」の1文。
- **§8.5 後（臨界線の歴史的文脈）**: #1（Fed の 7–20× 権威データ）＋#10（本研究の集計 1.9–2.5×）。
  実在ストレスが待ち行列臨界線に対しどこに位置したかの定量比較。
- **§9（TTP リスク議論）**: #2, #4, #5（連鎖デペッグ・障害点分類）。
  CCTP は当時未稼働（#6）＝本事例は反実仮想校正である旨も明記。
