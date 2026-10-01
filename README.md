# Latency as a Boundary Observable

**Auditing and Designing Cross-Chain Bridges from External Measurement**

Gaku Nagayoshi (Independent Researcher) and Akihiro Fujihara (Faculty of
Engineering, Chiba Institute of Technology). Manuscript prepared for submission
to *ACM Distributed Ledger Technologies: Research and Practice* (ACM DLT).

## 1. Description

This repository contains the measurement scripts, simulation code and data for a
controlled comparison of two production bridges that carry USDC between Arbitrum
One and Hyperliquid: the Hyperliquid native bridge, operated by a validator set,
and Circle CCTP V2 Fast, operated by the issuer. The endpoints, the asset, the
latency definition and the measuring party are held fixed and only the bridge
mechanism is exchanged, so the observed differences are mainly attributable to
the trust model. We measured 117 transfers per direction on the native bridge
(2025-11-27 to 2025-12-08) and 210 per direction on CCTP V2 Fast
(2026-06-02 to 2026-06-12), all funded by the authors and all traceable to public
transaction hashes, together with an exhaustive survey of native-bridge events
over the same windows and a census of 12,239 CCTP transfers across three chains
(2026-06-13 to 2026-06-25). The repository also contains the queueing,
mechanism-candidate and stress simulations used in the appendices.

## 2. Repository layout

```
deposit_latency_measure.py      native bridge, deposit  (Arbitrum -> Hyperliquid)
withdraw_latency_measure.py     native bridge, withdraw (Hyperliquid -> Arbitrum)

t4_cctp/
  deposit/                      CCTP V2 Fast deposit measurement
  withdraw/                     CCTP V2 Fast withdraw measurement
  scheduler/                    unattended driver for the 210-transfer campaign
  analysis/                     post-hoc enrichment from public RPC
  queueing_sim/                 M/G/c model of the attestation layer
  mechanism_sim/                three-candidate mechanism simulation
  svb_stress/                   SVB-crisis stress multipliers
  SCHEMA.md                     column definitions for all CCTP result CSVs

experiments/
  g_reconstruction.py           post-arrival unfinalized period G, both bridges
  t1_deposit_finality_gap.py    earlier G reconstruction, native bridge (superseded)
  native_bridge_arrival_survey.py   exhaustive native-bridge event survey
  native_bridge_stats.py        native-bridge fits, N, and tail risk metrics

result/
  deposit_latency.csv, withdraw_latency.csv        native bridge, n = 117 each
  batch_refix/                                     exact batch matching and G
  deposit_t1_l1_enriched.csv, deposit_t1_G_*       earlier native-bridge G (superseded)
  T4_cctp/                                         CCTP measurements and simulations
  native_bridge_survey/                            exhaustive survey output
```

Analysis and simulation scripts require no credentials and run offline from the
published CSVs; only the live measurement scripts need an endpoint and a funded
wallet. Every measurement script defaults to a dry run and requires an explicit
`--broadcast` flag to send anything. Running them moves real funds on mainnet.

## 3. Script to figure/table map

| Script | Output | Paper item |
|---|---|---|
| `deposit_latency_measure.py` | `result/deposit_latency.csv` | Fig. 5a, Table 3, Table 5, Table 6 |
| `withdraw_latency_measure.py` | `result/withdraw_latency.csv` | Fig. 5b, Table 5, Table 6 |
| `t4_cctp/deposit/deposit_cctp_measure.py` | `result/T4_cctp/deposit_cctp_latency.csv` | Fig. 6a, Fig. 8, Fig. 9, Table 6, Appendix A |
| `t4_cctp/withdraw/withdraw_cctp_measure.py` | `result/T4_cctp/withdraw_cctp_latency.csv` | Fig. 6b, Table 4, Table 6 |
| `t4_cctp/scheduler/scheduler.py` | drives the two scripts above | Sec. 3, measurement protocol |
| `t4_cctp/analysis/congestion_probe.py` | `result/T4_cctp/congestion_enriched.csv` | Fig. 9a |
| `result/batch_refix/refix_batch.py` | `result/batch_refix/batch_refix.csv`, `summary.txt` | exact *t*<sub>safe</sub> behind Fig. 10, Fig. 11, the *G* row of Table 6 |
| `experiments/g_reconstruction.py` | `result/batch_refix/g_reconstruction.csv` | *G* and *τ*<sub>F</sub> for Fig. 10, Fig. 11, Table 6 |
| `t4_cctp/mechanism_sim/mechanism_sim.py` | `result/T4_cctp/mechanism_sim_signatures.csv`, `mechanism_sim_figure.png` | Table B.1, Fig. B.1, Appendix B |
| `t4_cctp/queueing_sim/queueing_sim.py` | `result/T4_cctp/queueing_sim_v2_fig1_pooled.png`, `queueing_sim_v2_fig2_dedicated.png`, `queueing_sim_v2_results.csv` | Fig. 12, Appendix D |
| `t4_cctp/svb_stress/svb_stress_analysis.py` | `result/T4_cctp/svb_stress_multipliers.csv`, `svb_stress_figure.png` | Sec. 4.4.4, settlement-layer stress multipliers |
| `experiments/native_bridge_arrival_survey.py` | `result/native_bridge_survey/*.csv` | Sec. 4.4.4, arrival rates |
| `t4_cctp/analysis/enrich_l1.py` | `result/T4_cctp/deposit_l1_enriched.csv` | input to the earlier *t*<sub>safe</sub> matching (superseded; Sec. 4.2 below) |
| `t4_cctp/analysis/finality_gap.py` | `result/T4_cctp/finality_timeline.csv` | superseded by `result/batch_refix/` |
| `experiments/t1_deposit_finality_gap.py` | `result/deposit_t1_l1_enriched.csv`, `deposit_t1_G_hist.png`, `deposit_t1_G_summary.md` | superseded by `result/batch_refix/` |
| `experiments/native_bridge_stats.py` | printed statistics | Sec. 4.1.1, Sec. 4.2.1 (N and its CI, Fig. 7), Sec. 4.4.1, Table 5 |

The three simulation scripts use fixed random seeds and reproduce their outputs
bit for bit from the CSVs in this repository. Conceptual figures (Fig. 1-4, 13,
14) were drawn by hand. Appendix A is analytical and needs no script: it uses the
`iris_wait(ms)` and `arb_confirmations_at_attestation(blocks)` columns of
`result/T4_cctp/deposit_cctp_latency.csv` only.

No plotting script is included for Fig. 5, Fig. 6, Fig. 7, Fig. 8, Fig. 9b or
Fig. 15; the underlying data is published here, and the numbers in Fig. 5 and
Fig. 7 are reproduced by `native_bridge_stats.py` (bootstrap interval endpoints
may differ by ±0.1 from the printed values owing to resampling).
`result/T4_cctp/cctp_fast_standard_events.csv` (the Appendix C census, and the
source of Fig. 15) is likewise published as data without its collector.

Requires Python 3.10+ with `web3`, `eth-account`, `requests`, `websockets`,
`pandas`, `numpy`, `scipy` and `matplotlib`.

## 4. Data dictionary notes

`t4_cctp/SCHEMA.md` defines every column of the two CCTP result CSVs. Three
columns carry names that do not match the interval naming used in the paper, one
quantity has been recomputed since the earlier release, and one column is a
corrected series that the paper does not use; all three are recorded here so that
the published numbers can be traced without re-deriving them.

### 4.1 CCTP columns whose name differs from the paper's interval

In `result/T4_cctp/deposit_cctp_latency.csv`:

| Column | What it actually contains | Not to be read as |
|---|---|---|
| `src_inclusion(ms)` | *t*<sub>m</sub> − *t*<sub>1</sub>, i.e. destination mint minus source burn-block time | the *t*<sub>0</sub> → *t*<sub>1</sub> interval of Sec. 3.4 |
| `rtt_offset(ms)` | the raw difference *t*<sub>1</sub> − *t*<sub>0</sub> | a correction that has been applied to any other column |
| `iris_wait(ms)` | `t2_iris_complete_local(ms)` − *t*<sub>1</sub> | `t2_iris_attestation_complete(ms)` − *t*<sub>1</sub> |

The first two hold for all 210 rows. The two *t*<sub>2</sub> columns agree on 190
of 210 rows and differ by at most 2.647 s on the rest, because the public Iris
`COMPLETE` timestamp can lag the forwarder's own observation of completion;
`attestation_to_mint(ms)` is computed from the same local *t*<sub>2</sub> as
`iris_wait(ms)`. Both *t*<sub>2</sub> columns are published, so either convention
can be reconstructed.

### 4.2 *t*<sub>safe</sub> and *G*: current and superseded values

The current values live in `result/batch_refix/`. `refix_batch.py` asks the
Arbitrum node which batch actually contains a given L2 block, by calling
`findBatchContainingBlock(uint64)` on the NodeInterface precompile
(`0x00000000000000000000000000000000000000C8`), and then locates that batch
sequence number in the `SequencerBatchDelivered` log of the Arbitrum One
SequencerInbox (`0x1c479675ad559DC151F6Ec7ed3FbF8ceE79582B6`) on Ethereum
mainnet. The match succeeded for all 327 deposits (210 CCTP, 117 native);
`summary.txt` holds the consistency checks. `experiments/g_reconstruction.py`
turns that into *G* and *τ*<sub>F</sub> per transfer.

These files hold the **earlier** values, obtained in the earlier release with a
heuristic (superseded; Sec. 3.5 of the paper now describes the exact matching)
— the first `SequencerBatchDelivered` at or after the L1 block that
contains *t*<sub>1</sub>, which is a lower bound on *t*<sub>safe</sub> rather than
the batch that carries the transfer:

- `result/T4_cctp/finality_timeline.csv` — `t_safe_ms`, `safe_l1_block`,
  `G_lb_ms`, `G_epoch_ms`, `tau_F_lb_ms`, `safe_lag_s`
- `result/deposit_t1_l1_enriched.csv` — `batch_l1_block`, `batch_l1_ts`,
  `t_hard_ts`, `G_seconds`, `t1_to_safe_seconds`, and the derived
  `result/deposit_t1_G_summary.md` and `deposit_t1_G_hist.png`

In 42 of the 327 deposits (22 CCTP, 20 native) that heuristic returns a
*t*<sub>safe</sub> that precedes *t*<sub>1</sub>, which the exact matching
resolves; `summary.txt` lists all 42. The `G_epoch_ms` column of
`finality_timeline.csv` is not used in the paper. The `direction == 'withdraw'`
rows of `finality_timeline.csv` are unaffected: *G* = 0 there follows from
HyperBFT finality and uses no batch matching.

### 4.3 Native-bridge withdraw latency

The `latency(ms)` column of `result/withdraw_latency.csv` is clock-offset
corrected and is not used in the paper. The paper uses the uncorrected latency
`arb_block_timestamp(ms) − local_broadcast_time(ns)/1e6` (Sec. 3.3) and excludes
experiment_id 16 (152.8 s, which is impossible under the 200 s dispute window;
Sec. 4.1.1), leaving n = 116.

## 5. Environment variables

Live measurement reads these from a `.env` file that is not part of this
repository. No values are published here. Post-hoc analysis
(`result/batch_refix/refix_batch.py`) needs an Arbitrum One endpoint and an
Ethereum endpoint that serves archive-range `eth_getLogs`; both can be public and
keyless, and the script falls back across a list of public endpoints.

| Variable | Purpose | Secret |
|---|---|---|
| `ARBITRUM_HTTP_RPC` | Arbitrum One HTTP RPC endpoint | no |
| `ETH_RPC_URL` | Ethereum mainnet RPC endpoint, optional override for the batch matching | no |
| `ARB_CHAIN_ID` | Arbitrum chain id | no |
| `ARB_SENDER_ADDRESS` | measurement wallet address on Arbitrum | no |
| `ARB_SENDER_PRIVATE_KEY` | signing key; needed only with `--broadcast` | **yes** |
| `ARB_USDC_ADDRESS` | USDC contract on Arbitrum | no |
| `ARB_TOKEN_MESSENGER`, `ARB_CCTP_TOKEN_MESSENGER` | CCTP TokenMessenger contracts | no |
| `ARB_CCTP_DEST_DOMAIN` | CCTP destination domain (HyperEVM = 19) | no |
| `ARB_CCTP_AMOUNT_USDC` | per-transfer notional for deposit | no |
| `ARB_CCTP_DRY_RUN` | dry-run guard; `1` never broadcasts | no |
| `HL_EVM_RPC_ARCHIVE` | HyperEVM archive RPC used to capture the withdraw burn | **yes if the URL embeds a key** |
| `HL_EVM_WS_URL` | HyperEVM WebSocket endpoint | **yes if the URL embeds a key** |
| `HL_USER_ADDRESS` | measurement wallet address on Hyperliquid | no |
| `HL_DEPOSIT_BRIDGE_ADDRESS` | Hyperliquid Bridge2 contract | no |
| `HL_WITHDRAW_NET_USDC` | per-transfer notional for withdraw | no |

## 6. Licence

Code is released under the MIT Licence ([`LICENSE`](LICENSE)). Data and figures
under `result/` are released under CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).

## 7. Citation

```bibtex
@article{nagayoshi2026latency,
  author  = {Nagayoshi, Gaku and Fujihara, Akihiro},
  title   = {Latency as a Boundary Observable: Auditing and Designing
             Cross-Chain Bridges from External Measurement},
  journal = {ACM Distributed Ledger Technologies: Research and Practice},
  year    = {2026},
  note    = {Code and data: \url{https://github.com/rfgaku/token_transfer_measurement}}
}
```

These measurements characterise two bridges as they behaved during the
observation windows above. They are not a security audit, not an endorsement and
not advice.
