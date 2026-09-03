# Latency as a Boundary Observable

**Auditing and Designing Cross-Chain Bridges from External Measurement**

Gaku Nagayoshi (Independent Researcher) and Akihiro Fujihara (Faculty of
Engineering, Chiba Institute of Technology). Submitted to *ACM Distributed
Ledger Technologies: Research and Practice* (ACM DLT), 2026.

## 1. Description

This repository contains the measurement scripts, simulation code and data for a
controlled comparison of two production bridges that carry USDC between Arbitrum
One and Hyperliquid: the Hyperliquid native bridge, operated by a validator set,
and Circle CCTP V2 Fast, operated by the issuer. The endpoints, the asset, the
latency definition and the measuring party are held fixed and only the bridge
mechanism is exchanged, so every observed difference is attributable to the trust
model. We measured 117 transfers per direction on the native bridge
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
  deposit/                      CCTP V2 Fast deposit measurement and its spec
  withdraw/                     CCTP V2 Fast withdraw measurement
  scheduler/                    unattended driver for the 210-transfer campaign
  analysis/                     post-hoc enrichment from public RPC
  queueing_sim/                 M/G/c model of the attestation layer
  mechanism_sim/                three-candidate mechanism simulation
  svb_stress/                   SVB-crisis stress multipliers
  SCHEMA.md                     column definitions for all CCTP result CSVs
  DECISIONS.md, INFRASTRUCTURE.md   design decisions and endpoints used

experiments/
  t1_deposit_finality_gap.py    post-arrival unfinalized period G, native bridge
  native_bridge_arrival_survey.py   exhaustive native-bridge event survey

result/
  deposit_latency.csv, withdraw_latency.csv        native bridge, n = 117 each
  deposit_t1_l1_enriched.csv, deposit_t1_G_*       native-bridge G
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
| `deposit_latency_measure.py` | `result/deposit_latency.csv` | Fig. 5a, Table 3, Table 5 |
| `withdraw_latency_measure.py` | `result/withdraw_latency.csv` | Fig. 5b, Table 5 |
| `t4_cctp/deposit/deposit_cctp_measure.py` | `result/T4_cctp/deposit_cctp_latency.csv` | Fig. 3, Fig. 6a, Fig. 7, Fig. 8, Fig. 11a |
| `t4_cctp/withdraw/withdraw_cctp_measure.py` | `result/T4_cctp/withdraw_cctp_latency.csv` | Fig. 6b, Table 4 |
| `t4_cctp/scheduler/scheduler.py` | drives the two scripts above | Sec. 3, measurement protocol |
| `t4_cctp/analysis/enrich_l1.py` | `result/T4_cctp/deposit_l1_enriched.csv` | input to Fig. 8, Fig. 9, Table C.1 |
| `t4_cctp/analysis/finality_gap.py` | `result/T4_cctp/finality_timeline.csv` | Fig. 11 |
| `t4_cctp/analysis/congestion_probe.py` | `result/T4_cctp/congestion_enriched.csv` | Fig. 9a |
| `experiments/t1_deposit_finality_gap.py` | `result/deposit_t1_G_hist.png`, `deposit_t1_l1_enriched.csv`, `deposit_t1_G_summary.md` | Fig. 10, the *G* row of Table 6 |
| `t4_cctp/queueing_sim/queueing_sim.py` | `result/T4_cctp/queueing_sim_v2_fig1_pooled.png`, `queueing_sim_v2_fig2_dedicated.png`, `queueing_sim_v2_results.csv` | Fig. 12, Appendix E |
| `t4_cctp/mechanism_sim/mechanism_sim.py` | `result/T4_cctp/mechanism_sim_signatures.csv`, `mechanism_sim_figure.png` | Table C.1, Fig. C.1, Appendix C |
| `t4_cctp/svb_stress/svb_stress_analysis.py` | `result/T4_cctp/svb_stress_multipliers.csv`, `svb_stress_figure.png` | Fig. A.1, Appendix A |
| `experiments/native_bridge_arrival_survey.py` | `result/native_bridge_survey/*.csv` | Sec. 4.4.4 |

The three simulation scripts use fixed random seeds and reproduce their outputs
bit for bit from the CSVs in this repository. Conceptual figures (Fig. 1-4, 13,
14) were drawn by hand. The distribution-fit figures (Fig. 5-9, 15, D.1) were
produced by ad-hoc analysis and their plotting scripts are not included; the
underlying data is published here. `result/T4_cctp/cctp_fast_standard_events.csv`
(the Appendix D census) is likewise published as data without its collector.

Requires Python 3.10+ with `web3`, `eth-account`, `requests`, `websockets`,
`pandas`, `numpy`, `scipy` and `matplotlib`.

## 4. Environment variables

Live measurement reads these from a `.env` file that is not part of this
repository. No values are published here.

| Variable | Purpose | Secret |
|---|---|---|
| `ARBITRUM_HTTP_RPC` | Arbitrum One HTTP RPC endpoint | no |
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

## 5. Licence

Code is released under the MIT Licence ([`LICENSE`](LICENSE)). Data and figures
under `result/` are released under CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).

## 6. Citation

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
