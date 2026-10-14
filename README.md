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
  validator_vote_census/        on-chain deposit votes (VoteEthDepositAction), 36 deposits

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
| `t4_cctp/deposit/deposit_cctp_measure.py` | `result/T4_cctp/deposit_cctp_latency.csv` | Fig. 6a, Fig. 8, Fig. 9, Table 6 |
| `t4_cctp/withdraw/withdraw_cctp_measure.py` | `result/T4_cctp/withdraw_cctp_latency.csv` | Fig. 6b, Table 4, Table 6 |
| `t4_cctp/scheduler/scheduler.py` | drives the two scripts above | Sec. 3, measurement protocol |
| `t4_cctp/analysis/congestion_probe.py` | `result/T4_cctp/congestion_enriched.csv` | Fig. 9a |
| `result/batch_refix/refix_batch.py` | `result/batch_refix/batch_refix.csv`, `summary.txt` | exact *t*<sub>safe</sub> behind Fig. 10, Fig. 11, the *G* row of Table 6 |
| `experiments/g_reconstruction.py` | `result/batch_refix/g_reconstruction.csv` | *G* and *τ*<sub>F</sub> for Fig. 10, Fig. 11, Table 6 |
| `t4_cctp/mechanism_sim/mechanism_sim.py` | `result/T4_cctp/mechanism_sim_signatures.csv`, `mechanism_sim_figure.png` | Table A.1, Fig. A.1, Appendix A |
| `t4_cctp/queueing_sim/queueing_sim.py` | `result/T4_cctp/queueing_sim_v2_fig1_pooled.png`, `queueing_sim_v2_fig2_dedicated.png`, `queueing_sim_v2_results.csv` | Fig. 12, Appendix C |
| `t4_cctp/svb_stress/svb_stress_analysis.py` | `result/T4_cctp/svb_stress_multipliers.csv`, `svb_stress_figure.png` | Sec. 4.4.4, settlement-layer stress multipliers |
| `experiments/native_bridge_arrival_survey.py` | `result/native_bridge_survey/*.csv` | Sec. 4.4.4, arrival rates |
| `t4_cctp/analysis/enrich_l1.py` | `result/T4_cctp/deposit_l1_enriched.csv` | input to the earlier *t*<sub>safe</sub> matching (superseded; Sec. 4.2 below) |
| `t4_cctp/analysis/finality_gap.py` | `result/T4_cctp/finality_timeline.csv` | superseded by `result/batch_refix/` |
| `experiments/t1_deposit_finality_gap.py` | `result/deposit_t1_l1_enriched.csv`, `deposit_t1_G_hist.png`, `deposit_t1_G_summary.md` | superseded by `result/batch_refix/` |
| `experiments/native_bridge_stats.py` | printed statistics | Sec. 4.1.1, Sec. 4.2.1 (N ≈ 10 from Eq. 5 and its CI), Sec. 4.4.1, Table 5 |
| `experiments/dispute_window_census.py` | printed statistics | Sec. 4.1.1, Sec. 4.4.1 (required dispute-window length by exceedance probability), Table 5 (lower withdraw row), Sec. 4.4.4 |
| `experiments/cctp_inflow.py` | printed statistics | Sec. 3.8, Sec. 4.4.3 (value inflow and Eq. 13), Appendix B |
| `experiments/bridge2_params.py`, `experiments/bridge2_change_events.py` | `result/bridge2_params/` | Sec. 4.1.1 and Sec. 4.4.1 (disputePeriodSeconds = 200 s and blockDurationMillis = 350 ms; no change events since the start of measurement) |
| `experiments/c1_rejection_by_confirmations.py` | printed statistics | Sec. 3.8, Sec. 4.4.4 (rejection of c = 1 from the confirmation-count distribution), Appendix C (detection limit and occupancy threshold) |
| `experiments/amount_linked_protection.py` | printed statistics | Sec. 5.3 (2), Fig. 15 |
| `experiments/validator_vote_census/analysis_votes.py` | `results_votes.json`, `report_votes.md`, `per_deposit_votes.csv`, `figures/fig7a_vote_timeline.png`, `figures/fig7b_A_hist.png` | Sec. 4.2.1, Fig. 7 (deposit voter set, decomposition of deposit latency, numbers in Fig. 7) |
| `experiments/validator_vote_census/quorum_diagnosis.py` | printed statistics | Sec. 4.1.1 (uniform + normal, Gumbel and log-normal on 116 deposits), Sec. 4.2.1 (Eq. (5) and the quorum-time distribution, decomposition of deposit latency, order of the signers) |
| `experiments/validator_vote_census/coefficient_sensitivity.py` | printed statistics | Sec. 3.6, 4.2.1 (N under the single, parallel and PBFT broadcast coefficients; insensitivity of μ/η to the number of signers; sensitivity to the fixed part of the observation wait) |

The three simulation scripts use fixed random seeds and reproduce their outputs
bit for bit from the CSVs in this repository. Conceptual figures (Fig. 1-4, 13,
14) were drawn by hand.

No plotting script is included for Fig. 5, Fig. 6, Fig. 8, Fig. 9b or Fig. 15;
the underlying data is published here, and the numbers in Fig. 5 are reproduced
by `native_bridge_stats.py` (bootstrap interval endpoints may differ by ±0.1 from
the printed values owing to resampling) and those in Fig. 15 by
`amount_linked_protection.py`.
Fig. 7 is based on the vote data in `experiments/validator_vote_census/`: its
numbers are printed by `analysis_votes.py`, and the figure in the paper was drawn
by the authors from the same data. The value N ≈ 10 from Eq. (5) is produced by
`native_bridge_stats.py`.
`result/T4_cctp/cctp_fast_standard_events.csv` (the Appendix B census, and the
source of Fig. 15) is likewise published as data without its collector.

Requires Python 3.10+ with `web3`, `eth-account`, `requests`, `websockets`,
`pandas`, `numpy`, `scipy` and `matplotlib`.

## 4. Data dictionary notes

`t4_cctp/SCHEMA.md` defines every column of the two CCTP result CSVs. Three
columns carry names that do not match the interval naming used in the paper, one
quantity has been recomputed since the earlier release, and one column is a
corrected series that the paper does not use; all three are recorded here,
together with how the Appendix B observation windows are timed, so that the
published numbers can be traced without re-deriving them.

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

The `latency(ms)` column of `result/withdraw_latency.csv` differs from the
uncorrected latency in 5 of 117 rows (by up to 1.7 s; mean difference 0.06 s) and
is not used in the paper. The paper uses the uncorrected latency
`arb_block_timestamp(ms) − local_broadcast_time(ns)/1e6` (Sec. 3.3) and excludes
experiment_id 16 (152.8 s, impossible under the 200 s dispute window;
Sec. 4.1.1), leaving n = 116.

### 4.4 Appendix B observation windows

Each window spans a fixed number of blocks per chain (Arbitrum 50,000, Base
12,000, Ethereum 3,000), so the total observed time differs by chain (about
24.3 h, 46.5 h and 69.9 h). Arrival rates and value inflow are computed per chain
and then summed (`experiments/cctp_inflow.py`); the pooled row of Table B.1 is a
plain sum of counts and values.

### 4.5 Native-bridge deposit vote census

**Purpose.** To observe the native bridge's deposit quorum directly, rather than infer
it from latency. For each deposit we take from Hyperliquid L1 every
`VoteEthDepositAction` whose `ethId.h` equals the Arbitrum transaction hash. Each vote
is recorded as the signer (the outer `user` of the transaction), the L1 block and the
block time. From these we rebuild the sequence *t*<sub>1</sub> (Arbitrum block) →
individual votes → credit (*t*<sub>3</sub>).

**Data source.** The Hyperliquid explorer RPC, `POST
https://rpc.hyperliquid.xyz/explorer` with `{"type":"blockDetails","height":h}`.
`hl_block_number` in `result/deposit_latency.csv` is a HyperEVM block number, not an L1
height. The L1 credit block is therefore located as the L1 block whose `blockTime`
equals `hl_ledger_time(ms)`; this matched to the millisecond in all collected deposits
(`credit_blocks.csv`). The public S3 archive (`hl-mainnet-node-data`) was not used.

**Coverage.** 36 of the 117 deposits, covering all 11 measurement days. Deposits were
taken in a day-stratified round-robin order (`order.csv`) that includes the
largest-latency deposit (#77). From the credit block, `fetch_votes.py` walks back block
by block until 16 consecutive blocks contain no vote for the deposit; it never goes
below the height of *t*<sub>1</sub>. For the 6 deposits whose credit block held no
fourth vote, `post_credit_check.py` also reads the 15 blocks after the credit block.
Signer activity in October 2026 was checked with one `userDetails` request per address
(`cache/userDetails/`).

**API constraints.** With a fixed 2.2 s interval and a flat 10 s pause after HTTP 429,
the effective rate was about 9 blocks per minute, against the nominal
30 per minute. The explorer also enforces a daily cap on archived-block queries: it
returns HTTP 200 with `{"type":"error","message":"More than 100 archived blocks queried
in one day"}`. In practice the cap was reached after about 1,000 blocks in one day, and
it recovered the next day. Collection of the remaining 81 deposits was stopped for this
reason.

**Raw blocks not included.** The fetched blocks (`cache/blocks/`, 1,161 gzip JSON files,
86 MB) are not included. To rebuild them, run `fetch_votes.py` (resumable; it continues
from `progress.csv`, so delete it together with `votes_raw.csv` and `credit_blocks.csv`
for a full rebuild) and then `post_credit_check.py`. Expect several days because of the
daily cap. The analysis itself needs only the CSVs published here: `python3
analysis_votes.py`.

**Main results** (`report_votes.md`, `results_votes.json`):
- The voters are the same 4 addresses in all 36 deposits. None appears in the current
  `validatorSummaries` (validator or signer). Three of the four were still casting
  deposit votes in October 2026 (`identity_check.md`). The four coincide with the
  Bridge2 hot validator set described by L2BEAT (four signers with equal power; a
  request is accepted with signatures over two-thirds of the power).
- Every deposit is credited on its 3rd valid vote (N = 4, M = 3). *C* = *t*<sub>3</sub>
  − 3rd vote = 0 in 36/36 deposits.
- The 4th vote is rejected ("Deposit has already been credited."). It falls inside the
  credit block in 30 deposits and 0.22–0.93 s after credit in the other 6. No other
  address voted in the examined blocks.
- *B* = 3rd vote − first vote: mean 0.24 s, max 1.45 s; 14 deposits have the first and
  third vote in the same block.
- *A* = first vote − *t*<sub>1</sub>: mean 9.84 s. Var(*A*)/Var(*L*) = 1.02 (36
  deposits; 0.99 without #77) and Var(*B*)/Var(*L*) = 0.002. Almost all of the spread in
  deposit latency is therefore common observation delay, not quorum competition.
- Without #77, *A* = 8.74 ± 3.53 s (range 3.38–16.05 s, excess kurtosis -1.12). A
  uniform distribution plus Gaussian noise fits with start 3.57 s, width 11.16 s, SD
  1.01 s. By AIC it beats Gumbel and the two-parameter log-normal (186.4 against 189.4
  and 189.2; `report_votes.md` also lists the three-parameter log-normal). The noise SD
  is bounded below by 1/√12 s (1 s resolution of *t*<sub>1</sub>); without that bound the
  maximum lies on the pure-uniform boundary, and the ranking is unchanged.
- The order of the signers is not random: the first vote came from the same signer in
  23 of the 36 deposits, and the rejected fourth vote from a second signer in 26
  (chi-square against a random order, p = 6e-8 and 2e-9).
- The exact quorum-time distribution of He and Fujihara (ICBC 2026, eq. (14)) fitted to
  the 117 deposit latencies gives N = 8 (bootstrap 95% interval [7, 13]; 11 with
  *t*<sub>3</sub> − *t*<sub>1</sub> as the reference, 12 without the largest
  observation). The nominal configurations (N = 21, 24) predict a coefficient of
  variation of 0.27–0.29 against the measured 0.55. With N = 4, M = 3, competition alone
  would need a mean first-to-third-vote time of 7.4 s to explain the mean latency of the
  36 deposits (9.6 s), against the measured 0.24 s. The shape of the deposit latency
  therefore reflects the common observation wait, not the number of signers
  (`quorum_diagnosis.py`).
- The coefficient that links μ/η to ln N depends on the communication structure of the
  consensus (Fujihara, DLT: Research and Practice, 2025, doi:10.1145/3772080). The same
  Gumbel fit gives N = 10.0 for a single broadcast (Eq. (5)), 3.2 (bootstrap 95% interval
  [2.7, 3.8]) for a parallel broadcast, and 4.0 ([3.3, 5.0]) for one PBFT block. When the
  competition among n signers is simulated with the measured per-signer speed, μ/η of
  the latency stays within 1.91–1.94 for n = 4 to 100, whereas shifting the fixed part of
  the observation wait by −3 to +10 s moves the Eq. (5) value from 2.9 to 93.4. The
  shape therefore does not follow the number of signers under any coefficient
  (`coefficient_sensitivity.py`; the interval of Eq. (5) in the paper, [7.3, 14.6], is
  from `native_bridge_stats.py` with B = 10,000).

**Correspondence with the paper.** Sec. 4.2.1 uses this census for the deposit voter set
(N = 4, M = 3), for the decomposition of deposit latency into *A*, *B* and *C*, for the
order of the signers, for the diagnosis with the quorum-time distribution, and for the
numbers in Fig. 7; Sec. 4.1.1 uses `quorum_diagnosis.py` for the model comparison on
116 deposits. The figure in the paper was drawn by the authors from the same
data; `figures/fig7a_vote_timeline.png` and `figures/fig7b_A_hist.png` are reference
plots produced by `analysis_votes.py`. The Markdown notes in this directory
(`identity_check.md`, `probe_result.md`, `report_votes.md`) are working records in
Japanese.

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
