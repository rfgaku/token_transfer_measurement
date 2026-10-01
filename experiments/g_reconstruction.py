#!/usr/bin/env python3
"""Reconstruct the post-arrival unfinalized period G from exact batch matching.

Takes the exact L2-block-to-batch mapping in result/batch_refix/batch_refix.csv
(produced by result/batch_refix/refix_batch.py via the Arbitrum NodeInterface
call findBatchContainingBlock) and derives, per transfer, the source-side safe
and hard finality times and the resulting exposure windows.

Three variants of G are emitted, all measured from the destination-ledger credit
time t3:

  G_lb   t_safe + 768 s - t3    lower bound; 768 s = 2 epochs = 64 slots x 12 s
  G_std  t_final_std - t3       epoch-aligned finalisation of the L1 block that
                                carries the batch
  G_seq  max(t_safe - t3, 0)    sequencer-level exposure only, i.e. the part of
                                the window that ends once the batch is on L1

Usage:
  python3 g_reconstruction.py result/batch_refix/batch_refix.csv \
      result/T4_cctp/deposit_cctp_latency.csv result/deposit_latency.csv \
      result/batch_refix/g_reconstruction.csv

Reads public CSVs only; no endpoint and no credentials are required.
"""
import pandas as pd, numpy as np, sys
GENESIS = 1606824023
def t_final_std(ts):
    slot = np.floor((ts - GENESIS) / 12).astype(int); ep = slot // 32
    return GENESIS + 384 * np.where(slot % 32 == 0, ep + 2, ep + 3)
b = pd.read_csv(sys.argv[1]); c = pd.read_csv(sys.argv[2]).set_index('experiment_id')
n = pd.read_csv(sys.argv[3]).set_index('experiment_id'); rows = []
for _, r in b.iterrows():
    t0 = (c.loc[r.experiment_id, 't0_local_send(ns)'] if r.bridge == 'cctp'
          else n.loc[r.experiment_id, 'local_send_time(ns)']) / 1e9
    t1, t3, ts = r.t1_ms / 1000, r.t3_ms / 1000, float(r.new_tsafe_s)
    lb = ts + 768.0; std = float(t_final_std(np.array([ts]))[0])
    rows.append(dict(bridge=r.bridge, experiment_id=r.experiment_id, t_safe_s=ts, L_s=t3 - t0,
        t1_to_safe_s=ts - t1, G_lb_s=lb - t3, G_std_s=std - t3, G_seq_s=max(ts - t3, 0.0),
        tauF_lb_s=lb - t0, tauF_std_s=std - t0))
pd.DataFrame(rows).to_csv(sys.argv[4], index=False)
