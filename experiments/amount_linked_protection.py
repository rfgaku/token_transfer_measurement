#!/usr/bin/env python3
"""
experiments/amount_linked_protection.py

Amount-linked protection (Sec. 5.3 (2), Fig. 15).

Instead of assigning the deep (batch-posted) protection to a random 37.1% of
transfers, assign it to the largest 37.1% by amount. This script prints the
concentration of value, the amount threshold, the value left under shallow
protection, and the cost in waiting time and pipeline load.

  * Amounts: result/T4_cctp/cctp_fast_standard_events.csv,
      chain == 'Arbitrum One' and class == 'Fast' (n = 4,415).
  * Time from inclusion to batch posting: result/batch_refix/g_reconstruction.csv,
      bridge == 'cctp', column t1_to_safe_s.
  * Waiting time and service time of the measured deposits:
      result/T4_cctp/deposit_cctp_latency.csv (E2E_dep_wallclock(ms), iris_wait(ms),
      arb_confirmations_at_attestation(blocks)); shallow = confirmations <= 17,
      deep = 23..45 (the same groups as the queueing analysis).
  * Load: rho = lambda * E[S] / c with c = 2,
      lambda = 0.0688 (three chains combined) and 0.0503 (Arbitrum only).

No network access or credentials are needed. Results go to stdout.

Usage:
  python3 -u experiments/amount_linked_protection.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EVENTS_CSV = REPO / "result" / "T4_cctp" / "cctp_fast_standard_events.csv"
G_CSV = REPO / "result" / "batch_refix" / "g_reconstruction.csv"
DEP_CSV = REPO / "result" / "T4_cctp" / "deposit_cctp_latency.csv"

DEEP_SHARE = 0.371       # share of transfers given the deep protection
TOP_SHARE = 0.10         # top 10% by count
C = 2
LAMBDAS = {"three chains": 0.0688, "Arbitrum only": 0.0503}
SHALLOW_MAX, DEEP_MIN, CONF_MAX = 17, 23, 45


def main():
    ev = pd.read_csv(EVENTS_CSV)
    amt = ev.loc[(ev["chain"] == "Arbitrum One") & (ev["class"] == "Fast"),
                 "amount_usdc"].astype(float).to_numpy()
    amt_desc = np.sort(amt)[::-1]
    total = amt_desc.sum()
    n = amt_desc.size

    print("== Value concentration (Arbitrum One, Fast) ==")
    print(f"n transfers                         : {n:,}")
    print(f"total value [USDC]                  : {total:,.2f}")
    k10 = int(np.ceil(TOP_SHARE * n))
    print(f"value carried by top 10% by count   : {amt_desc[:k10].sum() / total:.4f} "
          f"(top {k10} transfers)")

    k = int(np.ceil(DEEP_SHARE * n))
    thr = amt_desc[k - 1]
    deep_val = amt_desc[:k].sum() / total
    print(f"\n== Deep protection for the largest {DEEP_SHARE:.1%} by amount ==")
    print(f"transfers given deep protection     : {k} of {n}")
    print(f"amount threshold [USDC]             : {thr:,.2f} "
          f"(smallest amount in the deep set; next amount {amt_desc[k]:,.2f})")
    ties = int((amt == thr).sum())
    print(f"transfers with exactly this amount  : {ties}")
    print(f"value under shallow protection      : {1.0 - deep_val:.4f}")
    print(f"value under shallow protection (random assignment, expected): {1.0 - DEEP_SHARE:.4f}")

    g = pd.read_csv(G_CSV)
    t_safe = g.loc[g["bridge"] == "cctp", "t1_to_safe_s"].astype(float).to_numpy()
    print(f"\n== Inclusion to batch posting (g_reconstruction.csv, cctp, n = {t_safe.size}) ==")
    print(f"t1_to_safe_s median / mean [s]      : {np.median(t_safe):.2f} / {t_safe.mean():.2f}")

    d = pd.read_csv(DEP_CSV)
    conf = d["arb_confirmations_at_attestation(blocks)"].astype(float).to_numpy()
    e2e = d["E2E_dep_wallclock(ms)"].astype(float).to_numpy() / 1000.0
    iris = d["iris_wait(ms)"].astype(float).to_numpy() / 1000.0
    shallow = conf <= SHALLOW_MAX
    deep = (conf >= DEEP_MIN) & (conf <= CONF_MAX)
    print(f"\n== Waiting time (deposit_cctp_latency.csv, E2E_dep_wallclock) ==")
    print(f"shallow / deep / all rows           : {shallow.sum()} / {deep.sum()} / {conf.size}")
    med_deep = np.median(e2e[deep])
    mean_all = e2e.mean()
    mean_deep_iris = iris[deep].mean()
    print(f"deep transfers, median wait [s]     : {med_deep:.2f} -> "
          f"{med_deep + np.median(t_safe):.2f} (median + median t1_to_safe_s)")
    new_mean = mean_all - DEEP_SHARE * mean_deep_iris + DEEP_SHARE * t_safe.mean()
    print(f"overall mean wait [s]               : {mean_all:.2f} -> {new_mean:.2f} "
          f"(= {mean_all:.2f} - {DEEP_SHARE} x {mean_deep_iris:.2f} "
          f"+ {DEEP_SHARE} x {t_safe.mean():.2f})")

    es_now = (1 - DEEP_SHARE) * iris[shallow].mean() + DEEP_SHARE * mean_deep_iris
    es_new = (1 - DEEP_SHARE) * iris[shallow].mean() + DEEP_SHARE * t_safe.mean()
    print(f"\n== Service time and load ==")
    print(f"mean service time per transfer [s]  : {es_now:.2f} -> {es_new:.2f} "
          f"(= {1 - DEEP_SHARE:.3f} x {iris[shallow].mean():.2f} "
          f"+ {DEEP_SHARE} x {t_safe.mean():.2f})")
    for name, lam in LAMBDAS.items():
        print(f"rho at c = {C}, lambda = {lam} ({name}): "
              f"{lam * es_now / C:.3f} -> {lam * es_new / C:.3f}")


if __name__ == "__main__":
    main()
