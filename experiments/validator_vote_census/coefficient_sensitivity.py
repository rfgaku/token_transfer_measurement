"""Coefficient dependence and sensitivity of the native-bridge size estimate (Sec. 3.6, 4.2.1).

Reproduces the numbers added to Sec. 4.2.1 in manuscript v21 (EN v5):

  [1] N read from the same Gumbel fit (mu, eta) of the 117 deposit latencies under the
      broadcast structures of Fujihara (DLT: Research and Practice, 2025,
      doi:10.1145/3772080), with bootstrap 95% intervals (B = 2,000, seed 0):
        single broadcast            mu/eta = ln N         -> N = exp(mu/eta)        (Eq. 5)
        parallel broadcast          mu/eta = 2 ln N       -> N = exp(mu/(2 eta))
        PBFT block (1 single + 2 parallel; mu~ = 5 mu, eta~ = 3 eta)
                                    mu/eta = (5/3) ln N   -> N = exp(3 mu/(5 eta))
  [2] Insensitivity of mu/eta to the number of signers (36 deposits with vote records):
      the competition part B is replaced by the (k-th minus 1st) order statistic of n
      i.i.d. exponential responses whose rate reproduces the measured mean of B for
      n = 4, k = 3; the observation wait A and t1 - t0 are kept as measured.
      n = 4 (k = 3), 10, 24, 100 (k = floor(2n/3) + 1); 500 repetitions, seed 1.
  [3] Sensitivity of Eq. (5) to the fixed part of the observation wait: A is shifted by
      -3, 0, +5, +10 s (race removed).

Usage: python3 coefficient_sensitivity.py   (needs numpy, pandas, scipy; no network)
"""
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEP = os.path.join(HERE, "..", "..", "result", "deposit_latency.csv")
PV = os.path.join(HERE, "per_deposit_votes.csv")


def ratio(x):
    mu, eta = stats.gumbel_r.fit(x)
    return mu / eta


def main():
    L = pd.read_csv(DEP)["latency(ms)"].values / 1000.0
    mu, eta = stats.gumbel_r.fit(L)
    r = mu / eta
    print(f"== [1] Gumbel fit of 117 deposits: mu={mu:.2f} eta={eta:.2f} mu/eta={r:.3f}")
    rng = np.random.default_rng(0)
    boot = np.array([ratio(rng.choice(L, len(L))) for _ in range(2000)])
    for name, c in (("single (Eq. 5)", 1.0), ("parallel", 2.0), ("PBFT block", 5.0 / 3.0)):
        lo, hi = np.percentile(np.exp(boot / c), [2.5, 97.5])
        print(f"  {name:15s} N = {math.exp(r / c):5.1f}  bootstrap 95% [{lo:.1f}, {hi:.1f}]")

    pv = pd.read_csv(PV)
    A, B = pv.A_s.values, pv.B_s.values
    t10 = (pv.t1_ms - pv.t0_ms).values / 1000.0
    th = (1 / 3 + 1 / 2) / B.mean()          # rate giving mean(3rd - 1st of 4) = mean(B)
    rng = np.random.default_rng(1)

    def race(n, k, size):
        e = np.sort(rng.exponential(1 / th, (size, n)), axis=1)
        return e[:, k - 1] - e[:, 0]

    print("\n== [2] mu/eta of t1-t0 + A + simulated competition (36 deposits, 500 repetitions)")
    print(f"  measured L: mu/eta = {ratio(pv.L_s.values):.3f}")
    for n in (4, 10, 24, 100):
        k = 3 if n == 4 else math.floor(2 * n / 3) + 1
        v = [ratio(t10 + A + race(n, k, len(A))) for _ in range(500)]
        print(f"  n={n:3d} k={k:2d}: median mu/eta = {np.median(v):.3f}  (5-95%: {np.percentile(v, 5):.3f}-{np.percentile(v, 95):.3f})")

    print("\n== [3] shift of the fixed part of the observation wait (no competition)")
    for off in (-3, 0, 5, 10):
        rr = ratio(t10 + A + off)
        print(f"  shift {off:+3d} s: mu/eta = {rr:.3f}  Eq. (5) N = {math.exp(rr):.1f}")


if __name__ == "__main__":
    main()
