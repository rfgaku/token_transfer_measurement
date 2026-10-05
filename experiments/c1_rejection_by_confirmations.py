#!/usr/bin/env python3
"""
experiments/c1_rejection_by_confirmations.py

Rejection of a single attestation server (c = 1) from the distribution of
source-chain confirmations at attestation (Sec. 3.8, Sec. 4.4.4, Appendix C).

A queueing wait in the attestation pipeline does not show up as an extra
residual in the latency regression: while a burn waits, Arbitrum keeps
producing blocks, so the wait appears as additional confirmations at the time
of signing. We therefore predict the confirmation count under M/G/c and
compare its distribution with the measured one.

  * Input: result/T4_cctp/deposit_cctp_latency.csv only
      iris_wait(ms), arb_confirmations_at_attestation(blocks)
  * Sample: confirmations > 45 removed (4 rows) -> n = 206;
      shallow group = confirmations <= 17 (132), deep group = >= 23 (74).
  * Service time: group chosen with probability 0.63 / 0.37, then a burn is
      drawn with replacement from that group (sample_services() of
      t4_cctp/queueing_sim/queueing_sim.py, applied to row indices so the
      drawn burn's confirmation count is kept together with its iris_wait).
  * Arrivals: Poisson(lambda), FCFS, c = 1 and 2,
      lambda = 0.0503 (Arbitrum only) and 0.0688 (three chains combined).
      200,000 jobs x 5 replications, fixed seeds, first 10% discarded.
  * Predicted confirmations = confirmations of the drawn burn
      + wait / (seconds per confirmation), where seconds per confirmation is
      the OLS slope of iris_wait on confirmations over all 210 burns.
  * Two-sample KS test (scipy.stats.ks_2samp) of the 206 measured counts
      against the pooled predicted counts; share in the 19-24 band.
  * Detection limit: service times scaled by r = 0.50..1.00 (step 0.05), c = 1.
  * Occupancy threshold r from Eq. (10) (Pollaczek-Khinchine) with
      E[S] = 5.37 s, E[S^2] = 34.5 s^2 and a mean wait of 0.6 s.

No network access or credentials are needed. Results go to stdout.

Usage:
  python3 -u experiments/c1_rejection_by_confirmations.py
"""
import sys
from heapq import heapreplace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "t4_cctp" / "queueing_sim"))
from queueing_sim import (N_JOBS, N_REPS, P_FAST, P_SLOW, SEED,  # noqa: E402
                          WARMUP_FRAC, sample_services)

CSV = REPO / "result" / "T4_cctp" / "deposit_cctp_latency.csv"
COL_IRIS = "iris_wait(ms)"
COL_CONF = "arb_confirmations_at_attestation(blocks)"

CONF_MAX = 45            # rows above this are the 4 tail burns, removed
SHALLOW_MAX = 17         # shallow group: confirmations <= 17
DEEP_MIN = 23            # deep group: confirmations >= 23
BAND = (19, 24)          # between the two peaks
LAMBDAS = {"Arbitrum only": 0.0503, "three chains": 0.0688}
C_LIST = [1, 2]
R_GRID = np.round(np.arange(0.50, 1.0001, 0.05), 2)

# Eq. (10) inputs as stated in the manuscript
ES_PAPER = 5.37
ES2_PAPER = 34.5
WQ_TARGET = 0.6


def load():
    df = pd.read_csv(CSV)
    conf_all = df[COL_CONF].astype(float).to_numpy()
    iris_all = df[COL_IRIS].astype(float).to_numpy() / 1000.0
    slope, intercept = np.polyfit(conf_all, iris_all, 1)
    keep = conf_all <= CONF_MAX
    conf = conf_all[keep]
    iris = iris_all[keep]
    shallow = np.flatnonzero(conf <= SHALLOW_MAX)
    deep = np.flatnonzero(conf >= DEEP_MIN)
    return df, conf_all, slope, intercept, conf, iris, shallow, deep


def simulate_waits(lam, c, services, rng):
    """Per-job FCFS waits for Poisson(lam) arrivals and c servers."""
    arrivals = np.cumsum(rng.exponential(1.0 / lam, size=services.size))
    free = [0.0] * c
    waits = np.empty(services.size)
    for i in range(services.size):
        a = arrivals[i]
        start = a if a >= free[0] else free[0]
        waits[i] = start - a
        heapreplace(free, start + services[i])
    return waits


def predict(lam, c, r, conf, iris, shallow, deep, sec_per_conf):
    """Pooled post-warm-up predicted confirmations and waits over N_REPS."""
    pred, wq = [], []
    w0 = int(N_JOBS * WARMUP_FRAC)
    for rep in range(N_REPS):
        rng = np.random.default_rng(SEED + 1000 * c + rep)
        # mixture draw on row indices (same scheme as queueing_sim.sample_services)
        idx = sample_services(rng, N_JOBS, shallow.astype(float),
                              deep.astype(float)).astype(int)
        waits = simulate_waits(lam, c, r * iris[idx], rng)
        pred.append(conf[idx][w0:] + waits[w0:] / sec_per_conf)
        wq.append(waits[w0:])
    return np.concatenate(pred), np.concatenate(wq)


def band_share(x):
    return float(((x >= BAND[0]) & (x <= BAND[1])).mean())


def pk_r(lam, es, es2, wq):
    """r such that lam*r^2*E[S^2] / (2(1 - lam*r*E[S])) = wq (Eq. (10))."""
    a = lam * es2 / 2.0
    b = wq * lam * es
    return (-b + np.sqrt(b * b + 4.0 * a * wq)) / (2.0 * a)


def crossing(xs, ps, thr):
    """Linear interpolation (in log p) of the mean wait where p falls below thr."""
    for i in range(1, len(xs)):
        if ps[i - 1] >= thr > ps[i]:
            l0, l1 = np.log10(max(ps[i - 1], 1e-300)), np.log10(max(ps[i], 1e-300))
            t = (np.log10(thr) - l0) / (l1 - l0)
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return float("nan")


def main():
    df, conf_all, slope, intercept, conf, iris, shallow, deep = load()
    es = P_FAST * iris[shallow].mean() + P_SLOW * iris[deep].mean()
    es2 = P_FAST * (iris[shallow] ** 2).mean() + P_SLOW * (iris[deep] ** 2).mean()

    print("== Sample ==")
    print(f"rows in CSV                      : {len(df)}")
    print(f"removed (confirmations > {CONF_MAX})     : {int((conf_all > CONF_MAX).sum())}")
    print(f"n                                : {conf.size}")
    print(f"shallow (<= {SHALLOW_MAX}) / deep (>= {DEEP_MIN})    : {shallow.size} / {deep.size}")
    print(f"mean iris_wait shallow / deep [s]: {iris[shallow].mean():.3f} / {iris[deep].mean():.3f}")
    print(f"mixture E[S] / E[S^2] (0.63/0.37): {es:.3f} s / {es2:.2f} s^2")
    print(f"seconds per confirmation (OLS slope of iris_wait on confirmations, all {len(df)}): "
          f"{slope:.4f} s  (intercept {intercept:.3f} s)")
    print(f"measured share in {BAND[0]}-{BAND[1]} confirmations: {band_share(conf):.4f} "
          f"({int(((conf >= BAND[0]) & (conf <= BAND[1])).sum())}/{conf.size})")
    print(f"simulation: {N_JOBS:,} jobs x {N_REPS} reps, warm-up {WARMUP_FRAC:.0%} discarded, "
          f"seed = {SEED} + 1000*c + rep\n")

    print("== Predicted vs measured confirmations (two-sample KS) ==")
    print(f"{'lambda':>22} {'c':>2} {'rho/server':>10} {'mean wait [s]':>14} "
          f"{'KS D':>7} {'KS p':>10} {'share 19-24':>12}")
    for name, lam in LAMBDAS.items():
        for c in C_LIST:
            pred, wq = predict(lam, c, 1.0, conf, iris, shallow, deep, slope)
            ks = stats.ks_2samp(conf, pred)
            print(f"{name + f' ({lam})':>22} {c:>2} {lam * es / c:>10.3f} {wq.mean():>14.4f} "
                  f"{ks.statistic:>7.4f} {ks.pvalue:>10.2e} {band_share(pred):>12.4f}")
    print()

    print("== Detection limit: service time scaled by r, c = 1 ==")
    for name, lam in LAMBDAS.items():
        print(f"-- lambda = {lam} ({name})")
        print(f"{'r':>5} {'rho':>6} {'mean wait [s]':>14} {'KS D':>7} {'KS p':>10} {'share 19-24':>12}")
        ws, ps = [], []
        for r in R_GRID:
            pred, wq = predict(lam, 1, r, conf, iris, shallow, deep, slope)
            ks = stats.ks_2samp(conf, pred)
            ws.append(wq.mean())
            ps.append(ks.pvalue)
            print(f"{r:>5.2f} {lam * r * es:>6.3f} {wq.mean():>14.4f} {ks.statistic:>7.4f} "
                  f"{ks.pvalue:>10.2e} {band_share(pred):>12.4f}")
        print(f"mean wait at which p falls below 0.05 : {crossing(ws, ps, 0.05):.3f} s")
        print(f"mean wait at which p falls below 0.001: {crossing(ws, ps, 0.001):.3f} s\n")

    print(f"== Eq. (10): occupancy r giving a mean wait of {WQ_TARGET} s at c = 1 ==")
    print(f"E[S] = {ES_PAPER} s, E[S^2] = {ES2_PAPER} s^2")
    for name, lam in LAMBDAS.items():
        r = pk_r(lam, ES_PAPER, ES2_PAPER, WQ_TARGET)
        print(f"lambda = {lam} ({name}): r = {r:.4f}")


if __name__ == "__main__":
    main()
