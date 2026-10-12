"""Quorum-distribution diagnosis of native-bridge deposit latency (Sec. 4.1.1, 4.2.1).

Reproduces the numbers in the paper that combine the deposit latency
(result/deposit_latency.csv, 117 deposits) with the vote census in this directory:

  [1] Eq. (5), N = exp(mu/eta) from a Gumbel fit, under three time references.
  [2] The exact quorum-time distribution of He and Fujihara (ICBC 2026, eq. (14)):
      T = (M-1)-th order statistic of N-1 iid Exp(lambda/N), M = ceil(2N/3),
      fitted by maximum likelihood with N profiled over 4..40; bootstrap 95% interval;
      coefficient of variation predicted for the nominal configurations (N = 21, 24)
      and likelihood-ratio statistics against them.
  [3] 4.1.1: uniform + normal convolution vs Gumbel vs log-normal (2 parameters)
      on the 116 deposits without the largest observation (AIC).
  [4] 4.2.1: decomposition L -> A (t1 -> first vote) and B (first -> third vote);
      AIC of the same three models on A (35 deposits without the largest);
      mean B that competition alone would need under N = 4, M = 3.
  [5] 4.2.1: order of the signers (first vote, rejected fourth vote), chi-square
      test against a uniformly random order.

Usage: python3 quorum_diagnosis.py   (needs numpy, pandas, scipy; no network)
"""
import collections
import math
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
DEP = os.path.join(HERE, "..", "..", "result", "deposit_latency.csv")

d = pd.read_csv(DEP)
L = d["latency(ms)"].values / 1000.0                                          # t3 - t0
T31 = (d["hl_ledger_time(ms)"] - d["arb_block_timestamp(ms)"]).values / 1000.0  # t3 - t1
L116 = np.sort(L)[:-1]

pv = pd.read_csv(P("per_deposit_votes.csv"))
votes = pd.read_csv(P("votes_raw.csv"))
post = pd.read_csv(P("post_credit_votes.csv"))


# ---------------------------------------------------------------- [1] Eq. (5)
def eq5(x):
    mu, eta = stats.gumbel_r.fit(x)
    return math.exp(mu / eta), mu, eta


# ---------------------------------------------------------------- [2] Eq. (14)
def order_logpdf(t, n, k, th):
    """log density of the k-th order statistic of n iid Exp(rate th)."""
    t = np.asarray(t, float)
    out = np.full(t.shape, -np.inf)
    m = t > 0
    tt = t[m]
    lc = math.lgamma(n + 1) - math.lgamma(k) - math.lgamma(n - k + 1)
    out[m] = lc + math.log(th) - (n - k + 1) * th * tt + (k - 1) * np.log(-np.expm1(-th * tt))
    return out


def order_moments(n, k):
    """mean and SD of the k-th order statistic of n iid Exp(1)."""
    r = np.arange(n, n - k, -1.0)
    return (1 / r).sum(), math.sqrt((1 / r ** 2).sum())


def quorum_ll(x, N):
    """profile log-likelihood of eq. (14) for given N; returns (ll, lambda)."""
    M = math.ceil(2 * N / 3)
    n, k = N - 1, M - 1
    m = order_moments(n, k)[0]
    f = lambda lt: -order_logpdf(x, n, k, math.exp(lt)).sum()
    c = math.log(m / x.mean())
    r = optimize.minimize_scalar(f, bounds=(c - 2, c + 2), method="bounded")
    return -r.fun, math.exp(r.x) * N


NS = range(4, 41)


def quorum_best(x):
    return max((quorum_ll(x, N)[0], N) for N in NS)


def quorum_cv(N):
    M = math.ceil(2 * N / 3)
    m, s = order_moments(N - 1, M - 1)
    return s / m


# ---------------------------------------------------------------- [3][4] three-model AIC
def aic_three(x, sd_floor=0.0):
    def nll(p):
        a, w, s = p
        if w <= 0 or s <= sd_floor:
            return 1e9
        pdf = (stats.norm.cdf((x - a) / s) - stats.norm.cdf((x - a - w) / s)) / w
        return -np.log(np.maximum(pdf, 1e-300)).sum()

    r = optimize.minimize(nll, [x.min(), x.max() - x.min(), 1.0], method="Nelder-Mead",
                          options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    g = stats.gumbel_r.logpdf(x, *stats.gumbel_r.fit(x)).sum()
    ln = stats.lognorm.logpdf(x, *stats.lognorm.fit(x, floc=0)).sum()
    return dict(start=r.x[0], width=r.x[1], sd=r.x[2],
                aic_uniform_normal=6 + 2 * r.fun, aic_gumbel=4 - 2 * g, aic_lognormal2=4 - 2 * ln)


def main():
    print("== [1] Eq. (5): N = exp(mu/eta)")
    for lab, x in (("t3 - t0, 117", L), ("t3 - t1, 117", T31), ("t3 - t0, 116 (w/o max)", L116)):
        N, mu, eta = eq5(x)
        print(f"  {lab:24s} mu={mu:.2f} eta={eta:.2f} N={N:.1f}")

    print("\n== [2] Eq. (14) of He and Fujihara, M = ceil(2N/3), N profiled over 4..40")
    for lab, x in (("t3 - t0, 117", L), ("t3 - t1, 117", T31), ("t3 - t0, 116 (w/o max)", L116)):
        ll, N = quorum_best(x)
        print(f"  {lab:24s} N_hat={N} (M={math.ceil(2 * N / 3)}) ll={ll:.2f}")
    rng = np.random.default_rng(7)
    boot = [quorum_best(rng.choice(L, len(L)))[1] for _ in range(400)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap 95% interval (t3 - t0, 117; B = 400, seed 7): [{lo:.0f}, {hi:.0f}]")
    print(f"  measured CV (t3 - t0, 117): {L.std(ddof=1) / L.mean():.2f}")
    llb, Nb = quorum_best(L)
    for N in (21, 24):
        ll, _ = quorum_ll(L, N)
        print(f"  nominal N={N} M={math.ceil(2 * N / 3)}: predicted CV={quorum_cv(N):.2f}, "
              f"2*dlogL vs N_hat={2 * (llb - ll):.1f}")

    print("\n== [3] Sec. 4.1.1: 116 deposits without the largest observation")
    r = aic_three(L116)
    print(f"  uniform+normal: start={r['start']:.2f} width={r['width']:.2f} sd={r['sd']:.2f}")
    print(f"  AIC uniform+normal={r['aic_uniform_normal']:.1f} Gumbel={r['aic_gumbel']:.1f} "
          f"log-normal(2p)={r['aic_lognormal2']:.1f}")

    print("\n== [4] Sec. 4.2.1: decomposition and observation wait A")
    A, B, Ls = pv.A_s.values, pv.B_s.values, pv.L_s.values
    print(f"  36 deposits: mean L={Ls.mean():.2f} s, mean A={A.mean():.2f} s, "
          f"mean B={B.mean():.2f} s (max {B.max():.2f}), first and third vote in the same block: "
          f"{int((pv.first_block == pv.third_block).sum())}")
    print(f"  Var(A)/Var(L)={A.var(ddof=1) / Ls.var(ddof=1):.3f}  Var(B)/Var(L)={B.var(ddof=1) / Ls.var(ddof=1):.4f}  "
          f"mean(t1 - t0)={((pv.t1_ms - pv.t0_ms) / 1000).mean():.2f} s")
    A35 = np.sort(A)[:-1]
    r = aic_three(A35, sd_floor=1 / math.sqrt(12))
    print(f"  A, 35 deposits: range {A35.min():.2f}-{A35.max():.2f} s, excess kurtosis {stats.kurtosis(A35):.2f}")
    print(f"  uniform+normal (sd >= 1/sqrt(12)): start={r['start']:.2f} width={r['width']:.2f} sd={r['sd']:.2f}")
    print(f"  AIC uniform+normal={r['aic_uniform_normal']:.1f} Gumbel={r['aic_gumbel']:.1f} "
          f"log-normal(2p)={r['aic_lognormal2']:.1f}")
    # competition alone, N = 4, M = 3 (3rd of 4 iid Exp(th)): mean = (1/4+1/3+1/2)/th
    th = (1 / 4 + 1 / 3 + 1 / 2) / Ls.mean()
    print(f"  competition alone (N=4, M=3) explaining mean L {Ls.mean():.2f} s needs mean "
          f"first->third = {(1 / 3 + 1 / 2) / th:.2f} s (measured {B.mean():.2f} s)")

    print("\n== [5] Sec. 4.2.1: order of the signers")
    tgt = post[post.is_target_deposit]
    first, fourth = [], []
    for e, g in votes.groupby("experiment_id"):
        ok = g[g.tx_error.isna()].sort_values("hl_block_time_ms")
        first.append(ok.validator.iloc[0])
        rej = g[g.tx_error.notna()]
        fourth.append(rej.validator.iloc[0] if len(rej) else tgt[tgt.experiment_id == e].validator.iloc[0])
    for lab, seq in (("first vote", first), ("rejected fourth vote", fourth)):
        c = collections.Counter(seq)
        obs = [c.get(v, 0) for v in sorted(set(first) | set(fourth))]
        p = stats.chisquare(obs).pvalue
        top = c.most_common(1)[0]
        print(f"  {lab:21s} n={len(seq)} most frequent {top[0][:10]}: {top[1]}  counts={obs}  chi-square p={p:.1e}")


if __name__ == "__main__":
    main()
