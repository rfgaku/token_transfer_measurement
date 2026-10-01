#!/usr/bin/env python3
"""Native-bridge statistics reported in Sec. 4.1.1, 4.2.1 and 4.4.1 (Table 5) of the paper.

Deposit : L = hl_ledger_time - local_send_time (no clock correction; Sec. 3.3), n = 117.
          Gumbel ML fit (mu, eta); effective validator count N = exp(mu/eta) (Eq. 5);
          95% CI by non-parametric bootstrap (B = 10,000).
Withdraw: T = arb_block_timestamp - local_broadcast_time (no clock correction; Sec. 3.3).
          The latency(ms) column of result/withdraw_latency.csv is clock-offset corrected
          and is NOT used. experiment_id 16 (152.8 s, impossible under the 200 s dispute
          window; Sec. 4.1.1) is excluded, n = 116. X = T - 200 s.
Usage: python3 experiments/native_bridge_stats.py
"""
import numpy as np, pandas as pd
from scipy import stats
B, SEED = 10_000, 0
dep = pd.read_csv('result/deposit_latency.csv')
L = dep['latency(ms)'].to_numpy() / 1000
wd = pd.read_csv('result/withdraw_latency.csv')
wd = wd[wd.experiment_id != 16]
T = ((wd['arb_block_timestamp(ms)'] - wd['local_broadcast_time(ns)'] / 1e6) / 1000).to_numpy()
X = T - 200.0
rng = np.random.default_rng(SEED)
def boot(x, f):
    v = np.array([f(rng.choice(x, len(x))) for _ in range(B)]); return np.percentile(v, [2.5, 97.5])
mu, eta = stats.gumbel_r.fit(L)
print(f'deposit n={len(L)} mean={L.mean():.2f} sd={L.std(ddof=1):.2f} max={L.max():.1f}')
print(f'  Gumbel mu={mu:.2f} eta={eta:.2f}  N=exp(mu/eta)={np.exp(mu/eta):.1f}')
print('  N 95% CI', boot(L, lambda x: np.exp(np.divide(*stats.gumbel_r.fit(x)))).round(1))
for q, lab in [(0.95, 'VaR95'), (0.99, 'T99'), (0.9999, 'T99.99')]:
    print(f'  {lab} = {stats.gumbel_r.ppf(q, mu, eta):.1f}')
print(f'withdraw n={len(T)} mean={T.mean():.1f} sd={T.std(ddof=1):.1f} min={T.min():.1f}')
m, s = stats.norm.fit(X)
print(f'  X skew={stats.skew(X):.2f} exkurt={stats.kurtosis(X):.2f}')
print('  Shapiro-Wilk', stats.shapiro(X)); print('  DAgostino', stats.normaltest(X))
print('  KS', stats.kstest(X, 'norm', args=(X.mean(), X.std(ddof=1))))
osm, osr = stats.probplot(X, dist='norm', fit=True); print(f'  Q-Q R^2={osr[2]**2:.3f}')
print(f'  X 99.99% (ML normal) = {stats.norm.ppf(0.9999, m, s):.1f}',
      boot(X, lambda x: stats.norm.ppf(0.9999, *stats.norm.fit(x))).round(1))
for q, lab in [(0.95, 'VaR95'), (0.99, 'T99'), (0.9999, 'T99.99')]:
    print(f'  {lab} = {200 + stats.norm.ppf(q, m, s):.1f}')
