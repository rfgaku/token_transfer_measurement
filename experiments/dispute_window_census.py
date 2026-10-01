#!/usr/bin/env python3
"""Census of native-bridge withdrawals used in Sec. 4.1.1, 4.4.1 (Table 5, lower row) and 4.4.4.

Pairs every RequestedWithdrawal with its FinalizedWithdrawal (same user and nonce) in the
Bridge2 contract event logs, and reports the request-to-finalisation time d, the delay after
the 200 s dispute window (d - 200), its tail quantiles with bootstrap CIs, the cadence of
finalisation transactions, the Arbitrum block rate, the tail by load quartile, and the
post-window delay of the authors' own 116 withdrawals (matched by finalisation tx hash).
Usage: python3 experiments/dispute_window_census.py
"""
import numpy as np, pandas as pd
D = 'result/native_bridge_survey/'
def pairs(fn):
    e = pd.read_csv(D + fn)
    r = e[e.event == 'RequestedWithdrawal']; f = e[e.event == 'FinalizedWithdrawal']
    m = r.merge(f, on=['user', 'nonce'], suffixes=('_r', '_f'))
    m['d'] = (m.block_timestamp_ms_f - m.block_timestamp_ms_r) / 1000
    return e, r, f, m
e, r, f, m = pairs('native_bridge_contract_events_2025-11.csv'); d = m.d.to_numpy(); p = d - 200
print(f'period A pairs={len(m)} mean={d.mean():.1f} sd={d.std(ddof=1):.1f} min={d.min():.0f} max={d.max():.0f}')
print(f'  post-window delay <=30 s: {(p <= 30).mean():.3f};  d>267 s: {(d > 267).mean():.4f} ({(d > 267).sum()})')
rng = np.random.default_rng(0)
bs = np.array([np.quantile(rng.choice(p, len(p)), [.99, .999, .9999]) for _ in range(500)])
for k, q in enumerate([.99, .999, .9999]):
    print(f'  post-window {q:.2%} point = {np.quantile(p, q):.0f} s  95% CI {np.percentile(bs[:, k], [2.5, 97.5]).round(0)}')
v95 = np.quantile(d, .95)
print(f'  Table 5 lower row: VaR95={v95:.0f} T99={np.quantile(d, .99):.0f} CVaR95={d[d > v95].mean():.1f} T99.99={np.quantile(d, .9999):.0f}')
ft = f.drop_duplicates('tx_hash').sort_values('block_timestamp_ms'); gaps = np.diff(ft.block_timestamp_ms.to_numpy()) / 1000
print(f'  finalisation tx gap median={np.median(gaps):.0f} s p99={np.quantile(gaps, .99):.0f} s max={gaps.max():.0f} s')
print(f'  Arbitrum blocks per second during the window: median={np.median((m.block_number_f - m.block_number_r) / m.d):.2f}')
t = m.block_timestamp_ms_r.to_numpy() / 1000; allr = np.sort(r.block_timestamp_ms.to_numpy() / 1000)
load = np.searchsorted(allr, t) - np.searchsorted(allr, t - 200)
qs = pd.qcut(load, 4, labels=False, duplicates='drop')
for k in sorted(set(qs)):
    s = qs == k; print(f'  load quartile {k}: median={np.median(d[s]):.0f} s, share >240 s={np.mean(d[s] > 240):.3f}')
w = pd.read_csv('result/withdraw_latency.csv'); w = w[w.experiment_id != 16]
own = m[m.tx_hash_f.str.lower().isin(w.arb_tx_hash.str.lower())]
me = own.user.value_counts().index[0]; own = own[own.user == me]
own = own.merge(w.assign(tx=w.arb_tx_hash.str.lower()), left_on=own.tx_hash_f.str.lower(), right_on='tx')
T = (own['arb_block_timestamp(ms)'] - own['local_broadcast_time(ns)'] / 1e6) / 1000
print(f'  own withdrawals matched={len(own)}: post-window max={(own.d - 200).max():.0f} s, '
      f'send->request mean={(T - own.d).mean():.1f} s, post-window mean={(own.d - 200).mean():.1f} s')
_, _, _, m2 = pairs('native_bridge_contract_events_2026-06.csv'); p2 = m2.d.to_numpy() - 200
print(f'period B pairs={len(m2)}: post-window 99% = {np.quantile(p2, .99):.0f} s, 99.9% = {np.quantile(p2, .999):.0f} s, max = {p2.max():.0f} s')
