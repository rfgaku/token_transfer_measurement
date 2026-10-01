#!/usr/bin/env python3
"""CCTP Fast arrival rate and value inflow used in Sec. 3.8, 4.4.2, 4.4.3 and Appendix C.

Observation windows contain a fixed number of blocks per chain, so their total duration
differs by chain. Rates are therefore computed per chain (count or value / observed time)
and then summed. Little's law: mean outstanding value = inflow x mean G
(G from result/batch_refix/g_reconstruction.csv).
Usage: python3 experiments/cctp_inflow.py
"""
import pandas as pd
e = pd.read_csv('result/T4_cctp/cctp_fast_standard_events.csv'); e['t'] = pd.to_datetime(e.block_time_utc)
span = e.groupby(['chain', 'window_id']).t.agg(lambda x: (x.max() - x.min()).total_seconds()).groupby('chain').sum()
fa = e[e['class'] == 'Fast']
lam = fa.groupby('chain').size() / span; usd = fa.groupby('chain').amount_usdc.sum() / span
print('observed hours per chain:', (span / 3600).round(1).to_dict())
print('Fast arrival rate per chain [tx/s]:', lam.round(4).to_dict(), ' total =', round(lam.sum(), 4))
print('Fast value inflow per chain [USD/s]:', usd.round(1).to_dict(), ' total =', round(usd.sum(), 1))
G = pd.read_csv('result/batch_refix/g_reconstruction.csv'); g = G[G.bridge == 'cctp'].G_lb_s
print(f'mean G = {g.mean():.1f} s -> mean outstanding = {usd.sum() * g.mean():,.0f} USD '
      f'(median/p90 G reference: {usd.sum() * g.median():,.0f} / {usd.sum() * g.quantile(.9):,.0f})')
