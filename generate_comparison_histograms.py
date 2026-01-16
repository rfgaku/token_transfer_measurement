"""
比較ヒストグラム生成スクリプト

実測データ vs シミュレーション結果を重ね合わせて表示
藤原先生への報告用
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 出力ディレクトリ
RESULT_DIR = "result"

# ====================================
# データ読み込み
# ====================================

# 実測データ
measured_deposit = pd.read_csv(os.path.join(RESULT_DIR, "deposit_latency.csv"))
measured_withdraw = pd.read_csv(os.path.join(RESULT_DIR, "withdraw_latency.csv"))

# シミュレーションデータ
sim_deposit = pd.read_csv(os.path.join(RESULT_DIR, "sim_trace_deposit_physics.csv"))
sim_withdraw = pd.read_csv(os.path.join(RESULT_DIR, "sim_trace_withdraw_physics.csv"))

# ====================================
# データ変換（単位統一: 秒）
# ====================================

# 実測データはミリ秒 → 秒に変換
measured_deposit_latency = measured_deposit["latency(ms)"].dropna() / 1000.0
measured_withdraw_latency = measured_withdraw["latency(ms)"].dropna() / 1000.0

# シミュレーションデータは既に秒
sim_deposit_latency = sim_deposit["t_total_latency"].dropna()
sim_withdraw_latency = sim_withdraw["t_total_latency"].dropna()

# ====================================
# 統計量計算
# ====================================

def calc_stats(data, name):
    """統計量を計算して辞書で返す"""
    return {
        "name": name,
        "n": len(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
    }

# Deposit統計
stats_measured_dep = calc_stats(measured_deposit_latency, "Measured")
stats_sim_dep = calc_stats(sim_deposit_latency, "Simulated")

# Withdraw統計
stats_measured_wd = calc_stats(measured_withdraw_latency, "Measured")
stats_sim_wd = calc_stats(sim_withdraw_latency, "Simulated")

# ====================================
# Deposit 比較ヒストグラム
# ====================================

fig, ax = plt.subplots(figsize=(12, 7))

# 実測データ（青）
ax.hist(measured_deposit_latency, bins=30, alpha=0.5, color='blue', 
        label=f"Measured (N={stats_measured_dep['n']}, μ={stats_measured_dep['mean']:.2f}s, σ={stats_measured_dep['std']:.2f}s)",
        density=True, edgecolor='darkblue')

# シミュレーション結果（オレンジ）
ax.hist(sim_deposit_latency, bins=30, alpha=0.5, color='orange',
        label=f"Simulated (N={stats_sim_dep['n']}, μ={stats_sim_dep['mean']:.2f}s, σ={stats_sim_dep['std']:.2f}s)",
        density=True, edgecolor='darkorange')

# 平均線
ax.axvline(x=stats_measured_dep['mean'], color='blue', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(x=stats_sim_dep['mean'], color='orange', linestyle='-', linewidth=2, alpha=0.8)

# タイトルとラベル
ax.set_title("Deposit Latency: Measured vs Physical Simulation (N=21)\n"
             f"Measured: μ={stats_measured_dep['mean']:.2f}s, σ={stats_measured_dep['std']:.2f}s | "
             f"Simulated: μ={stats_sim_dep['mean']:.2f}s, σ={stats_sim_dep['std']:.2f}s",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Latency (seconds)", fontsize=11)
ax.set_ylabel("Probability Density", fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 統計テキストボックス
textstr = (f"=== Statistics Comparison ===\n"
           f"Measured:\n  Mean: {stats_measured_dep['mean']:.2f}s\n  Std:  {stats_measured_dep['std']:.2f}s\n  N:    {stats_measured_dep['n']}\n\n"
           f"Simulated:\n  Mean: {stats_sim_dep['mean']:.2f}s\n  Std:  {stats_sim_dep['std']:.2f}s\n  N:    {stats_sim_dep['n']}")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
deposit_filename = os.path.join(RESULT_DIR, "comparison_deposit_physics.png")
plt.savefig(deposit_filename, dpi=150)
plt.close()
print(f"[Output] {deposit_filename}")

# ====================================
# Withdraw 比較ヒストグラム
# ====================================

fig, ax = plt.subplots(figsize=(12, 7))

# 実測データ（青）
ax.hist(measured_withdraw_latency, bins=30, alpha=0.5, color='blue',
        label=f"Measured (N={stats_measured_wd['n']}, μ={stats_measured_wd['mean']:.2f}s, σ={stats_measured_wd['std']:.2f}s)",
        density=True, edgecolor='darkblue')

# シミュレーション結果（オレンジ）
ax.hist(sim_withdraw_latency, bins=30, alpha=0.5, color='orange',
        label=f"Simulated (N={stats_sim_wd['n']}, μ={stats_sim_wd['mean']:.2f}s, σ={stats_sim_wd['std']:.2f}s)",
        density=True, edgecolor='darkorange')

# 平均線
ax.axvline(x=stats_measured_wd['mean'], color='blue', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(x=stats_sim_wd['mean'], color='orange', linestyle='-', linewidth=2, alpha=0.8)

# Dispute Period ライン（参考）
ax.axvline(x=200, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Dispute Period (200s)')

# タイトルとラベル
ax.set_title("Withdraw Latency: Measured vs Physical Simulation\n"
             f"Measured: μ={stats_measured_wd['mean']:.2f}s, σ={stats_measured_wd['std']:.2f}s | "
             f"Simulated: μ={stats_sim_wd['mean']:.2f}s, σ={stats_sim_wd['std']:.2f}s\n"
             "(Dispute Period: 200s included)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Latency (seconds)", fontsize=11)
ax.set_ylabel("Probability Density", fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 統計テキストボックス
textstr = (f"=== Statistics Comparison ===\n"
           f"Measured:\n  Mean: {stats_measured_wd['mean']:.2f}s\n  Std:  {stats_measured_wd['std']:.2f}s\n  N:    {stats_measured_wd['n']}\n\n"
           f"Simulated:\n  Mean: {stats_sim_wd['mean']:.2f}s\n  Std:  {stats_sim_wd['std']:.2f}s\n  N:    {stats_sim_wd['n']}\n\n"
           f"Dispute Period: 200s")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
withdraw_filename = os.path.join(RESULT_DIR, "comparison_withdraw_physics.png")
plt.savefig(withdraw_filename, dpi=150)
plt.close()
print(f"[Output] {withdraw_filename}")

# ====================================
# 統計サマリー出力
# ====================================

print("\n=== DEPOSIT ===")
print(f"  Measured:  N={stats_measured_dep['n']}, Mean={stats_measured_dep['mean']:.2f}s, Std={stats_measured_dep['std']:.2f}s")
print(f"  Simulated: N={stats_sim_dep['n']}, Mean={stats_sim_dep['mean']:.2f}s, Std={stats_sim_dep['std']:.2f}s")

print("\n=== WITHDRAW ===")
print(f"  Measured:  N={stats_measured_wd['n']}, Mean={stats_measured_wd['mean']:.2f}s, Std={stats_measured_wd['std']:.2f}s")
print(f"  Simulated: N={stats_sim_wd['n']}, Mean={stats_sim_wd['mean']:.2f}s, Std={stats_sim_wd['std']:.2f}s")

print("\n=== 完了 ===")
