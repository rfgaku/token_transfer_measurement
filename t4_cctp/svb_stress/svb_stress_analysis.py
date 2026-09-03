#!/usr/bin/env python3
"""
t4_cctp/svb_stress/svb_stress_analysis.py

T4 補強調査: 2023年3月 SVB 危機時の USDC デペッグ — ストレス倍率の歴史的校正。

目的:
  待ち行列分析 v1（queueing_sim_summary.md）の臨界倍率
    - W_q > 0.5s: λ*/λ_real = 1.9–2.6×（c=2）
    - 飽和 λ_sat/λ_real = 5.4–7.4×（c=2）
  に対し、「歴史的に実在したストレス（SVB 危機時の USDC オンチェーン活動急増）は
  臨界線に対してどこまで迫ったか」を 1 つの倍率 X で言えるようにする。

データ源（すべて公開・読み取りのみ・新規送金なし）:
  - オンチェーン: Coin Metrics community API（無料枠・キー不要）
      asset=usdc, metric=TxTfrCnt（Ethereum ERC-20 USDC の日次転送件数）
      2023-02-01 .. 2023-03-31。ローカル保存 cm_usdc.json。
      ※ TxTfrValAdjUSD（転送額）は community 枠では forbidden のため件数のみ。
      ※ 転送「件数」は待ち行列の到着数 λ に対応する最も自然な量。
  - 文献由来（出典は svb_stress_biblio.md / summary.md 参照）:
      Fed FEDS Note (Watsky et al. 2024): DEX 二次市場出来高が 2023-03-11 に
        $20B 超、平時 $1–3B ＝ 約 7–20×。
      CoinDesk 2023-03-15: DEX 全体で過去最高 $25B/日（前高 $24.3B, 2021-05）。
      Curve $6.03B（03-11）/ Uniswap ~$12B（24h）。

出力:
  result/T4_cctp/svb_stress_multipliers.csv  （日次と倍率の集計）
  result/T4_cctp/svb_stress_figure.png       （2 パネル, 300dpi）
  ※ summary.md / biblio.md は別途手書き（本スクリプトは数値と図のみ担当）。

seed 不要（決定的集計）。既存の配色・フォントスタイル（v2）を踏襲。
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CM_JSON = HERE / "cm_usdc.json"
OUT_CSV = REPO / "result" / "T4_cctp" / "svb_stress_multipliers.csv"
OUT_PNG = REPO / "result" / "T4_cctp" / "svb_stress_figure.png"

# v1 待ち行列分析の臨界倍率（queueing_sim_summary.md, c=2, λ_real 基準）
CRIT_WQ05 = (1.9, 2.6)     # W_q > 0.5s になる λ*/λ_real
CRIT_SAT = (5.4, 7.4)      # 飽和 λ_sat/λ_real

# 文献由来のストレス倍率（出典付き・summary/biblio 参照）
LIT_DEX_FED = (7.0, 20.0)  # Fed FEDS Note: >$20B vs $1–3B typical
LIT_DEX_VENUE = (20.0, 60.0)  # Curve/Uniswap swap 特化 venue（$6–12B vs 平時0.1–0.6B）

V2_RC = {
    "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 10.5,
    "ytick.labelsize": 11, "legend.fontsize": 10, "figure.titlesize": 14,
}


def load_series():
    d = json.load(open(CM_JSON))
    if "error" in d:
        raise SystemExit(f"Coin Metrics error: {d['error']}")
    rows = d["data"]
    dates = np.array([r["time"][:10] for r in rows])
    cnt = np.array([int(r["TxTfrCnt"]) for r in rows], dtype=float)
    return dates, cnt


def daterange_mask(dates, lo, hi):
    return (dates >= lo) & (dates <= hi)


def main():
    dates, cnt = load_series()

    # 2/25 は近傍の約 1/4 の欠測アーティファクト（Coin Metrics のデータギャップ）。
    # 中央値ベース統計から除外して頑健化（除外の事実は summary に明記）。
    anomaly = "2023-02-25"
    valid = dates != anomaly

    # --- 平時ベースライン（複数定義で幅を持たせる）---
    m_lateFeb = daterange_mask(dates, "2023-02-15", "2023-02-28") & valid
    m_feb = daterange_mask(dates, "2023-02-01", "2023-02-28") & valid
    m_precrisis = daterange_mask(dates, "2023-03-03", "2023-03-09")  # 危機直前週

    base = {
        "late_Feb(15-28)": float(np.median(cnt[m_lateFeb])),
        "all_Feb(01-28)": float(np.median(cnt[m_feb])),
        "pre_crisis_wk(03-09)": float(np.median(cnt[m_precrisis])),
    }

    # --- ピーク（デペッグ週末 03-10..03-13、最高日は 03-11）---
    peak_day = "2023-03-11"
    peak_val = float(cnt[dates == peak_day][0])
    m_peakwin = daterange_mask(dates, "2023-03-10", "2023-03-12")
    peak_win_mean = float(cnt[m_peakwin].mean())

    print("=== USDC daily transfer count (Coin Metrics, Ethereum ERC-20) ===")
    print(f"peak day {peak_day}: {peak_val:,.0f}")
    print(f"peak window 03-10..03-12 mean: {peak_win_mean:,.0f}")
    for k, v in base.items():
        print(f"baseline median [{k}]: {v:,.0f}")

    # --- ストレス倍率 X = ピーク / 各ベースライン ---
    X_day = {k: peak_val / v for k, v in base.items()}
    X_win = {k: peak_win_mean / v for k, v in base.items()}
    print("\n=== stress multiplier X (peak / baseline) ===")
    for k in base:
        print(f"  X_peakday / {k:22s} = {X_day[k]:.2f}x   "
              f"X_peakwin = {X_win[k]:.2f}x")

    # 主指標はピーク単日 / 各ベースライン（headline）。3日窓は補助（平滑・下振れ）。
    X_lo, X_hi = min(X_day.values()), max(X_day.values())
    Xw_lo, Xw_hi = min(X_win.values()), max(X_win.values())
    print(f"\n>> aggregate transfer-count stress X (peak day) = {X_lo:.2f}–{X_hi:.2f}x")
    print(f">> (3-day window 03-10..03-12 mean) = {Xw_lo:.2f}–{Xw_hi:.2f}x")
    print(f">> our critical (c=2): W_q>0.5s {CRIT_WQ05[0]}–{CRIT_WQ05[1]}x, "
          f"saturation {CRIT_SAT[0]}–{CRIT_SAT[1]}x")

    # --- CSV 出力 ---
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerow(["daily", "date", "TxTfrCnt"])
        for dt, c in zip(dates, cnt):
            w.writerow(["daily", dt, int(c)])
        w.writerow(["peak", "peak_day", peak_day])
        w.writerow(["peak", "peak_day_TxTfrCnt", int(peak_val)])
        w.writerow(["peak", "peak_win_0310_0312_mean", int(peak_win_mean)])
        for k, v in base.items():
            w.writerow(["baseline_median", k, int(v)])
        for k in base:
            w.writerow(["X_peakday", k, round(X_day[k], 3)])
            w.writerow(["X_peakwin", k, round(X_win[k], 3)])
        w.writerow(["X_aggregate_range", "lo_hi", f"{X_lo:.2f}-{X_hi:.2f}"])
        w.writerow(["crit_Wq05", "lo_hi_x", f"{CRIT_WQ05[0]}-{CRIT_WQ05[1]}"])
        w.writerow(["crit_saturation", "lo_hi_x", f"{CRIT_SAT[0]}-{CRIT_SAT[1]}"])
    print(f"[write] {OUT_CSV}")

    make_figure(dates, cnt, anomaly, base, peak_day, peak_val, X_lo, X_hi)
    print(f"[write] {OUT_PNG}")

    return X_lo, X_hi


def make_figure(dates, cnt, anomaly, base, peak_day, peak_val, X_lo, X_hi):
    x = np.arange(len(dates))
    base_med = float(np.median(list(base.values())))

    with plt.rc_context(V2_RC):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2),
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})

        # ---------- 左: 日次転送件数の時系列 ----------
        ax = axes[0]
        cnt_plot = cnt.copy()
        an_i = int(np.where(dates == anomaly)[0][0])
        # 欠測日は線を途切れさせて表示（マーカーは残す）
        ax.plot(x, cnt_plot / 1e6, "-", color="#1f3b73", lw=1.4, zorder=3)
        ax.plot(np.delete(x, an_i), np.delete(cnt_plot, an_i) / 1e6, "o",
                color="#1f3b73", ms=3, zorder=4)
        ax.plot(x[an_i], cnt_plot[an_i] / 1e6, "x", color="0.5", ms=7,
                zorder=4, label="data gap (02-25, excluded)")

        # SVB デペッグ週末（03-10..03-13）を強調
        s = int(np.where(dates == "2023-03-10")[0][0])
        e = int(np.where(dates == "2023-03-13")[0][0])
        ax.axvspan(s - 0.4, e + 0.4, color="#fde68a", alpha=0.55, zorder=0,
                   label="SVB depeg 03-10…03-13")
        # 平時ベースライン中央値の帯
        ax.axhline(base_med / 1e6, color="#0f766e", ls="--", lw=1.3,
                   label=f"baseline median ≈ {base_med/1e6:.2f}M/day")
        # ピーク注記
        pk_i = int(np.where(dates == peak_day)[0][0])
        ax.annotate(f"peak {peak_day}\n{peak_val/1e6:.2f}M  ({X_lo:.1f}–{X_hi:.1f}× base)",
                    xy=(pk_i, peak_val / 1e6), xytext=(pk_i - 19, 1.30),
                    fontsize=10, color="#c0392b", fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))

        tick_i = [i for i, d in enumerate(dates) if d[-2:] in ("01", "08", "15", "22")]
        ax.set_xticks(tick_i)
        ax.set_xticklabels([dates[i][5:] for i in tick_i], rotation=0)
        ax.set_ylabel("USDC transfers  [million / day]")
        ax.set_xlabel("date  (2023, Ethereum ERC-20 USDC)")
        ax.set_title("(a) USDC daily transfer count — SVB depeg spike",
                     fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.92)

        # ---------- 右: ストレス倍率 vs 臨界帯 ----------
        ax = axes[1]
        # 我々の臨界帯（縦スパン）
        ax.axhspan(CRIT_WQ05[0], CRIT_WQ05[1], color="#f59e0b", alpha=0.20, zorder=0)
        ax.axhspan(CRIT_SAT[0], CRIT_SAT[1], color="#c0392b", alpha=0.16, zorder=0)
        ax.text(0.02, np.mean(CRIT_WQ05), " $W_q>0.5$s critical\n (c=2)",
                fontsize=9, color="#b45309", va="center")
        ax.text(0.02, np.mean(CRIT_SAT), " saturation (c=2)",
                fontsize=9, color="#c0392b", va="center")

        bars = [
            ("aggregate\nUSDC transfers\n(this study)", (X_lo, X_hi), "#1f3b73"),
            ("DEX 2nd-mkt\nvolume\n(Fed 2024)", LIT_DEX_FED, "#0f766e"),
            ("Curve/Uniswap\nswap venues\n(news)", LIT_DEX_VENUE, "#6b21a8"),
        ]
        xpos = np.arange(len(bars)) + 0.5
        for xp, (lab, (lo, hi), col) in zip(xpos, bars):
            ax.plot([xp, xp], [lo, hi], color=col, lw=9, solid_capstyle="round",
                    alpha=0.85, zorder=3)
            ax.plot(xp, lo, "_", color=col, ms=16, mew=2.5)
            ax.plot(xp, hi, "_", color=col, ms=16, mew=2.5)
            ax.text(xp, hi * 1.06, f"{lo:.0f}–{hi:.0f}×" if hi >= 10
                    else f"{lo:.1f}–{hi:.1f}×", ha="center", fontsize=9.5,
                    color=col, fontweight="bold")

        ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
        ax.text(len(bars) - 0.02, 1.0, " calm (1×)", ha="right", va="bottom",
                fontsize=8.5, color="0.4")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 90)
        ax.set_xlim(0, len(bars))
        ax.set_xticks(xpos)
        ax.set_xticklabels([b[0] for b in bars], fontsize=9)
        ax.set_ylabel("stress multiplier  X  (peak / calm)")
        ax.set_title("(b) historical SVB stress vs our critical multipliers",
                     fontsize=12)
        ax.grid(True, axis="y", which="both", alpha=0.22)

        fig.suptitle("T4 supplement: historical calibration of SVB-crisis (Mar 2023) USDC "
                     "stress vs our queueing critical multipliers\n"
                     "(CCTP was NOT live during the crisis — mainnet 2023-04-26)",
                     y=1.04, fontsize=12.5)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
