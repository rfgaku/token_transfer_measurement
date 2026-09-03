#!/usr/bin/env python3
"""
t4_cctp/queueing_sim/queueing_sim.py

T4 CCTP attestation パイプラインの待ち行列分析
（容量下界の逆推定 ＋ 臨界スループットの掃引）。

背景:
  T4 deposit の二峰性は mechanism_sim/ の順問題検証により「無記憶な二値確認ターゲット」
  機構（シナリオB）と同定済み。共同研究者から「遅い側の処理が貯まると待ち行列が発生
  しうる。シナリオBの設定で処理待ちの待ち時間・行列長を調べよ」との提案を受けた。
  実測データには待ち行列の痕跡がない（slow 群残差は fast 群と同一・混雑指標と無相関）。
  本スクリプトはこれを逆手に取り、
    (a) 実効処理並列度 c の下界の逆推定
    (b) 到着率 λ 掃引による臨界スループットの見積り
  を行う。

モデル（attestation パイプライン = c 台並列サーバの M/G/c 待ち行列）:
  - 到着: ポアソン過程（率 λ [tx/s]、指数分布間隔）。
  - サービス時間 S: シナリオB準拠の二値混合。確率 0.63 で fast、0.37 で slow。
    各群のサービス時間は実測 iris_wait(ms) の *経験分布* からのリサンプリング
    （分布仮定を置かないノンパラメトリック方式）。入力は
    deposit_l1_enriched.csv の group∈{fast,slow}。
  - 規律: FCFS。ウォームアップ後の定常状態で平均待ち時間 W_q・平均行列長 L_q を推定。
    各 (λ, c) につき N_JOBS ジョブ × N_REPS 反復、95% 区間を併記。
    L_q は Little の法則 L_q = λ·W_q で算出。

λ_real の実測推定（悉皆調査データから、read-only）:
  cctp_fast_standard_events.csv（Fast/Standard 悉皆イベント）の窓ごとの実時間長
  （各 window_id 内 block_time_utc の max−min）を合算し、
    λ_chain = (その chain の Fast 件数) / (窓実時間長の総和)
  を算出。λ_arb = Arbitrum One の Fast 率、λ_3chain = 3 チェーン Fast 率の和
  （独立ポアソン過程の重ね合わせ = 率の和、Iris 総負荷の上界側）。

出力（新規のみ・result/ 配下）:
  result/T4_cctp/queueing_sim_results.csv
  result/T4_cctp/queueing_sim_figure.png   （2 パネル, 300dpi）
  result/T4_cctp/queueing_sim_summary.md

使い方:
  python3 -u t4_cctp/queueing_sim/queueing_sim.py

依存: pandas / numpy / scipy / matplotlib のみ。乱数シード固定 seed=42。
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
ENR_CSV = REPO / "result" / "T4_cctp" / "deposit_l1_enriched.csv"
EVENTS_CSV = REPO / "result" / "T4_cctp" / "cctp_fast_standard_events.csv"
OUT_CSV = REPO / "result" / "T4_cctp" / "queueing_sim_results.csv"
OUT_PNG = REPO / "result" / "T4_cctp" / "queueing_sim_figure.png"
OUT_MD = REPO / "result" / "T4_cctp" / "queueing_sim_summary.md"

SEED = 42
N_JOBS = 200_000            # 各 (λ,c) 反復あたりのジョブ数
WARMUP_FRAC = 0.10          # 先頭 10% をウォームアップとして破棄
N_REPS = 5                  # 反復回数（95% 区間用）
P_SLOW = 0.37               # シナリオB: slow（遅い山）比率
P_FAST = 1.0 - P_SLOW       # = 0.63

C_LIST = [1, 2, 4, 8, 16]   # 掃引する並列度
DETECT_LIMIT_S = 0.5        # 検出限界: W_q>0.5s なら実測 slow 残差に現れたはず
SLA_LIMITS_S = [0.5, 5.0]   # 臨界 λ* を求める W_q 閾値
WAIT_THR_S = 2.5            # 待ち時間分布統計 P_wait_gt2p5 の閾値 [s]

COL_GROUP = "group"
COL_IRIS = "iris_wait(ms)"


# --------------------------------------------------------------------------- #
# 待ち時間分布の統計量（平均以外）: P(W_q>0), P(W_q>WAIT_THR_S), W_q の 95% 点
#   既存の乱数系列・平均値には一切影響しない（同じ waits 配列から追加算出するだけ）。
# --------------------------------------------------------------------------- #
def wait_dist_stats(w):
    """ウォームアップ後の待ち時間配列 w から分布統計量を返す。"""
    return {
        "P_wait": float((w > 0.0).mean()),
        "P_wait_gt": float((w > WAIT_THR_S).mean()),
        "Wq_p95": float(np.percentile(w, 95.0)),
    }


def mean_dist_stats(dists):
    """反復ごとの wait_dist_stats を平均して報告用のキー名に直す。"""
    return {
        "P_wait": float(np.mean([d["P_wait"] for d in dists])),
        "P_wait_gt2p5": float(np.mean([d["P_wait_gt"] for d in dists])),
        "Wq_p95": float(np.mean([d["Wq_p95"] for d in dists])),
    }


# --------------------------------------------------------------------------- #
# サービス時間: 実測 iris_wait の経験分布（fast/slow）
# --------------------------------------------------------------------------- #
def load_service_pools():
    """deposit_l1_enriched.csv の group 別 iris_wait(ms) を秒に直して返す。"""
    df = pd.read_csv(ENR_CSV)
    d = df[df[COL_GROUP].isin(["fast", "slow"])]
    fast = d.loc[d[COL_GROUP] == "fast", COL_IRIS].astype(float).to_numpy() / 1000.0
    slow = d.loc[d[COL_GROUP] == "slow", COL_IRIS].astype(float).to_numpy() / 1000.0
    return fast, slow


def expected_service(fast, slow):
    """混合サービス時間の理論的 E[S], E[S^2]（群平均は経験平均）。"""
    es = P_FAST * fast.mean() + P_SLOW * slow.mean()
    es2 = P_FAST * (fast ** 2).mean() + P_SLOW * (slow ** 2).mean()
    return es, es2


def sample_services(rng, n, fast, slow):
    """n 件の混合サービス時間を経験分布からリサンプリング（ベクトル化）。"""
    is_slow = rng.random(n) < P_SLOW
    n_slow = int(is_slow.sum())
    out = np.empty(n)
    out[is_slow] = rng.choice(slow, size=n_slow, replace=True)
    out[~is_slow] = rng.choice(fast, size=n - n_slow, replace=True)
    return out


# --------------------------------------------------------------------------- #
# M/G/c FCFS のイベント駆動シミュレーション（自作・SimPy 不使用）
# --------------------------------------------------------------------------- #
def simulate_mgc(lam, c, fast, slow, rng, n_jobs=N_JOBS, warmup_frac=WARMUP_FRAC):
    """
    率 λ のポアソン到着・c 台並列サーバ・FCFS の待ち時間を返す。

    FCFS + 同一 c サーバでは「到着ジョブは最も早く空くサーバに割当てられる」ため、
    c 個のサーバ空き時刻を最小ヒープで保持し O(N log c) で各ジョブの待ち時間を得る。
    戻り値: (mean_Wq, rho)  ウォームアップ後ジョブの平均待ち時間 [s] と負荷率。
    """
    from heapq import heapreplace

    interarr = rng.exponential(1.0 / lam, size=n_jobs)
    arrivals = np.cumsum(interarr)
    services = sample_services(rng, n_jobs, fast, slow)

    free = [0.0] * c                       # 各サーバの空き時刻（最小ヒープ）
    waits = np.empty(n_jobs)
    a_arr = arrivals
    s_arr = services
    for i in range(n_jobs):
        a = a_arr[i]
        earliest = free[0]                 # 最も早く空くサーバ
        start = a if a >= earliest else earliest
        waits[i] = start - a
        heapreplace(free, start + s_arr[i])

    w0 = int(n_jobs * warmup_frac)
    w_post = waits[w0:]
    wq = float(w_post.mean())
    rho = lam * (P_FAST * fast.mean() + P_SLOW * slow.mean()) / c
    return wq, rho, wait_dist_stats(w_post)


def sim_point(lam, c, fast, slow, n_reps=N_REPS):
    """(λ,c) を n_reps 反復し W_q・L_q の平均と 95% 区間を返す。"""
    wqs = []
    dists = []
    for r in range(n_reps):
        rng = np.random.default_rng(SEED + 1000 * c + r)
        wq, rho, dst = simulate_mgc(lam, c, fast, slow, rng)
        wqs.append(wq)
        dists.append(dst)
    wqs = np.asarray(wqs)
    wq_mean = float(wqs.mean())
    # 95% 区間（反復間、正規近似 or 分位。反復少数のため t 的に幅を持たせず分位で）
    wq_lo = float(np.percentile(wqs, 2.5))
    wq_hi = float(np.percentile(wqs, 97.5))
    lq_mean = lam * wq_mean               # Little の法則
    lq_lo, lq_hi = lam * wq_lo, lam * wq_hi
    out = {
        "lambda": lam, "c": c, "rho": lam * (P_FAST * fast.mean() + P_SLOW * slow.mean()) / c,
        "Wq_mean": wq_mean, "Wq_lo": wq_lo, "Wq_hi": wq_hi,
        "Lq_mean": lq_mean, "Lq_lo": lq_lo, "Lq_hi": lq_hi,
    }
    out.update(mean_dist_stats(dists))     # 5 反復の平均
    return out


# --------------------------------------------------------------------------- #
# λ_real の実測推定（悉皆イベントの窓実時間長）
# --------------------------------------------------------------------------- #
def estimate_lambdas():
    """cctp_fast_standard_events.csv から chain 別 Fast 到着率と窓時間長を返す。"""
    ev = pd.read_csv(EVENTS_CSV)
    ev["t"] = pd.to_datetime(ev["block_time_utc"], utc=True)
    info = {}
    for chain, g in ev.groupby("chain"):
        dur = 0.0
        for _wid, gw in g.groupby("window_id"):
            dur += (gw["t"].max() - gw["t"].min()).total_seconds()
        fast = int((g["class"] == "Fast").sum())
        info[chain] = {"fast": fast, "dur_s": dur, "lam": fast / dur,
                       "n_win": int(g["window_id"].nunique())}
    lam_arb = info["Arbitrum One"]["lam"]
    lam_3chain = sum(v["lam"] for v in info.values())   # ポアソン重ね合わせ = 率の和
    return info, lam_arb, lam_3chain


# --------------------------------------------------------------------------- #
# 臨界 λ*（W_q が閾値を超える λ）を掃引結果から線形補間で求める
# --------------------------------------------------------------------------- #
def critical_lambda(lams, wqs, thr):
    """W_q(λ) が thr を初めて超える λ* を対数線形補間で推定（無ければ NaN）。"""
    lams = np.asarray(lams)
    wqs = np.asarray(wqs)
    order = np.argsort(lams)
    lams, wqs = lams[order], wqs[order]
    for i in range(1, len(lams)):
        if wqs[i - 1] < thr <= wqs[i]:
            # log-lambda 上で線形補間
            x0, x1 = np.log(lams[i - 1]), np.log(lams[i])
            y0, y1 = wqs[i - 1], wqs[i]
            xr = x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
            return float(np.exp(xr))
    return float("nan")


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def main():
    fast, slow = load_service_pools()
    ES, ES2 = expected_service(fast, slow)
    SCV = ES2 / ES ** 2 - 1.0
    print(f"[service] fast n={fast.size} mean={fast.mean():.3f}s | "
          f"slow n={slow.size} mean={slow.mean():.3f}s | "
          f"E[S]={ES:.4f}s E[S^2]={ES2:.4f} SCV={SCV:.3f}")

    info, lam_arb, lam_3chain = estimate_lambdas()
    for ch, v in info.items():
        print(f"[lambda] {ch:14s} fast={v['fast']:5d} dur={v['dur_s']/3600:6.2f}h "
              f"n_win={v['n_win']} lam={v['lam']:.5f} tx/s")
    print(f"[lambda] lam_arb={lam_arb:.5f}  lam_3chain={lam_3chain:.5f} tx/s")

    # ---- (a) Little の法則で in-flight 数、検出限界からの c 下界 ----
    lam_real = {"arb": lam_arb, "3chain": lam_3chain}
    n_inflight = {k: v * ES for k, v in lam_real.items()}
    print(f"[little] E[N_in_flight] arb={n_inflight['arb']:.3f} "
          f"3chain={n_inflight['3chain']:.3f}")

    # 実測レートで W_q < 0.5s を満たす最小 c を探索（下界）
    c_lower = {}
    real_wq = {}   # (label, c) -> point
    for label, lam in lam_real.items():
        real_wq[label] = {}
        cmin = None
        for c in C_LIST:
            pt = sim_point(lam, c, fast, slow)
            real_wq[label][c] = pt
            print(f"[a] lam_{label}={lam:.5f} c={c:2d} rho={pt['rho']:.3f} "
                  f"Wq={pt['Wq_mean']:.4f}s [{pt['Wq_lo']:.4f},{pt['Wq_hi']:.4f}] "
                  f"Lq={pt['Lq_mean']:.4f}")
            if cmin is None and pt["Wq_mean"] < DETECT_LIMIT_S:
                cmin = c
        c_lower[label] = cmin

    # ---- (b) λ 掃引 ----
    # 共通の λ グリッド（対数）。各 c は安定域 ρ<0.97 のみシミュレート。
    lam_grid = np.geomspace(0.01, 3.2, 26)
    sweep = {c: [] for c in C_LIST}       # c -> list of points
    for c in C_LIST:
        cap = c / ES                      # 飽和到着率 λ_sat = c/E[S]
        for lam in lam_grid:
            if lam >= 0.97 * cap:
                continue
            pt = sim_point(lam, c, fast, slow)
            sweep[c].append(pt)
        # 実測レートも掃引点に含める（図の縦線位置での値確認用）
        print(f"[b] c={c:2d} lam_sat={cap:.4f} n_points={len(sweep[c])}")

    # 臨界 λ*
    crit = {}   # c -> {thr: lam*}
    for c in C_LIST:
        lams = [p["lambda"] for p in sweep[c]]
        wqs = [p["Wq_mean"] for p in sweep[c]]
        crit[c] = {thr: critical_lambda(lams, wqs, thr) for thr in SLA_LIMITS_S}

    # ---- results.csv ----
    rows = []
    for label, lam in lam_real.items():
        for c in C_LIST:
            p = real_wq[label][c]
            rows.append({"block": f"a_real_{label}", "lambda": round(lam, 6), "c": c,
                         "rho": round(p["rho"], 4),
                         "Wq_mean_s": round(p["Wq_mean"], 5),
                         "Wq_lo_s": round(p["Wq_lo"], 5), "Wq_hi_s": round(p["Wq_hi"], 5),
                         "Lq_mean": round(p["Lq_mean"], 5),
                         "Lq_lo": round(p["Lq_lo"], 5), "Lq_hi": round(p["Lq_hi"], 5)})
    for c in C_LIST:
        for p in sweep[c]:
            rows.append({"block": "b_sweep", "lambda": round(p["lambda"], 6), "c": c,
                         "rho": round(p["rho"], 4),
                         "Wq_mean_s": round(p["Wq_mean"], 5),
                         "Wq_lo_s": round(p["Wq_lo"], 5), "Wq_hi_s": round(p["Wq_hi"], 5),
                         "Lq_mean": round(p["Lq_mean"], 5),
                         "Lq_lo": round(p["Lq_lo"], 5), "Lq_hi": round(p["Lq_hi"], 5)})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {OUT_CSV}")

    # ---- 図: 2 パネル ----
    make_figure(sweep, lam_real, crit)
    print(f"[write] {OUT_PNG}")

    # ---- summary.md ----
    write_summary(fast, slow, ES, ES2, SCV, info, lam_arb, lam_3chain,
                  lam_real, n_inflight, c_lower, real_wq, crit)
    print(f"[write] {OUT_MD}")


def make_figure(sweep, lam_real, crit):
    colors = {1: "#c0392b", 2: "#e67e22", 4: "#0f766e", 8: "#1f3b73", 16: "#6b21a8"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    for panel, (ax, key, ylab, title) in enumerate([
        (axes[0], "Wq_mean", "mean queue wait  $W_q$  [s]",
         "(left) queue wait  $W_q$  vs arrival rate  $\\lambda$"),
        (axes[1], "Lq_mean", "mean queue length  $L_q$  [jobs]",
         "(right) queue length  $L_q$  vs arrival rate  $\\lambda$"),
    ]):
        for c in C_LIST:
            pts = sweep[c]
            lams = [p["lambda"] for p in pts]
            ys = [p[key] for p in pts]
            lo = [p[key.replace("_mean", "_lo")] for p in pts]
            hi = [p[key.replace("_mean", "_hi")] for p in pts]
            ax.plot(lams, ys, "-o", ms=3, lw=1.4, color=colors[c], label=f"c={c}")
            ax.fill_between(lams, lo, hi, color=colors[c], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("arrival rate  $\\lambda$  [tx/s]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        # λ_real 縦線
        ax.axvline(lam_real["arb"], color="0.25", ls="--", lw=1.2)
        ax.axvline(lam_real["3chain"], color="0.45", ls=":", lw=1.2)
        ax.text(lam_real["arb"], ax.get_ylim()[1], " $\\lambda_{arb}$",
                rotation=90, va="top", ha="right", fontsize=8, color="0.25")
        ax.text(lam_real["3chain"], ax.get_ylim()[1], " $\\lambda_{3ch}$",
                rotation=90, va="top", ha="left", fontsize=8, color="0.45")
        if panel == 0:
            for thr, c_ in [(0.5, "#888"), (5.0, "#555")]:
                ax.axhline(thr, color=c_, ls="-", lw=0.9, alpha=0.7)
                ax.text(ax.get_xlim()[0], thr, f" {thr}s ", va="bottom",
                        ha="left", fontsize=8, color=c_)
        ax.legend(loc="lower right", fontsize=8, title="parallelism", framealpha=0.9)

    fig.suptitle("T4 CCTP attestation queue: $W_q$ / $L_q$ vs arrival rate "
                 "(M/G/c, FCFS, empirical service, seed=42)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(fast, slow, ES, ES2, SCV, info, lam_arb, lam_3chain,
                  lam_real, n_inflight, c_lower, real_wq, crit):
    L = []
    L.append("# T4 CCTP attestation パイプラインの待ち行列分析\n")
    L.append("（容量下界の逆推定 ＋ 臨界スループットの掃引）\n")

    L.append("## 1. モデル定義\n")
    L.append("attestation パイプラインを **c 台並列サーバの M/G/c 待ち行列**（FCFS）とみなす。")
    L.append("- **到着**: ポアソン過程（率 λ [tx/s]、指数分布間隔）。")
    L.append("- **サービス時間 S**: シナリオB（無記憶な二値確認ターゲット）準拠の二値混合。"
             f"確率 {P_FAST:.2f} で fast、{P_SLOW:.2f} で slow。各群のサービス時間は実測 "
             "`iris_wait(ms)` の**経験分布からのリサンプリング**（分布仮定を排除）。"
             f"入力は `deposit_l1_enriched.csv` の group∈{{fast,slow}}（fast n={fast.size}, slow n={slow.size}）。")
    L.append("- **規律**: FCFS。ウォームアップ（先頭 "
             f"{int(WARMUP_FRAC*100)}%）破棄後の定常状態で平均待ち時間 $W_q$ を測定、"
             f"$L_q=\\lambda W_q$（Little の法則）。各 (λ,c) は "
             f"{N_JOBS:,} ジョブ × {N_REPS} 反復（seed={SEED} 固定）、95% 区間併記。\n")

    L.append("### サービス時間モーメント（実測経験分布）\n")
    L.append(f"- fast 群: 平均 {fast.mean():.3f}s（min {fast.min():.3f}, max {fast.max():.3f}）")
    L.append(f"- slow 群: 平均 {slow.mean():.3f}s（min {slow.min():.3f}, max {slow.max():.3f}）")
    L.append(f"- 混合 **E[S] = {ES:.4f} s**、E[S²] = {ES2:.4f} s²、変動係数² SCV = {SCV:.3f}\n")

    L.append("## 2. 実到着率 λ_real の導出（悉皆調査データ）\n")
    L.append("`cctp_fast_standard_events.csv`（Fast/Standard 悉皆イベント）の "
             "window_id ごとの実時間長（窓内 `block_time_utc` の max−min）を合算し、"
             "`λ_chain = (chain の Fast 件数) / (窓実時間長の総和)` で推定。\n")
    L.append("| chain | Fast 件数 | 窓数 | 窓実時間長 合計 | λ_Fast [tx/s] |")
    L.append("|---|---|---|---|---|")
    for ch, v in info.items():
        L.append(f"| {ch} | {v['fast']} | {v['n_win']} | {v['dur_s']/3600:.2f} h | {v['lam']:.5f} |")
    L.append("")
    L.append(f"- **λ_arb（Arbitrum One Fast）= {lam_arb:.5f} tx/s**（≈ {lam_arb*3600:.1f} tx/h）")
    L.append(f"- **λ_3chain（3 チェーン Fast 率の和 = 独立ポアソン過程の重ね合わせ）= "
             f"{lam_3chain:.5f} tx/s**（≈ {lam_3chain*3600:.1f} tx/h）\n")

    L.append("## 3. (a) 容量下界の逆推定\n")
    L.append("Little の法則より、同時進行中の attestation 数 "
             "$E[N_\\text{in-flight}] = \\lambda\\,E[S]$:")
    L.append(f"- λ_arb: E[N_in-flight] = {lam_arb:.5f} × {ES:.3f} = **{n_inflight['arb']:.3f}**")
    L.append(f"- λ_3chain: E[N_in-flight] = {lam_3chain:.5f} × {ES:.3f} = **{n_inflight['3chain']:.3f}**\n")
    L.append("実測には待ち行列の痕跡が無い（slow 群残差は fast 群と同一・混雑指標と無相関）。"
             f"検出限界を **W_q > {DETECT_LIMIT_S}s なら slow 残差に現れたはず**と置き、"
             f"実測レートで $W_q(\\lambda_\\text{{real}}, c) < {DETECT_LIMIT_S}$s を満たす"
             "最小 c を下界として報告する。\n")
    L.append("| c | ρ (arb) | W_q (arb) [s] | ρ (3ch) | W_q (3ch) [s] |")
    L.append("|---|---|---|---|---|")
    for c in C_LIST:
        pa = real_wq["arb"][c]
        p3 = real_wq["3chain"][c]
        def mark(p):
            return "✅" if p["Wq_mean"] < DETECT_LIMIT_S else "❌"
        L.append(f"| {c} | {pa['rho']:.3f} | {pa['Wq_mean']:.4f} "
                 f"[{pa['Wq_lo']:.4f}, {pa['Wq_hi']:.4f}] {mark(pa)} | "
                 f"{p3['rho']:.3f} | {p3['Wq_mean']:.4f} "
                 f"[{p3['Wq_lo']:.4f}, {p3['Wq_hi']:.4f}] {mark(p3)} |")
    L.append("")
    L.append(f"- **容量下界**: λ_arb では W_q<{DETECT_LIMIT_S}s を満たす最小 c = "
             f"**{c_lower['arb']}** → 実効並列度 **c ≥ {c_lower['arb']}**。")
    L.append(f"  λ_3chain では最小 c = **{c_lower['3chain']}** → **c ≥ {c_lower['3chain']}**。")
    L.append("  すなわち c=1（単一直列サーバ）なら slow 残差に有意な追加待ちが観測された"
             "はずだが、実測にそれが無いことから、パイプラインの律速段は**逐次 1 段ではなく"
             "少なくとも複数並列**であることが下から言える。\n")

    L.append("## 4. (b) 臨界スループットの掃引\n")
    L.append(f"c ∈ {{{', '.join(map(str, C_LIST))}}} について λ を掃引し、$W_q$ が "
             f"{SLA_LIMITS_S[0]}s / {SLA_LIMITS_S[1]}s を超える臨界 λ* を求める"
             "（飽和到着率 λ_sat = c/E[S]）。\n")
    L.append("| c | λ_sat=c/E[S] | λ* (W_q>0.5s) | λ* (W_q>5s) | 余裕倍率 λ*(0.5s)/λ_arb | /λ_3chain |")
    L.append("|---|---|---|---|---|---|")
    for c in C_LIST:
        cap = c / ES
        c05 = crit[c][0.5]
        c50 = crit[c][5.0]
        m_arb = c05 / lam_arb if c05 == c05 else float("nan")
        m_3 = c05 / lam_3chain if c05 == c05 else float("nan")
        def f(x):
            return f"{x:.4f}" if x == x else "—"
        L.append(f"| {c} | {cap:.4f} | {f(c05)} | {f(c50)} | "
                 f"{m_arb:.1f}× | {m_3:.1f}× |")
    L.append("")

    # 余裕倍率の一文（c 下界 = 最小並列度での最保守評価）
    c_ref = c_lower["arb"]
    c05_ref = crit[c_ref][0.5]
    marg_arb = c05_ref / lam_arb
    marg_3 = c05_ref / lam_3chain
    cap_ref = c_ref / ES
    L.append(f"**実 λ は臨界 λ* に対して何倍の余裕か**: 下界の並列度 c={c_ref}（最保守）"
             f"でも、W_q が {SLA_LIMITS_S[0]}s を超える臨界 λ*≈{c05_ref:.4f} tx/s は "
             f"λ_arb の **{marg_arb:.1f} 倍**、λ_3chain の **{marg_3:.1f} 倍**（飽和 λ_sat="
             f"{cap_ref:.4f} tx/s まではそれぞれ {cap_ref/lam_arb:.1f} 倍 / "
             f"{cap_ref/lam_3chain:.1f} 倍）の余裕がある。並列度が下界 c={c_ref} より大きければ"
             f"（c=4 なら λ*(0.5s)/λ_arb ≈ {crit[4][0.5]/lam_arb:.0f} 倍、c=8 なら "
             f"≈ {crit[8][0.5]/lam_arb:.0f} 倍）余裕は一〜二桁に拡大する。いずれにせよ"
             "現行トラフィックは待ち行列が立ち上がる領域より十分下にあり、実測に行列痕跡が"
             "無いことと整合する。\n")

    L.append("## 5. 図\n")
    L.append("`queueing_sim_figure.png`（300dpi, 2 パネル）: "
             "(左) $W_q$ vs λ、(右) $L_q$ vs λ。c 別曲線＋95%帯、"
             "λ_arb / λ_3chain の縦線、左パネルに 0.5s/5s の水平参照線。両軸対数。\n")

    L.append("## 6. 限界\n")
    L.append(f"- **λ_3chain は上界側**: サービス時間 E[S] は Arbitrum 発 deposit の実測 "
             "iris_wait から推定しており、Iris は複数ドメイン（Arbitrum/Base/Ethereum 等）"
             "を処理するため、実負荷は λ_3chain の**上界側**にある可能性がある"
             "（他ドメインのサービス時間分布は未計測）。")
    L.append("- **FCFS 仮定**: 実際の Iris は優先度・バッチ処理を行う可能性があり、"
             "厳密な FCFS ではないかもしれない。")
    L.append("- **c はパイプラインの直列律速段の抽象化**: 確認待ち（finality 待ち）は"
             "本来トランザクションごとに並列進行する物理過程であり、c 台サーバは"
             "「同時に処理できる本数の直列律速段」を抽象化したもの。したがって推定される "
             "c は物理的なサーバ台数ではなく**実効並列度の下界**として解釈すべき。")
    L.append("- **経験分布の標本規模**: サービス時間は fast n="
             f"{fast.size} / slow n={slow.size} の実測リサンプリングであり、"
             "裾の希少事象は十分に表現されていない可能性がある。\n")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")


# =========================================================================== #
# v2: 稼働率 ρ 横軸への統一 ＋ fast/slow 専用二系統（dedicated two-class）
#
#   共同研究者（待ち行列理論の教授）からの改訂依頼:
#     (1) 横軸を到着率 λ でなく稼働率 ρ = λE[S] に変更し統一評価する。
#     (2) 二峰性の機構（63/37 の二値振り分け）を踏まえ、fast 処理の待ち行列と
#         slow 処理の待ち行列を別々の系として扱い、両者の W_q・L_q を ρ 横軸で見る。
#
#   出力（v1 とは別ファイル・既存結果は保持）:
#     result/T4_cctp/queueing_sim_v2_results.csv
#       列: model, system, lambda, c, rho_perserver, rho_offered,
#           Wq_mean_s, Wq_lo_s, Wq_hi_s, Lq_mean, Lq_lo, Lq_hi,
#           P_wait, P_wait_gt2p5, Wq_p95_s
#       P_wait       = P(W_q > 0)               待たされる確率
#       P_wait_gt2p5 = P(W_q > WAIT_THR_S=2.5s) 実測 iris_wait の分解能で検知しうる待ち
#       Wq_p95_s     = W_q の 95% 点 [s]
#       いずれもウォームアップ後ジョブ全件から算出し、5 反復の平均を報告する。
#       model が *_oppoint_{arb,3chain} の行は ρ 掃引グリッド上に無い実測動作点
#       （λ_arb = Arbitrum 単独、λ_3chain = 3 チェーン合算）を個別に走らせた結果。
#     result/T4_cctp/queueing_sim_v2_fig1_pooled.png      （2×2, 300dpi）
#     result/T4_cctp/queueing_sim_v2_fig2_dedicated.png   （2 パネル, 300dpi）
#     result/T4_cctp/queueing_sim_v2_summary.md
# =========================================================================== #
from math import factorial

OUT_CSV_V2 = REPO / "result" / "T4_cctp" / "queueing_sim_v2_results.csv"
OUT_PNG_V2_F1 = REPO / "result" / "T4_cctp" / "queueing_sim_v2_fig1_pooled.png"
OUT_PNG_V2_F2 = REPO / "result" / "T4_cctp" / "queueing_sim_v2_fig2_dedicated.png"
OUT_MD_V2 = REPO / "result" / "T4_cctp" / "queueing_sim_v2_summary.md"

C_LIST_B = [1, 2]                       # dedicated 二系統の並列度（基本 1・併走 2）
RHO_GRID = np.linspace(0.03, 0.95, 24)  # per-server 稼働率の掃引グリッド
SLA_LIMITS_S_V2 = [0.5, 5.0]

# v2 配色（既存スタイル踏襲: teal / red 系）
COL_FAST = "#0f766e"   # fast 系（teal）
COL_SLOW = "#c0392b"   # slow 系（red）
LS_C = {1: "-", 2: "--"}   # c=1 実線 / c=2 破線

# 大きめフォント（前回 v3 並み）
V2_RC = {
    "axes.titlesize": 13,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "legend.title_fontsize": 11,
    "figure.titlesize": 14,
}


# --------------------------------------------------------------------------- #
# 単一経験分布プールの M/G/c シミュレーション（dedicated 系・単一クラス用）
# --------------------------------------------------------------------------- #
def simulate_mgc_single(lam, c, pool, rng, n_jobs=N_JOBS, warmup_frac=WARMUP_FRAC):
    """率 λ ポアソン到着・c 台並列・FCFS。サービス時間は単一プールから経験リサンプル。"""
    from heapq import heapreplace

    interarr = rng.exponential(1.0 / lam, size=n_jobs)
    arrivals = np.cumsum(interarr)
    services = rng.choice(pool, size=n_jobs, replace=True)

    free = [0.0] * c
    waits = np.empty(n_jobs)
    for i in range(n_jobs):
        a = arrivals[i]
        earliest = free[0]
        start = a if a >= earliest else earliest
        waits[i] = start - a
        heapreplace(free, start + services[i])

    w0 = int(n_jobs * warmup_frac)
    w_post = waits[w0:]
    return float(w_post.mean()), wait_dist_stats(w_post)


def sim_point_single(lam, c, pool, es, n_reps=N_REPS):
    """(λ,c) 単一系を n_reps 反復し、その系自身の ρ=λE[S]/c・W_q・L_q（95% 区間付）を返す。"""
    wqs = []
    dists = []
    for r in range(n_reps):
        rng = np.random.default_rng(SEED + 1000 * c + r)
        wq, dst = simulate_mgc_single(lam, c, pool, rng)
        wqs.append(wq)
        dists.append(dst)
    wqs = np.asarray(wqs)
    wq_mean = float(wqs.mean())
    wq_lo = float(np.percentile(wqs, 2.5))
    wq_hi = float(np.percentile(wqs, 97.5))
    out = {
        "lambda": lam, "c": c, "rho": lam * es / c,
        "Wq_mean": wq_mean, "Wq_lo": wq_lo, "Wq_hi": wq_hi,
        "Lq_mean": lam * wq_mean, "Lq_lo": lam * wq_lo, "Lq_hi": lam * wq_hi,
    }
    out.update(mean_dist_stats(dists))     # 5 反復の平均
    return out


# --------------------------------------------------------------------------- #
# 理論式（教授の講義流儀）: M/G/1 P-K 厳密・M/G/c は M/M/c 近似 + Allen-Cunneen
# --------------------------------------------------------------------------- #
def pk_wq_mg1(lam, es, es2):
    """M/G/1 の Pollaczek–Khinchine 厳密式: W_q = λE[S²] / (2(1−ρ))。"""
    rho = lam * es
    return lam * es2 / (2.0 * (1.0 - rho))


def erlang_c(c, a):
    """Erlang C（M/M/c で待つ確率）。a = 提供負荷 = λE[S] = ρc。"""
    rho = a / c
    s = sum(a ** k / factorial(k) for k in range(c))
    top = a ** c / (factorial(c) * (1.0 - rho))
    return top / (s + top)


def wq_mmc(lam, es, c):
    """M/M/c の平均待ち時間 W_q = C(c,a) / (cμ − λ)、μ = 1/E[S]。"""
    a = lam * es
    mu = 1.0 / es
    return erlang_c(c, a) / (c * mu - lam)


def wq_mgc_approx(lam, es, c, scv):
    """M/G/c 近似（Allen–Cunneen）: W_q ≈ W_q(M/M/c) × (C_a²+C_s²)/2、Poisson 到着 C_a²=1。
       c=1 では (1+SCV)/2·(M/M/1) が P-K 厳密式に一致する。"""
    return wq_mmc(lam, es, c) * (1.0 + scv) / 2.0


def moments(pool):
    """経験プールの E[S]・E[S²]・SCV。"""
    es = float(pool.mean())
    es2 = float((pool ** 2).mean())
    return es, es2, es2 / es ** 2 - 1.0


# --------------------------------------------------------------------------- #
# 図1: pooled M/G/c を per-server ρ と offered ρ の両軸で（2×2）
# --------------------------------------------------------------------------- #
def make_fig1_pooled(sweep, ES, lam_real):
    colors = {1: "#c0392b", 2: "#e67e22", 4: "#0f766e", 8: "#1f3b73", 16: "#6b21a8"}
    rho_off_real = {k: v * ES for k, v in lam_real.items()}   # offered ρ = λE[S]

    with plt.rc_context(V2_RC):
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
        # 行: 0=per-server ρ, 1=offered ρ。列: 0=W_q, 1=L_q
        rows = [
            ("rho", "per-server utilization  $\\rho = \\lambda E[S]/c$"),
            ("rho_off", "offered load  $\\rho_{\\mathrm{off}} = \\lambda E[S]$"),
        ]
        cols = [
            ("Wq_mean", "mean queue wait  $W_q$  [s]"),
            ("Lq_mean", "mean queue length  $L_q$  [jobs]"),
        ]
        for ri, (xkey, xlab) in enumerate(rows):
            for ci, (ykey, ylab) in enumerate(cols):
                ax = axes[ri, ci]
                for c in C_LIST:
                    pts = sweep[c]
                    xs = [p[xkey] for p in pts]
                    ys = [p[ykey] for p in pts]
                    lo = [p[ykey.replace("_mean", "_lo")] for p in pts]
                    hi = [p[ykey.replace("_mean", "_hi")] for p in pts]
                    ax.plot(xs, ys, "-o", ms=3, lw=1.5, color=colors[c], label=f"c={c}")
                    ax.fill_between(xs, lo, hi, color=colors[c], alpha=0.15, linewidth=0)
                ax.set_yscale("log")
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
                ax.grid(True, which="both", alpha=0.25)
                # offered ρ 行のみ実測動作点の縦線（offered ρ は c 非依存で単値）
                if xkey == "rho_off":
                    ax.axvline(rho_off_real["arb"], color="0.25", ls="--", lw=1.3)
                    ax.axvline(rho_off_real["3chain"], color="0.45", ls=":", lw=1.3)
                    ax.text(rho_off_real["arb"], ax.get_ylim()[1],
                            " $\\rho_{arb}$", rotation=90, va="top", ha="right",
                            fontsize=9, color="0.25")
                    ax.text(rho_off_real["3chain"], ax.get_ylim()[1],
                            " $\\rho_{3ch}$", rotation=90, va="top", ha="left",
                            fontsize=9, color="0.45")
                if ci == 0:  # W_q パネルに 0.5s / 5s 参照線
                    for thr, cc in [(0.5, "#888"), (5.0, "#555")]:
                        ax.axhline(thr, color=cc, ls="-", lw=0.9, alpha=0.7)
                        ax.text(ax.get_xlim()[0], thr, f" {thr}s ", va="bottom",
                                ha="left", fontsize=8.5, color=cc)
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                          fontsize=9.5, title="parallelism", framealpha=0.9)
        axes[0, 0].set_title("(a) per-server $\\rho$", fontsize=12)
        axes[0, 1].set_title("(b) per-server $\\rho$", fontsize=12)
        axes[1, 0].set_title("(c) offered $\\rho$", fontsize=12)
        axes[1, 1].set_title("(d) offered $\\rho$", fontsize=12)
        fig.suptitle("T4 CCTP pooled M/G/c: $W_q$ / $L_q$ vs utilization $\\rho$  "
                     "(63/37 mixture empirical service, FCFS, seed=42)", y=1.0)
        fig.tight_layout()
        fig.savefig(OUT_PNG_V2_F1, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 図2: dedicated two-class（fast系/slow系）を各系自身の ρ 横軸で（2 パネル）
# --------------------------------------------------------------------------- #
def make_fig2_dedicated(sweepB, real_B):
    from matplotlib.lines import Line2D

    with plt.rc_context(V2_RC):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
        panels = [
            (axes[0], "Wq_mean", "mean queue wait  $W_q$  [s]",
             "(left)  $W_q$  vs per-system utilization  $\\rho_i$"),
            (axes[1], "Lq_mean", "mean queue length  $L_q$  [jobs]",
             "(right)  $L_q$  vs per-system utilization  $\\rho_i$"),
        ]
        for ax, ykey, ylab, title in panels:
            for sysname, col in [("fast", COL_FAST), ("slow", COL_SLOW)]:
                for c in C_LIST_B:
                    pts = sweepB[sysname][c]
                    xs = [p["rho"] for p in pts]
                    ys = [p[ykey] for p in pts]
                    lo = [p[ykey.replace("_mean", "_lo")] for p in pts]
                    hi = [p[ykey.replace("_mean", "_hi")] for p in pts]
                    ax.plot(xs, ys, LS_C[c], marker="o", ms=2.5, lw=1.6, color=col)
                    ax.fill_between(xs, lo, hi, color=col, alpha=0.12, linewidth=0)
            ax.set_yscale("log")
            ax.set_xlabel("per-system utilization  $\\rho_i = \\lambda_i E[S_i]/c_i$")
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=12)
            ax.grid(True, which="both", alpha=0.25)
            # 実測動作点（c=1 基本ケース）: fast/slow を色、arb/3ch を線種で
            for rate, rls in [("arb", "--"), ("3chain", ":")]:
                for sysname, col in [("fast", COL_FAST), ("slow", COL_SLOW)]:
                    rho = real_B[rate][sysname][1]["rho"]   # c=1
                    ax.axvline(rho, color=col, ls=rls, lw=1.2, alpha=0.8)
            if ykey == "Wq_mean":
                for thr, cc in [(0.5, "#888"), (5.0, "#555")]:
                    ax.axhline(thr, color=cc, ls="-", lw=0.9, alpha=0.7)
                    ax.text(ax.get_xlim()[0], thr, f" {thr}s ", va="bottom",
                            ha="left", fontsize=8.5, color=cc)

        # プロット外へ凡例（系×並列度、および実測動作点の線種）
        handles = [
            Line2D([0], [0], color=COL_FAST, lw=2.4, label="fast system  ($E[S_f]$=3.70s)"),
            Line2D([0], [0], color=COL_SLOW, lw=2.4, label="slow system  ($E[S_s]$=8.19s)"),
            Line2D([0], [0], color="0.3", lw=1.8, ls="-", label="$c_i$ = 1 (solid)"),
            Line2D([0], [0], color="0.3", lw=1.8, ls="--", label="$c_i$ = 2 (dashed)"),
            Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="op. pt. $\\lambda_{arb}$"),
            Line2D([0], [0], color="0.3", lw=1.2, ls=":", label="op. pt. $\\lambda_{3ch}$"),
        ]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
                   fontsize=10.5, framealpha=0.95, title="legend")
        fig.suptitle("T4 CCTP dedicated two-class: fast vs slow queue "
                     "($W_q$ / $L_q$ vs $\\rho_i$, M/G/$c$, FCFS, empirical service, seed=42)",
                     y=1.02)
        fig.tight_layout(rect=[0, 0, 0.86, 1])
        fig.savefig(OUT_PNG_V2_F2, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main_v2():
    fast, slow = load_service_pools()
    ES, ES2 = expected_service(fast, slow)              # pooled 混合
    SCV = ES2 / ES ** 2 - 1.0
    ES_f, ES2_f, SCV_f = moments(fast)                  # fast 系
    ES_s, ES2_s, SCV_s = moments(slow)                  # slow 系
    print(f"[v2 service] pooled E[S]={ES:.4f} E[S2]={ES2:.4f} SCV={SCV:.3f}")
    print(f"[v2 service] fast   E[S]={ES_f:.4f} E[S2]={ES2_f:.4f} SCV={SCV_f:.4f}")
    print(f"[v2 service] slow   E[S]={ES_s:.4f} E[S2]={ES2_s:.4f} SCV={SCV_s:.4f}")

    info, lam_arb, lam_3chain = estimate_lambdas()
    lam_real = {"arb": lam_arb, "3chain": lam_3chain}
    print(f"[v2 lambda] lam_arb={lam_arb:.5f} lam_3chain={lam_3chain:.5f} tx/s")

    # ---- モデルA: pooled M/G/c を per-server ρ グリッドで掃引 ----
    sweep = {c: [] for c in C_LIST}
    for c in C_LIST:
        for rho_ps in RHO_GRID:
            lam = rho_ps * c / ES
            pt = sim_point(lam, c, fast, slow)
            pt["rho_off"] = lam * ES
            pt["rho_off_lo"] = pt["rho_off"]
            pt["rho_off_hi"] = pt["rho_off"]
            sweep[c].append(pt)
        print(f"[v2 A] pooled c={c} points={len(sweep[c])}")

    # ---- モデルB: fast系/slow系を各系自身の per-server ρ グリッドで掃引 ----
    sys_def = {"fast": (fast, ES_f), "slow": (slow, ES_s)}
    sweepB = {name: {c: [] for c in C_LIST_B} for name in sys_def}
    for name, (pool, es) in sys_def.items():
        for c in C_LIST_B:
            for rho_ps in RHO_GRID:
                lam_i = rho_ps * c / es
                sweepB[name][c].append(sim_point_single(lam_i, c, pool, es))
            print(f"[v2 B] {name} system c={c} points={len(sweepB[name][c])}")

    # ---- 実測動作点での各系 ρ・W_q・L_q（sim） ----
    #   全体 λ を 0.63/0.37 に分流 → λ_fast=0.63λ, λ_slow=0.37λ（独立ポアソン）
    real_B = {}   # rate -> sysname -> c -> point
    for rate, lam in lam_real.items():
        real_B[rate] = {"fast": {}, "slow": {}}
        splits = {"fast": (P_FAST * lam, fast, ES_f), "slow": (P_SLOW * lam, slow, ES_s)}
        for name, (lam_i, pool, es) in splits.items():
            for c in C_LIST_B:
                real_B[rate][name][c] = sim_point_single(lam_i, c, pool, es)

    # pooled 実測動作点（offered ρ・per-server ρ）
    real_pool = {}
    for rate, lam in lam_real.items():
        real_pool[rate] = {c: sim_point(lam, c, fast, slow) for c in C_LIST}

    # ---- results.csv ----
    #   Wq_mean_s / Wq_lo_s / Wq_hi_s / Lq_* は従来どおり。
    #   P_wait, P_wait_gt2p5, Wq_p95_s は待ち時間分布の追加統計量（5 反復平均）。
    def _row(model, system, p, c, rho_off=""):
        return {"model": model, "system": system, "lambda": round(p["lambda"], 6),
                "c": c, "rho_perserver": round(p["rho"], 4),
                "rho_offered": rho_off,
                "Wq_mean_s": round(p["Wq_mean"], 5),
                "Wq_lo_s": round(p["Wq_lo"], 5), "Wq_hi_s": round(p["Wq_hi"], 5),
                "Lq_mean": round(p["Lq_mean"], 5),
                "Lq_lo": round(p["Lq_lo"], 5), "Lq_hi": round(p["Lq_hi"], 5),
                "P_wait": round(p["P_wait"], 5),
                "P_wait_gt2p5": round(p["P_wait_gt2p5"], 5),
                "Wq_p95_s": round(p["Wq_p95"], 5)}

    rows = []
    for c in C_LIST:
        for p in sweep[c]:
            rows.append(_row("A_pooled", "mixture", p, c, round(p["rho_off"], 4)))
    for name in sys_def:
        for c in C_LIST_B:
            for p in sweepB[name][c]:
                rows.append(_row("B_dedicated", name, p, c))
    # 実測動作点（λ_arb / λ_3chain）は ρ 掃引グリッド上に無いので個別に走らせた結果を追加。
    for rate in ("arb", "3chain"):
        for c in C_LIST:
            p = real_pool[rate][c]
            rows.append(_row(f"A_pooled_oppoint_{rate}", "mixture", p, c,
                             round(p["lambda"] * ES, 4)))
        for name in ("fast", "slow"):
            for c in C_LIST_B:
                rows.append(_row(f"B_dedicated_oppoint_{rate}", name, real_B[rate][name][c], c))
    with open(OUT_CSV_V2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {OUT_CSV_V2}")

    # ---- 図 ----
    make_fig1_pooled(sweep, ES, lam_real)
    print(f"[write] {OUT_PNG_V2_F1}")
    make_fig2_dedicated(sweepB, real_B)
    print(f"[write] {OUT_PNG_V2_F2}")

    # ---- summary.md ----
    write_summary_v2(fast, slow, ES, ES2, SCV, ES_f, ES2_f, SCV_f, ES_s, ES2_s, SCV_s,
                     info, lam_real, real_pool, real_B, sweepB)
    print(f"[write] {OUT_MD_V2}")


def write_summary_v2(fast, slow, ES, ES2, SCV, ES_f, ES2_f, SCV_f, ES_s, ES2_s, SCV_s,
                     info, lam_real, real_pool, real_B, sweepB):
    lam_arb, lam_3chain = lam_real["arb"], lam_real["3chain"]
    L = []
    L.append("# T4 CCTP 待ち行列分析 v2 — 稼働率 ρ 横軸 ＋ fast/slow 専用二系統\n")
    L.append("共同研究者（待ち行列理論の教授）の改訂依頼に対応: "
             "(1) 横軸を到着率 λ でなく **稼働率 ρ = λE[S]** に統一、"
             "(2) 63/37 の二値振り分けを踏まえ **fast 処理の待ち行列**と"
             "**slow 処理の待ち行列**を別々の系として評価。"
             "v1（result/T4_cctp/queueing_sim_*）は保持し、本 v2 を別ファイルで追加。\n")

    L.append("## 1. 稼働率 ρ の定義（2 通りを併用）\n")
    L.append("- **offered load（提供負荷）** $\\rho_{\\mathrm{off}} = \\lambda E[S]$: "
             "系に投入される仕事量そのもの。c に依存しない単一量で、実測動作点の位置を "
             "c 横断で一意に示せる（安定条件は $\\rho_{\\mathrm{off}} < c$）。")
    L.append("- **per-server utilization（サーバ当たり稼働率）** "
             "$\\rho = \\lambda E[S]/c$: 各サーバの実稼働率で、安定条件は常に $\\rho<1$。"
             "c 別曲線を同一スケールで重ねて比較できる（M/M/c・M/G/c の定石軸）。")
    L.append("- 使い分け: **c 間の比較可能性**を重視する図では per-server ρ を主軸に、"
             "**実負荷の絶対位置**（実測レートがどの ρ に居るか）を語る図では offered ρ を用いる。"
             "図1（pooled）は両方を上下段に併載した。\n")

    L.append("## 2. サービス時間モーメント（実測経験分布）\n")
    L.append("| 系 | n | E[S] [s] | E[S²] [s²] | SCV=$C_s^2$ |")
    L.append("|---|---|---|---|---|")
    L.append(f"| pooled（63/37 混合） | {fast.size + slow.size} | {ES:.4f} | {ES2:.4f} | {SCV:.4f} |")
    L.append(f"| fast 系 | {fast.size} | {ES_f:.4f} | {ES2_f:.4f} | {SCV_f:.4f} |")
    L.append(f"| slow 系 | {slow.size} | {ES_s:.4f} | {ES2_s:.4f} | {SCV_s:.4f} |")
    L.append("")
    L.append(f"fast/slow はいずれも群内 SCV が小さく（{SCV_f:.3f} / {SCV_s:.3f}、ほぼ決定的）、"
             f"混合すると二値の隔たりで SCV={SCV:.3f} に上がる。"
             f"**slow 系の E[S] は fast 系の {ES_s/ES_f:.2f} 倍**であり、"
             "同じ λ でも slow 系のほうが稼働率 ρ が先に立ち上がる（後述 §5）。\n")

    L.append("## 3. モデル定義\n")
    L.append("### モデルA（pooled・v1 踏襲）\n")
    L.append("単一の **M/G/c**。サービス時間 = 確率 0.63 で fast 群・0.37 で slow 群の"
             "経験分布リサンプル（混合）。到着はポアソン λ。図1は per-server ρ（上段）と "
             "offered ρ（下段）の両軸で W_q・L_q を提示。\n")
    L.append("### モデルB（dedicated two-class・新規）\n")
    L.append("全体ポアソン λ を確率 0.63 / 0.37 で分流し、"
             "**λ_fast = 0.63λ、λ_slow = 0.37λ**（分流された各過程は独立ポアソン）。")
    L.append("- **fast 系** = M/G/$c_f$（サービス = fast 群経験分布、$E[S_f]$="
             f"{ES_f:.3f}s）")
    L.append("- **slow 系** = M/G/$c_s$（サービス = slow 群経験分布、$E[S_s]$="
             f"{ES_s:.3f}s）")
    L.append("各系を独立に λ 掃引し、**その系自身の稼働率 $\\rho_i = \\lambda_i E[S_i]/c_i$** を"
             "横軸に W_q・L_q を描く。基本ケース $c_f=c_s=1$、併走で $c=2$。"
             f"各 (λ,c) は {N_JOBS:,} ジョブ×{N_REPS} 反復（seed={SEED} 固定）、95% 区間併記。\n")

    L.append("## 4. 実測動作点での ρ・W_q・L_q（シミュレーション）\n")
    L.append(f"実測到着率 **λ_arb = {lam_arb:.5f}**、**λ_3chain = {lam_3chain:.5f} tx/s**"
             "（v1 と同一・悉皆イベントの窓実時間長から推定、CSV は read-only）。"
             "モデルB は全体 λ を 0.63/0.37 に分流した各系の値。\n")

    L.append("### pooled（モデルA, offered ρ = λE[S]）\n")
    L.append("| rate | λ [tx/s] | offered ρ | ρ (c=1) | W_q [s] (c=1) | ρ (c=2) | W_q [s] (c=2) | L_q (c=1) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rate, lam in lam_real.items():
        p1 = real_pool[rate][1]
        p2 = real_pool[rate][2]
        L.append(f"| {rate} | {lam:.5f} | {lam*ES:.3f} | {p1['rho']:.3f} | "
                 f"{p1['Wq_mean']:.4f} | {p2['rho']:.3f} | {p2['Wq_mean']:.4f} | {p1['Lq_mean']:.4f} |")
    L.append("")

    L.append("### dedicated 二系統（モデルB, 各系 ρ_i = λ_i E[S_i]/c_i, 基本 c=1）\n")
    L.append("| rate | 系 | λ_i [tx/s] | ρ_i (c=1) | W_q [s] (c=1) | L_q (c=1) | ρ_i (c=2) | W_q [s] (c=2) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rate, lam in lam_real.items():
        for name, frac in [("fast", P_FAST), ("slow", P_SLOW)]:
            q1 = real_B[rate][name][1]
            q2 = real_B[rate][name][2]
            L.append(f"| {rate} | {name} | {frac*lam:.5f} | {q1['rho']:.3f} | "
                     f"{q1['Wq_mean']:.4f} | {q1['Lq_mean']:.4f} | {q2['rho']:.3f} | {q2['Wq_mean']:.4f} |")
    L.append("")
    # slow が先に立つことの定量
    rf1 = real_B["3chain"]["fast"][1]["rho"]
    rs1 = real_B["3chain"]["slow"][1]["rho"]
    L.append(f"→ 例えば λ_3chain（{lam_3chain:.5f}）で c=1 のとき、"
             f"**slow 系 ρ={rs1:.3f} は fast 系 ρ={rf1:.3f} の {rs1/rf1:.2f} 倍**。"
             "同じ全体到着でも slow 系の稼働率が先行して高くなる（§7 の帰結）。\n")

    L.append("## 5. 理論照合（P-K 厳密 / M/M/c 近似）\n")
    L.append("各 dedicated 系は M/G/1（c=1）または M/G/c（c=2）。"
             "教授の講義流儀に沿い、**c=1 は Pollaczek–Khinchine 厳密式** "
             "$W_q=\\lambda E[S^2]/(2(1-\\rho))$、**c=2 は M/M/c 近似**"
             "（Erlang-C の $W_q^{M/M/c}$ に Allen–Cunneen 補正 $(C_a^2+C_s^2)/2$、"
             "Poisson 到着 $C_a^2=1$ を掛けた M/G/c 近似）とシミュレーションを比較する。"
             "なお c=1 では $(1+SCV)/2\\cdot W_q^{M/M/1}$ が P-K と厳密一致する。\n")

    def theory_rows(name, pool, es, es2, scv):
        out = []
        # ρ を 0.3/0.5/0.7/0.9 で照合
        for c in C_LIST_B:
            for rho in [0.3, 0.5, 0.7, 0.9]:
                lam_i = rho * c / es
                sim = sim_point_single(lam_i, c, pool, es)
                if c == 1:
                    th = pk_wq_mg1(lam_i, es, es2)
                    tag = "P-K"
                else:
                    th = wq_mgc_approx(lam_i, es, c, scv)
                    tag = "M/M/c+AC"
                err = (sim["Wq_mean"] - th) / th * 100.0 if th > 0 else float("nan")
                out.append((name, c, rho, sim["Wq_mean"], th, tag, err))
        return out

    L.append("| 系 | c | ρ_i | W_q sim [s] | W_q 理論 [s] | 理論式 | 相対差 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, pool, es, es2, scv in [
        ("fast", fast, ES_f, ES2_f, SCV_f), ("slow", slow, ES_s, ES2_s, SCV_s)]:
        for (nm, c, rho, sim, th, tag, err) in theory_rows(name, pool, es, es2, scv):
            L.append(f"| {nm} | {c} | {rho:.1f} | {sim:.4f} | {th:.4f} | {tag} | {err:+.1f}% |")
    L.append("")
    L.append("シミュレーション（経験分布リサンプル）と理論式（P-K は経験 E[S²] を使用）は"
             "全 ρ 域で整合。c=2 の M/M/c 近似は Allen–Cunneen 補正で群内 SCV≪1 を反映し、"
             "指数サービス仮定の素の M/M/c より小さい W_q を与える（決定的に近い実サービスと整合）。\n")

    L.append("## 6. 図\n")
    L.append("- **fig1**（`queueing_sim_v2_fig1_pooled.png`, 2×2, 300dpi）: pooled M/G/c の "
             "W_q・L_q。上段 = per-server ρ（c 比較用・主図候補）、下段 = offered ρ"
             "（実測動作点 $\\rho_{arb},\\rho_{3ch}$ の縦線付き）。W_q パネルに 0.5s/5s 参照線。")
    L.append("- **fig2**（`queueing_sim_v2_fig2_dedicated.png`, 2 パネル, 300dpi）: dedicated "
             "二系統の W_q・L_q vs 各系 ρ_i。fast=teal / slow=red、c=1 実線 / c=2 破線、"
             "95% 帯付き。実測動作点は fast/slow 色 × arb(破線)/3ch(点線) の縦線で明示。"
             "凡例はプロット外（右）に配置。\n")

    L.append("## 7. 解釈: 行列が先に立つのは slow 系\n")
    L.append(f"slow 系のサービス時間は fast 系の {ES_s/ES_f:.2f} 倍（{ES_s:.2f}s vs {ES_f:.2f}s）。"
             "全体到着 λ を 0.63/0.37 に分流すると、単位到着あたりの負荷寄与は "
             f"slow 側で $0.37\\times{ES_s:.2f}={P_SLOW*ES_s:.2f}$、"
             f"fast 側で $0.63\\times{ES_f:.2f}={P_FAST*ES_f:.2f}$ となり、"
             "**同じ全体 λ でも slow 系の ρ が先に上がる**。")
    # 定量: 各系が ρ=0.9 に達する全体 λ
    lam_fast_09 = 0.9 * 1 / ES_f / P_FAST     # fast系c=1 が ρ=0.9 になる全体 λ
    lam_slow_09 = 0.9 * 1 / ES_s / P_SLOW     # slow系c=1 が ρ=0.9 になる全体 λ
    L.append(f"定量的には、c=1 で各系が ρ=0.9（行列が急伸する領域）に達する全体到着率は、"
             f"**slow 系で λ≈{lam_slow_09:.4f} tx/s、fast 系で λ≈{lam_fast_09:.4f} tx/s**。"
             f"slow 系のほうが {lam_fast_09/lam_slow_09:.2f} 倍低い λ で飽和に近づく"
             "＝ **系全体のボトルネックは slow 系**であり、"
             "容量設計は slow 系（$c_s$ の増強）に振るのが効くことを示す。")
    L.append(f"実測動作点は両系とも低 ρ 域（λ_3chain・c=1 で slow ρ={rs1:.3f}, fast ρ={rf1:.3f}）"
             "にあり、v1 の結論（実トラフィックは行列立ち上がり領域の十分下）と整合する。"
             "本 v2 は、その余裕が **slow 系側で先に食い潰される**構造を稼働率軸で明示した。\n")

    L.append("## 8. 限界（v1 と共通）\n")
    L.append("- λ_3chain は Iris 総負荷の上界側（サービス時間は Arbitrum 発 deposit の "
             "iris_wait から推定）。- FCFS 仮定（実 Iris は優先度・バッチの可能性）。"
             "- c は実効並列度の下界の抽象化で物理サーバ台数ではない。"
             f"- サービス時間標本は fast n={fast.size} / slow n={slow.size} で裾は希少。"
             "- モデルB は fast/slow を完全独立系と仮定（実際は共有資源の可能性があり、"
             "その場合は結合系となり本モデルは各系を分離した上界的評価）。\n")

    OUT_MD_V2.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "v2"
    if mode == "v1":
        main()
    elif mode == "v2":
        main_v2()
    else:
        main()
        main_v2()
