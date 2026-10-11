"""Per-deposit vote timelines and summary statistics for the 36 collected deposits.

Inputs : votes_raw.csv, post_credit_votes.csv, credit_blocks.csv, deposit_latency.csv (see fetch_votes.SRC)
Outputs: results_votes.json, report_votes.md, per_deposit_votes.csv,
         figures/fig7a_vote_timeline.png, figures/fig7b_A_hist.png
"""
import json, os
import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
MAX_EXP = 77  # largest A (and largest latency); excluded for shape statistics of A

# ------------------------------------------------------------------ load
from fetch_votes import SRC  # locates deposit_latency.csv in the working tree or the public repo
src = pd.read_csv(SRC).set_index("experiment_id")
raw = pd.read_csv(P("votes_raw.csv"))
post = pd.read_csv(P("post_credit_votes.csv"))
cb = pd.read_csv(P("credit_blocks.csv")).set_index("experiment_id")
post = post[post.is_target_deposit.astype(bool)]
raw["tx_error"] = raw.tx_error.fillna("")
post["tx_error"] = post.tx_error.fillna("")
raw["src"], post["src"] = "scan", "post_credit"
# votes_raw rows are appended walking down from the credit block, txs in block order;
# keep a sequence number so ties inside one block keep their in-block order
raw["seq"] = -np.arange(len(raw))
post["seq"] = np.arange(len(post))
cols = ["experiment_id", "validator", "hl_block", "hl_block_time_ms", "tx_error", "src", "seq"]
votes = pd.concat([raw[cols], post[cols]], ignore_index=True)
exps = sorted(raw.experiment_id.unique())

# ------------------------------------------------------------------ per deposit
rows = []
for e in exps:
    s = src.loc[e]
    t0 = s["local_send_time(ns)"] / 1e6
    t1, t3 = s["arb_block_timestamp(ms)"], s["hl_ledger_time(ms)"]
    v = votes[votes.experiment_id == e]
    ok = v[v.tx_error == ""].sort_values(["hl_block_time_ms", "hl_block", "seq"])
    rej = v[v.tx_error != ""].sort_values("hl_block_time_ms")
    first, third = ok.iloc[0], ok.iloc[2]
    r4 = rej.iloc[0] if len(rej) else None
    rows.append({
        "experiment_id": e, "date": str(s["experiment_start_iso"])[:10],
        "n_votes": len(v), "n_valid": len(ok), "n_rejected": len(rej), "n_voters": v.validator.nunique(),
        "t0_ms": t0, "t1_ms": t1, "t_first_ms": first.hl_block_time_ms, "t_third_ms": third.hl_block_time_ms, "t3_ms": t3,
        "first_block": first.hl_block, "third_block": third.hl_block, "credit_block": cb.loc[e, "credit_l1_height"],
        "A_s": (first.hl_block_time_ms - t1) / 1e3,
        "B_s": (third.hl_block_time_ms - first.hl_block_time_ms) / 1e3,
        "C_s": (t3 - third.hl_block_time_ms) / 1e3,
        "L_s": (t3 - t0) / 1e3,
        "G_s": (t3 - t1) / 1e3,
        "fourth_validator": r4.validator if r4 is not None else "",
        "fourth_delay_s": (r4.hl_block_time_ms - t3) / 1e3 if r4 is not None else np.nan,
        "fourth_in_credit_block": bool(r4 is not None and r4.hl_block == cb.loc[e, "credit_l1_height"]),
        "first_vote_t1_rel_s": (first.hl_block_time_ms - t1) / 1e3,
    })
pdv = pd.DataFrame(rows)
pdv.to_csv(P("per_deposit_votes.csv"), index=False)

# ------------------------------------------------------------------ voters
votes = votes.merge(pdv[["experiment_id"]], on="experiment_id")
votes = votes.sort_values(["experiment_id", "hl_block_time_ms", "hl_block", "seq"])
votes["rank"] = votes.groupby("experiment_id").cumcount() + 1
voters = (votes.groupby("validator")
          .agg(appearances=("experiment_id", "nunique"), rejected=("tx_error", lambda x: int((x != "").sum())),
               mean_rank=("rank", "mean"))
          .reset_index().sort_values("mean_rank"))
known4 = set(voters.validator)

# ------------------------------------------------------------------ fits for A
A_all = pdv.A_s.values
A35 = pdv.loc[pdv.experiment_id != MAX_EXP, "A_s"].values


def ll_unif_noise(p, x):
    a, w, sd = p
    if w <= 0 or sd <= 0:
        return -np.inf
    d = (stats.norm.cdf((x - a) / sd) - stats.norm.cdf((x - a - w) / sd)) / w
    return np.sum(np.log(np.maximum(d, 1e-300)))


SD_FLOOR = 1 / np.sqrt(12)  # t1 is an Arbitrum block timestamp with 1 s resolution -> quantisation SD >= 1/sqrt(12) s


def fit_unif_noise(x, sd_floor=SD_FLOOR):
    """MLE of U(a, a+w) + N(0, sd^2) with sd >= sd_floor.
    Without the floor the likelihood is maximised on the boundary sd -> 0 (pure uniform on [min, max]);
    the interior local maximum is reported as the fit, the boundary value separately."""
    best = None
    for a0 in np.linspace(x.min() - 1, x.min() + 1, 5):
        for sd0 in (0.5, 1.0, 2.0):
            r = optimize.minimize(lambda p: -ll_unif_noise(p, x), [a0, x.max() - a0, sd0], method="L-BFGS-B",
                                  bounds=[(None, None), (1e-3, None), (sd_floor, None)])
            if best is None or r.fun < best.fun:
                best = r
    a, w, sd = best.x
    return {"start_s": a, "width_s": w, "sd_s": sd, "loglik": -best.fun, "k": 3, "sd_floor_s": sd_floor,
            "sd_at_floor": bool(sd <= sd_floor * 1.001),
            "boundary_pure_uniform": {"start_s": x.min(), "width_s": x.max() - x.min(),
                                      "loglik": -len(x) * np.log(x.max() - x.min()), "k": 2}}


def fit_scipy(dist, x, **kw):
    p = dist.fit(x, **kw)
    return {"params": [float(v) for v in p], "loglik": float(np.sum(dist.logpdf(x, *p))), "k": len(p) - len(kw)}


def fits(x):
    out = {"uniform_plus_normal": fit_unif_noise(x),
           "gumbel_r": fit_scipy(stats.gumbel_r, x),
           "lognorm": fit_scipy(stats.lognorm, x),  # 3-parameter (free loc)
           "lognorm_loc0": fit_scipy(stats.lognorm, x, floc=0),
           "norm": fit_scipy(stats.norm, x)}
    for v in out.values():
        v["aic"] = 2 * v["k"] - 2 * v["loglik"]
    return out


F35, F36 = fits(A35), fits(A_all)


def cdf_unif_noise(x, a, w, sd):
    # CDF of U(a, a+w) + N(0, sd): integral of the density above
    x = np.asarray(x, float)
    g = lambda z: z * stats.norm.cdf(z / sd) + sd * stats.norm.pdf(z / sd)
    return (g(x - a) - g(x - a - w)) / w


u = F35["uniform_plus_normal"]
ks35 = stats.kstest(A35, lambda x: cdf_unif_noise(x, u["start_s"], u["width_s"], u["sd_s"]))

# ------------------------------------------------------------------ summary
varL = np.var(pdv.L_s, ddof=1)
same_block = int((pdv.first_block == pdv.third_block).sum())
fd = pdv.fourth_delay_s
res = {
    "n_deposits": int(len(pdv)), "n_days": int(pdv.date.nunique()),
    "voters": voters.to_dict("records"),
    "n_distinct_voters": int(len(known4)),
    "every_deposit_has_same_4": bool(all(set(votes[votes.experiment_id == e].validator) == known4 for e in exps)),
    "valid_votes_per_deposit": pdv.n_valid.value_counts().to_dict(),
    "C_zero_count": int((pdv.C_s == 0).sum()),
    "B": {"mean_s": pdv.B_s.mean(), "max_s": pdv.B_s.max(), "sd_s": pdv.B_s.std(ddof=1),
          "same_block_first_and_third": same_block},
    "A": {"mean_s": pdv.A_s.mean(), "sd_s": pdv.A_s.std(ddof=1), "min_s": pdv.A_s.min(), "max_s": pdv.A_s.max(),
          "max_exp": int(pdv.loc[pdv.A_s.idxmax(), "experiment_id"])},
    "L": {"mean_s": pdv.L_s.mean(), "sd_s": pdv.L_s.std(ddof=1)},
    "variance_ratio": {"A_over_L": np.var(pdv.A_s, ddof=1) / varL, "B_over_L": np.var(pdv.B_s, ddof=1) / varL,
                       "A_over_L_excl_max": np.var(A35, ddof=1) / np.var(pdv.loc[pdv.experiment_id != MAX_EXP, "L_s"], ddof=1),
                       "C_over_L": np.var(pdv.C_s, ddof=1) / varL,
                       "t1_minus_t0_over_L": np.var((pdv.t1_ms - pdv.t0_ms) / 1e3, ddof=1) / varL},
    "corr_t1t0_A": stats.pearsonr((pdv.t1_ms - pdv.t0_ms) / 1e3, pdv.A_s)[0],
    "corr": {"A_B": stats.pearsonr(pdv.A_s, pdv.B_s)[0], "A_L": stats.pearsonr(pdv.A_s, pdv.L_s)[0],
             "B_L": stats.pearsonr(pdv.B_s, pdv.L_s)[0]},
    "A35": {"n": int(len(A35)), "mean_s": A35.mean(), "sd_s": A35.std(ddof=1), "min_s": A35.min(), "max_s": A35.max(),
            "skew": stats.skew(A35), "excess_kurtosis": stats.kurtosis(A35),
            "skew_unbiased": stats.skew(A35, bias=False), "excess_kurtosis_unbiased": stats.kurtosis(A35, bias=False),
            "fits": F35, "ks_uniform_plus_normal": {"D": ks35.statistic, "p": ks35.pvalue}},
    "A36_fits": F36,
    "fourth_vote": {"in_credit_block": int(pdv.fourth_in_credit_block.sum()),
                    "after_credit": int((~pdv.fourth_in_credit_block & fd.notna()).sum()),
                    "after_credit_delay_min_s": fd[~pdv.fourth_in_credit_block].min(),
                    "after_credit_delay_max_s": fd[~pdv.fourth_in_credit_block].max(),
                    "missing": int(fd.isna().sum())},
    "votes_from_outside_4_post_credit": int((~pd.read_csv(P("post_credit_votes.csv")).in_known_4.astype(bool)).sum()),
}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


res = _clean(res)
json.dump(res, open(P("results_votes.json"), "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ reference check
REF = [  # (label, computed, expected, tolerance)
    ("C=0 の件数", res["C_zero_count"], 36, 0),
    ("B 平均 [s]", res["B"]["mean_s"], 0.24, 0.005),
    ("B 最大 [s]", res["B"]["max_s"], 1.45, 0.005),
    ("B 同一ブロック件数", res["B"]["same_block_first_and_third"], 14, 0),
    ("A 平均 [s]", res["A"]["mean_s"], 9.84, 0.005),
    ("分散比 A/L（36件）", res["variance_ratio"]["A_over_L"], 1.00, 0.005),
    ("分散比 A/L（35件）", res["variance_ratio"]["A_over_L_excl_max"], 1.00, 0.01),
    ("分散比 B/L", res["variance_ratio"]["B_over_L"], 0.002, 0.0005),
    ("A35 平均 [s]", res["A35"]["mean_s"], 8.74, 0.005),
    ("A35 SD [s]", res["A35"]["sd_s"], 3.53, 0.005),
    ("A35 最小 [s]", res["A35"]["min_s"], 3.38, 0.005),
    ("A35 最大 [s]", res["A35"]["max_s"], 16.05, 0.005),
    ("A35 超過尖度", res["A35"]["excess_kurtosis"], -1.12, 0.005),
    ("一様+正規 開始 [s]", F35["uniform_plus_normal"]["start_s"], 3.57, 0.005),
    ("一様+正規 幅 [s]", F35["uniform_plus_normal"]["width_s"], 11.16, 0.005),
    ("一様+正規 SD [s]", F35["uniform_plus_normal"]["sd_s"], 1.01, 0.005),
    ("一様+正規 対数尤度", F35["uniform_plus_normal"]["loglik"], -90.2, 0.05),
    ("Gumbel 対数尤度", F35["gumbel_r"]["loglik"], -92.7, 0.05),
    ("対数正規 対数尤度", F35["lognorm"]["loglik"], -92.5, 0.05),
    ("4票目 着金ブロック内", res["fourth_vote"]["in_credit_block"], 30, 0),
    ("4票目 着金後の件数", res["fourth_vote"]["after_credit"], 6, 0),
    ("4票目 着金後の最小遅れ [s]", res["fourth_vote"]["after_credit_delay_min_s"], 0.22, 0.005),
    ("4票目 着金後の最大遅れ [s]", res["fourth_vote"]["after_credit_delay_max_s"], 0.93, 0.005),
    ("4者以外の票", res["votes_from_outside_4_post_credit"], 0, 0),
]
check = [(l, c, e, bool(abs(c - e) <= t)) for l, c, e, t in REF]
res["reference_check"] = _clean([{"item": l, "computed": c, "expected": e, "match": m} for l, c, e, m in check])
json.dump(res, open(P("results_votes.json"), "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
W = 16 / 2.54  # one Word column, ~16 cm

# fig7a: one row per deposit, sorted by A; broken x-axis so that #77 (~48 s) does not compress the rest
d = pdv.sort_values("A_s").reset_index(drop=True)
gmax_main = d.loc[d.experiment_id != MAX_EXP, "G_s"].max()
r77 = d[d.experiment_id == MAX_EXP].iloc[0]
lim_main = (0, np.ceil(gmax_main + d.fourth_delay_s.max()) + 1)
lim_far = (np.floor(r77.A_s) - 1, np.ceil(r77.G_s) + 1)
fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(W, W * 0.62),
                              gridspec_kw={"width_ratios": [lim_main[1] - lim_main[0], 1.6 * (lim_far[1] - lim_far[0])], "wspace": 0.06})
for axx in (ax, ax2):
    for i, r in d.iterrows():
        ev = votes[votes.experiment_id == r.experiment_id]
        okv = ev[ev.tx_error == ""]
        axx.plot([0, r.G_s], [i, i], color="0.88", lw=0.7, zorder=1)
        axx.scatter((okv.hl_block_time_ms - r.t1_ms) / 1e3, [i] * len(okv), s=5, color="#4C78A8", zorder=2,
                    label="valid vote" if i == 0 else None)
        axx.scatter([r.A_s], [i], s=24, facecolor="none", edgecolor="#4C78A8", lw=0.9, zorder=3,
                    label="first valid vote" if i == 0 else None)
        axx.scatter([r.G_s], [i], s=30, marker="|", color="#E45756", lw=1.6, zorder=4,
                    label="credit = 3rd valid vote ($t_3$)" if i == 0 else None)
        if not np.isnan(r.fourth_delay_s):
            axx.scatter([r.G_s + r.fourth_delay_s], [i], s=12, marker="x", color="0.3", lw=0.8, zorder=5,
                        label="rejected 4th vote" if i == 0 else None)
ax.set_xlim(*lim_main); ax2.set_xlim(*lim_far)
ax.set_xticks(np.arange(0, lim_main[1], 2)); ax2.set_xticks([48, 50])
ax.set_ylim(-1, len(d))
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"#{e}" for e in d.experiment_id], fontsize=5)
ax.set_ylabel("Deposit (sorted by $A$)")
ax2.spines["left"].set_visible(False); ax2.tick_params(left=False)
k = dict(color="k", clip_on=False, lw=0.8)  # break marks
for axx, xs in ((ax, 1), (ax2, 0)):
    axx.plot([xs - 0.012 * (1 if xs else 6), xs + 0.012 * (1 if xs else 6)], [-0.012, 0.012], transform=axx.transAxes, **k)
fig.supxlabel("Time since Arbitrum block timestamp $t_1$ (s)", fontsize=8, y=0.005)
ax.legend(loc="lower right", frameon=False, fontsize=7)
fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.11)
fig.savefig(P("figures", "fig7a_vote_timeline.png"), dpi=300)
plt.close(fig)

# fig7b: A histogram + uniform-plus-noise fit (35 deposits, #77 marked separately)
fig, ax = plt.subplots(figsize=(W, W * 0.42))
bins = np.arange(0, np.ceil(A35.max()) + 4, 1.0)
ax.hist(A35, bins=bins, color="#4C78A8", alpha=0.75, edgecolor="white", label=f"$A$ (n = {len(A35)})")
amax = pdv.loc[pdv.experiment_id == MAX_EXP, "A_s"].iloc[0]
ax.plot([], [], " ", label=f"#{MAX_EXP}: $A$ = {amax:.1f} s (off-scale, excluded from fit)")
xs = np.linspace(0, bins[-1], 600)
dens = (stats.norm.cdf((xs - u["start_s"]) / u["sd_s"]) - stats.norm.cdf((xs - u["start_s"] - u["width_s"]) / u["sd_s"])) / u["width_s"]
ax.plot(xs, dens * len(A35) * 1.0, color="#E45756", lw=1.4,
        label=f"Uniform({u['start_s']:.2f}, {u['start_s'] + u['width_s']:.2f}) + N(0, {u['sd_s']:.2f}$^2$)")
ax.set_xlim(0, bins[-1])
ax.set_xlabel("$A$ = first valid vote $-$ $t_1$ (s)")
ax.set_ylabel("Deposits")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(P("figures", "fig7b_A_hist.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ report
f = lambda x, n=2: f"{x:.{n}f}"
R = ["# 入金承認投票の時刻列（36件）", "",
     "`analysis_votes.py` の出力。全数値は `results_votes.json`、送金ごとの値は `per_deposit_votes.csv` にある。", "",
     "## 定義",
     "- t0 = ローカル送信時刻、t1 = Arbitrum ブロックの時刻、t3 = Hyperliquid の着金時刻（hl_ledger_time）",
     "- 有効票 = `tx_error` が空の VoteEthDepositAction。同じブロック内の順序はブロック内の取引順に従う",
     "- A = 最初の有効票 − t1、B = 3票目の有効票 − 最初の有効票、C = t3 − 3票目の有効票、L = t3 − t0",
     "- 4票目の遅れ = 拒否された票の時刻 − t3（着金ブロック内の場合は 0）", "",
     "## データ範囲",
     f"- 送金 {res['n_deposits']} 件（117件中。測定 {res['n_days']} 日分すべてを含む）",
     "- 収集方法：着金ブロックから遡り、16ブロック続けて票がなければ打ち切る。着金ブロックに4票目がなかった6件は、着金後15ブロックを追加で取得した", "",
     "## 投票者",
     f"- 投票アドレスは **{res['n_distinct_voters']} つ**。全件が同じ4アドレスか：{res['every_deposit_has_same_4']}",
     f"- 1件あたりの有効票：{res['valid_votes_per_deposit']}（全件3票で着金）", "",
     "| アドレス | 出現件数 | 拒否された数 | 平均順位（1〜4） |", "|---|---|---|---|"]
R += [f"| `{v['validator']}` | {v['appearances']} | {v['rejected']} | {f(v['mean_rank'])} |" for v in res["voters"]]
R += ["", "## 区間の統計",
      f"- C = 0：{res['C_zero_count']}/{res['n_deposits']} 件（着金時刻は3票目のブロック時刻と一致）",
      f"- B：平均 {f(res['B']['mean_s'])} s、SD {f(res['B']['sd_s'])} s、最大 {f(res['B']['max_s'])} s。最初の票と3票目が同じブロックの件数 {res['B']['same_block_first_and_third']}",
      f"- A（36件）：平均 {f(res['A']['mean_s'])} s、SD {f(res['A']['sd_s'])} s、範囲 {f(res['A']['min_s'])}〜{f(res['A']['max_s'])} s（最大は exp {res['A']['max_exp']}）",
      f"- L：平均 {f(res['L']['mean_s'])} s、SD {f(res['L']['sd_s'])} s",
      f"- 分散比（Var(X)/Var(L)）：A {f(res['variance_ratio']['A_over_L'], 3)}（exp {MAX_EXP} を除く35件では {f(res['variance_ratio']['A_over_L_excl_max'], 3)}）、B {f(res['variance_ratio']['B_over_L'], 4)}、"
      f"C {f(res['variance_ratio']['C_over_L'], 3)}、t1−t0 {f(res['variance_ratio']['t1_minus_t0_over_L'], 4)}",
      f"- 相関：A–B {f(res['corr']['A_B'], 3)}、A–L {f(res['corr']['A_L'], 3)}、B–L {f(res['corr']['B_L'], 3)}", "",
      f"## A の分布（exp {MAX_EXP} を除く35件）",
      f"- 平均 {f(res['A35']['mean_s'])} ± {f(res['A35']['sd_s'])} s、範囲 {f(res['A35']['min_s'])}〜{f(res['A35']['max_s'])} s",
      f"- 歪度 {f(res['A35']['skew'], 3)}、超過尖度 {f(res['A35']['excess_kurtosis'], 3)}（scipy 既定の標本推定量。不偏版はそれぞれ {f(res['A35']['skew_unbiased'], 3)}、{f(res['A35']['excess_kurtosis_unbiased'], 3)}）", "",
      "| モデル | パラメータ | 対数尤度 | AIC |", "|---|---|---|---|",
      f"| 一様＋正規揺らぎ | 開始 {f(u['start_s'])} s、幅 {f(u['width_s'])} s、SD {f(u['sd_s'])} s | {f(u['loglik'], 1)} | {f(u['aic'], 1)} |",
      f"| Gumbel（右に裾） | loc {f(F35['gumbel_r']['params'][0])}、scale {f(F35['gumbel_r']['params'][1])} | {f(F35['gumbel_r']['loglik'], 1)} | {f(F35['gumbel_r']['aic'], 1)} |",
      f"| 対数正規（3母数） | σ {f(F35['lognorm']['params'][0], 3)}、loc {f(F35['lognorm']['params'][1])}、scale {f(F35['lognorm']['params'][2])} | {f(F35['lognorm']['loglik'], 1)} | {f(F35['lognorm']['aic'], 1)} |",
      f"| 対数正規（loc=0） | σ {f(F35['lognorm_loc0']['params'][0], 3)}、scale {f(F35['lognorm_loc0']['params'][2])} | {f(F35['lognorm_loc0']['loglik'], 1)} | {f(F35['lognorm_loc0']['aic'], 1)} |",
      f"| （参考）純一様（σ→0 の境界） | 開始 {f(u['boundary_pure_uniform']['start_s'])} s、幅 {f(u['boundary_pure_uniform']['width_s'])} s | {f(u['boundary_pure_uniform']['loglik'], 1)} | {f(2 * 2 - 2 * u['boundary_pure_uniform']['loglik'], 1)} |",
      f"| 正規 | μ {f(F35['norm']['params'][0])}、σ {f(F35['norm']['params'][1])} | {f(F35['norm']['loglik'], 1)} | {f(F35['norm']['aic'], 1)} |",
      "", f"一様＋正規の当てはめでは σ ≥ {f(SD_FLOOR, 3)} s（t1 の1秒分解能による量子化の SD = 1/√12）という下限を置いた。"
      "下限を置かないと尤度は σ→0 の境界（min〜max の純一様）で最大になり、内部の極大（上の行）は局所解になる。"
      f"下限に張り付いたか：{u['sd_at_floor']}。",
      "", f"一様＋正規に対する KS 検定：D = {f(res['A35']['ks_uniform_plus_normal']['D'], 3)}、p = {f(res['A35']['ks_uniform_plus_normal']['p'], 3)}（パラメータを同じデータから推定しているため、p は参考値）",
      "", "## 4票目（拒否票）",
      f"- 着金ブロック内：{res['fourth_vote']['in_credit_block']} 件、着金後：{res['fourth_vote']['after_credit']} 件"
      f"（遅れ {f(res['fourth_vote']['after_credit_delay_min_s'])}〜{f(res['fourth_vote']['after_credit_delay_max_s'])} s）、見つからなかった件：{res['fourth_vote']['missing']} 件",
      f"- 着金後15ブロックの範囲にあった、4者以外からの票：{res['votes_from_outside_4_post_credit']} 票", "",
      "## 照合値との一致", "", "| 項目 | 計算値 | 照合値 | 一致 |", "|---|---|---|---|"]
R += [f"| {l} | {c:.4g} | {e} | {'✓' if m else '**✗**'} |" for l, c, e, m in check]
R += ["", "### 照合値との差の説明",
      f"- 分散比 A/L：36件では {f(res['variance_ratio']['A_over_L'], 3)}、exp {MAX_EXP} を除く35件では {f(res['variance_ratio']['A_over_L_excl_max'], 3)}。"
      "照合値「≈1.00」は、35件の値（0.99）とは小数第2位の丸めで一致する。36件の値は 1.02 で、0.005 の許容幅を外れる。"
      f"A/L が 1 を少し超えるのは、t1−t0 が A と弱く負に相関し（r = {f(res['corr_t1t0_A'], 2)}）、L の分散を少し打ち消しているため。"
      "t1 は秒単位に切り捨てられたブロック時刻なので、t1−t0 は平均で負になる。"
      "論文に載せる場合は「≈1（36件で1.02、35件で0.99）」と書けば両方を満たす。",
      "- 一様＋正規：σ に下限を置かない場合の最尤推定は σ→0 の境界解（純一様、開始 "
      f"{f(u['boundary_pure_uniform']['start_s'])} s、幅 {f(u['boundary_pure_uniform']['width_s'])} s、対数尤度 {f(u['boundary_pure_uniform']['loglik'], 1)}）になり、照合値とは一致しない。"
      "照合値（開始3.57、幅11.16、SD1.01、−90.2）は内部の局所極大と一致する。本スクリプトでは、t1 の1秒分解能に由来する σ ≥ 1/√12 s を下限として、この局所解を採用した。"
      "どちらの解でも、一様型が Gumbel と対数正規より対数尤度で上回る、という結論は変わらない。",
      "- 対数正規：照合値 −92.5 は3母数（loc も推定）の −92.48 と一致する。loc=0 に固定すると −92.59 になる。", ""]
R += ["", "## 図",
      "- `figures/fig7a_vote_timeline.png`：36件を A の小さい順に縦に並べたもの。横軸は t1 からの経過秒で、exp 77 を描くため右側を区切った別の目盛りにしている。青丸＝有効票（白抜き丸＝最初の票）、赤縦線＝着金（3票目）、灰×＝拒否された4票目",
      f"- `figures/fig7b_A_hist.png`：A のヒストグラム（35件、1秒幅）に、一様＋正規揺らぎの当てはめ曲線（件数に換算）を重ねたもの。exp {MAX_EXP}（A = {amax:.1f} s）は軸の範囲外なので凡例に値だけ示し、当てはめには使っていない", ""]
open(P("report_votes.md"), "w").write("\n".join(R))
for l, c, e, m in check:
    print(f"{'OK ' if m else 'NG '} {l}: {c:.6g} (ref {e})")
