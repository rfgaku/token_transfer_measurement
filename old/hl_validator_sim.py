#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hl_validator_sim.py

Hyperliquid validator-latency simulator + calibrator
- Deposit: Truncated lognormal base + deterministic top-m stalls (from observed top values)
- Withdraw residual X = T_withdraw - dispute_ref_s:
  top-q mean of N validator delays + clipped measurement noise
  with optional floor/cap/forced min outlier / forced max once.

This version guarantees:
- Robust CSV column auto-detection (fallback to "latency(ms)" etc.)
- Lock mode for deposit/withdraw params via JSON (no re-calibration)
- ALWAYS writes (OVERWRITE):
    result/validator_env_deposit_fit.csv
    result/validator_env_withdraw_res_fit.csv
    result/validator_env_best.json
  and prints "===== WRITE (OVERWRITE) =====" with paths.
- Avoids Python scoping bug: NO "import os" inside main().

"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Logging helpers
# -------------------------

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _info(msg: str) -> None:
    print(msg)


# -------------------------
# CSV IO
# -------------------------

def detect_latency_column(cols: List[str]) -> str:
    candidates = [
        "latency_ms", "latency(ms)", "latency", "ms", "delay_ms", "delay(ms)",
        "duration_ms", "duration(ms)"
    ]
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in cols:
        lc = c.lower()
        if ("latency" in lc) or ("delay" in lc) or ("duration" in lc):
            return c
    return cols[0]

def read_latency_csv(path: str, col: Optional[str]) -> Tuple[np.ndarray, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols:
            raise ValueError(f"No header found in {path}")

        use_col = col
        if use_col is None or use_col not in cols:
            det = detect_latency_column(cols)
            if use_col is not None and use_col not in cols:
                _warn(f"Column '{use_col}' not found in {path}. Fallback to detected column '{det}'.")
            use_col = det

        vals: List[float] = []
        for row in reader:
            s = row.get(use_col, "")
            if s is None or str(s).strip() == "":
                continue
            vals.append(float(s))

    if len(vals) == 0:
        raise ValueError(f"No numeric values read from {path} col={use_col}")

    return np.asarray(vals, dtype=np.float64), use_col

def write_fit_csv(path: str, obs_ms: np.ndarray, sim_ms: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = min(len(obs_ms), len(sim_ms))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_ms", "obs_s", "sim_ms", "sim_s"])
        for i in range(n):
            w.writerow([f"{obs_ms[i]:.6f}", f"{obs_ms[i]/1000.0:.6f}", f"{sim_ms[i]:.6f}", f"{sim_ms[i]/1000.0:.6f}"])

def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Stats
# -------------------------

def summarize_ms(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(x)),
        "std_ms": float(np.std(x, ddof=0)),
        "p90_ms": float(np.quantile(x, 0.90)),
        "p99_ms": float(np.quantile(x, 0.99)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }

def fmt_stats_sec(label: str, st: Dict[str, float]) -> str:
    return (
        f"{label}: mean={st['mean_ms']/1000:.3f}s std={st['std_ms']/1000:.3f}s "
        f"p90={st['p90_ms']/1000:.3f}s p99={st['p99_ms']/1000:.3f}s "
        f"min={st['min_ms']/1000:.3f}s max={st['max_ms']/1000:.3f}s"
    )


# -------------------------
# Models
# -------------------------

@dataclass
class DepositParams:
    base_mu: float
    base_sigma: float
    base_floor_ms: float
    base_cap_ms: float
    m_stalls: int
    stall_values_ms: List[float]
    ref_N: Optional[int] = None  # reference N for deterministic stalls

@dataclass
class WithdrawParams:
    N: int
    mu_b: float
    sigma_b: float
    top_q: float
    meas_sigma_ms: float
    clip_k: float
    floor_nonneg: bool
    cap_ms: float
    shift_ms: float
    force_min_outlier: bool
    force_max_once: bool


# -------------------------
# Simulation primitives
# -------------------------

def sample_lognormal_ms(rng: np.random.Generator, mu: float, sigma: float, size) -> np.ndarray:
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def simulate_deposit_ms(params: DepositParams, trials: int, rng: np.random.Generator) -> np.ndarray:
    base = sample_lognormal_ms(rng, params.base_mu, params.base_sigma, size=trials)
    base = np.clip(base, params.base_floor_ms, params.base_cap_ms)

    m = int(params.m_stalls)
    if m <= 0 or len(params.stall_values_ms) == 0:
        return base

    stalls_sorted = np.asarray(sorted(params.stall_values_ms), dtype=np.float64)
    m = max(0, min(m, len(stalls_sorted), trials))
    ref_N = params.ref_N if (params.ref_N is not None and params.ref_N > 0) else None

    if ref_N is None or trials == ref_N:
        # deterministic top-m overwrite
        idx = np.argpartition(base, trials - m)[trials - m:]
        idx_sorted = idx[np.argsort(base[idx])]
        base[idx_sorted] = stalls_sorted[-m:]
    else:
        # probabilistic scaling for trials != ref_N (rare stall rate scales ~1/ref_N)
        available = np.arange(trials, dtype=np.int64)
        rng.shuffle(available)
        pos = 0
        p_each = 1.0 / float(ref_N)
        for v in stalls_sorted[-m:]:
            c = int(rng.binomial(trials, p_each))
            if c <= 0:
                continue
            if pos + c > trials:
                c = trials - pos
            if c <= 0:
                break
            idx = available[pos:pos + c]
            pos += c
            base[idx] = v

    return base

def top_q_mean_rows(x: np.ndarray, top_q: float) -> np.ndarray:
    if not (0.0 < top_q <= 1.0):
        raise ValueError("top_q must be in (0,1].")
    trials, n = x.shape
    k = int(math.ceil(top_q * n))
    k = max(1, min(k, n))
    part = np.partition(x, n - k, axis=1)
    topk = part[:, n - k:]
    return np.mean(topk, axis=1)

def simulate_withdraw_residual_ms(params: WithdrawParams, trials: int, rng: np.random.Generator) -> Tuple[np.ndarray, float, int]:
    delays = sample_lognormal_ms(rng, params.mu_b, params.sigma_b, size=(trials, params.N))
    bcast = top_q_mean_rows(delays, params.top_q)

    noise = rng.normal(loc=0.0, scale=params.meas_sigma_ms, size=trials)
    if params.clip_k is not None and params.clip_k > 0:
        lim = params.clip_k * params.meas_sigma_ms
        noise = np.clip(noise, -lim, lim)

    x = bcast + noise + params.shift_ms

    # forced outliers (deterministic)
    if params.force_min_outlier and trials >= 1:
        x[0] = 0.0
    if params.force_max_once and trials >= 2:
        x[1] = params.cap_ms

    if params.floor_nonneg:
        x = np.maximum(x, 0.0)

    cap = float(params.cap_ms)
    if cap > 0:
        before = x.copy()
        x = np.minimum(x, cap)
        cap_hit_rate = float(np.mean(before >= cap))
    else:
        cap_hit_rate = 0.0

    neg_count = int(np.sum(x < 0.0))
    return x, cap_hit_rate, neg_count


# -------------------------
# Withdraw micro-adjust (optional)
# -------------------------

def withdraw_micro_adjust_one_shot(
    base: WithdrawParams,
    obs_res_ms: np.ndarray,
    trials: int,
    seed: int,
    *,
    target_p99_ms: float,
    bounded_clip_k=(1.2, 2.5),
    meas_sigma_grid=(0.85, 1.25, 0.05),
) -> Tuple[WithdrawParams, Dict[str, float]]:
    obs = np.asarray(obs_res_ms, dtype=np.float64)
    obs_st = summarize_ms(obs)
    obs_mean = float(obs_st["mean_ms"])

    best = base
    best_meta = {"score": float("inf")}

    clip_lo, clip_hi = bounded_clip_k
    clip_k = base.clip_k
    if not (clip_lo <= clip_k <= clip_hi):
        return base, {"score": float("inf"), "cap_hit_rate": float("nan"), "neg_count": float("nan")}

    for scale in np.arange(meas_sigma_grid[0], meas_sigma_grid[1] + 1e-9, meas_sigma_grid[2]):
        wp = WithdrawParams(**asdict(base))
        wp.meas_sigma_ms = float(base.meas_sigma_ms * scale)
        wp.clip_k = float(clip_k)

        rng = np.random.default_rng(seed)
        sim0, cap0, neg0 = simulate_withdraw_residual_ms(wp, trials=trials, rng=rng)
        wp.shift_ms = float(obs_mean - float(np.mean(sim0)))

        rng = np.random.default_rng(seed)
        sim, cap, neg = simulate_withdraw_residual_ms(wp, trials=trials, rng=rng)
        st = summarize_ms(sim)

        if cap > 0.05:
            continue

        score = (
            abs(st["p99_ms"] - target_p99_ms) / 1000.0 * 2.0 +
            abs(st["std_ms"] - obs_st["std_ms"]) / 1000.0 * 1.0 +
            (cap * 100.0) +
            (neg * 10.0)
        )

        if score < best_meta["score"]:
            best = wp
            best_meta = {"score": float(score), "cap_hit_rate": float(cap), "neg_count": float(neg)}

    return best, best_meta


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_deposit", required=True)
    ap.add_argument("--in_withdraw", required=True)
    ap.add_argument("--deposit_col", default=None)
    ap.add_argument("--withdraw_col", default=None)
    ap.add_argument("--out_dir", default="result")
    ap.add_argument("--dispute_ref_s", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--trials_dep_eval", type=int, default=6000)
    ap.add_argument("--trials_wd_eval", type=int, default=6000)

    ap.add_argument("--lock_deposit_json", default=None)
    ap.add_argument("--lock_withdraw_json", default=None)
    ap.add_argument("--enable_micro_adjust", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dep_ms, dep_col = read_latency_csv(args.in_deposit, args.deposit_col)
    wd_ms, wd_col = read_latency_csv(args.in_withdraw, args.withdraw_col)

    _info("Loaded:")
    _info(f" deposit_csv: {args.in_deposit}")
    _info(f" deposit_col: {dep_col} (ms; shown in seconds)")
    _info(f" withdraw_csv: {args.in_withdraw}")
    _info(f" withdraw_col: {wd_col} (ms; shown in seconds)")
    _info("")

    dispute_ref_ms = float(args.dispute_ref_s) * 1000.0
    obs_wd_res = wd_ms - dispute_ref_ms

    obs_dep_st = summarize_ms(dep_ms)
    obs_wd_st = summarize_ms(obs_wd_res)

    # ---- Load locked params (required for your workflow) ----
    if not args.lock_deposit_json:
        raise ValueError("This workflow expects --lock_deposit_json (deposit fixed).")
    if not args.lock_withdraw_json:
        raise ValueError("This workflow expects --lock_withdraw_json (withdraw fixed).")

    with open(args.lock_deposit_json, "r", encoding="utf-8") as f:
        jd = json.load(f)
    with open(args.lock_withdraw_json, "r", encoding="utf-8") as f:
        jw = json.load(f)

    if "deposit" not in jd:
        raise KeyError(f"'deposit' not found in {args.lock_deposit_json}")
    if "withdraw" not in jw:
        raise KeyError(f"'withdraw' not found in {args.lock_withdraw_json}")

    dp = DepositParams(**jd["deposit"])
    wp = WithdrawParams(**jw["withdraw"])

    # ---- Simulate deposit ----
    rng = np.random.default_rng(args.seed)
    sim_dep = simulate_deposit_ms(dp, trials=args.trials_dep_eval, rng=rng)
    sim_dep_st = summarize_ms(sim_dep)

    _info("==== CALIBRATED: DEPOSIT (Truncated lognormal base + deterministic/probabilistic stalls) ====")
    _info("OBS: " + fmt_stats_sec("", obs_dep_st).replace(": ", ": ").strip())
    _info("SIM: " + fmt_stats_sec("", sim_dep_st).replace(": ", ": ").strip())
    _info(
        f"Deposit model: base_mu={dp.base_mu:.6f} base_sigma={dp.base_sigma:.6f} "
        f"m_stalls={dp.m_stalls} base_floor_ms={dp.base_floor_ms:.3f} base_cap_ms={dp.base_cap_ms:.3f} "
        f"ref_N={dp.ref_N} stall_values_ms={[round(x,3) for x in dp.stall_values_ms]}"
    )
    _info("")

    # ---- Simulate withdraw residual ----
    rng = np.random.default_rng(args.seed)
    sim_wd_before, cap_rate_before, neg_before = simulate_withdraw_residual_ms(wp, trials=args.trials_wd_eval, rng=rng)
    st_before = summarize_ms(sim_wd_before)

    _info("==== CALIBRATED: WITHDRAW RESIDUAL (std-prioritized, BEFORE micro-adjust) ====")
    _info(f"dispute_ref: {args.dispute_ref_s:.1f} sec")
    _info("OBS : " + fmt_stats_sec("", obs_wd_st).replace(": ", ": ").strip())
    _info("SIM : " + fmt_stats_sec("", st_before).replace(": ", ": ").strip())
    _info(
        f"Withdraw residual model: N={wp.N} mu_b={wp.mu_b:.6f} sigma_b={wp.sigma_b:.6f} top_q={wp.top_q:.3f} "
        f"meas_sigma_ms={wp.meas_sigma_ms:.3f} clip_k={wp.clip_k:.2f} floor_nonneg={wp.floor_nonneg} "
        f"cap_ms={wp.cap_ms:.3f} shift_ms={wp.shift_ms:.3f} force_min_outlier={wp.force_min_outlier} "
        f"force_max_once={wp.force_max_once} score=nan"
    )
    _info(f"cap_hit_rate(before)={cap_rate_before:.6f} neg_count(before)={neg_before}")
    _info("")

    wp_final = wp
    sim_wd_final = sim_wd_before
    st_final = st_before
    cap_rate_final = cap_rate_before
    neg_final = neg_before

    # micro-adjust only if explicitly enabled AND you are not locking (but your workflow locks; still keep safe)
    if args.enable_micro_adjust:
        _info("==== WITHDRAW MICRO-ADJUST (ONE SHOT, bounded grid) ====")
        _info(
            f"before: clip_k={wp.clip_k:.2f} meas_sigma_ms={wp.meas_sigma_ms:.3f} shift_ms={wp.shift_ms:.3f} "
            f"cap_hit_rate={cap_rate_before:.6f} p99={st_before['p99_ms']/1000:.3f}s std={st_before['std_ms']/1000:.3f}s"
        )
        wp_final, meta = withdraw_micro_adjust_one_shot(
            base=wp,
            obs_res_ms=obs_wd_res,
            trials=min(len(obs_wd_res), args.trials_wd_eval),
            seed=args.seed,
            target_p99_ms=float(obs_wd_st["p99_ms"]),
        )
        rng = np.random.default_rng(args.seed)
        sim_wd_final, cap_rate_final, neg_final = simulate_withdraw_residual_ms(wp_final, trials=args.trials_wd_eval, rng=rng)
        st_final = summarize_ms(sim_wd_final)
        _info(
            f"after : clip_k={wp_final.clip_k:.2f} meas_sigma_ms={wp_final.meas_sigma_ms:.3f} shift_ms={wp_final.shift_ms:.3f} "
            f"cap_hit_rate={cap_rate_final:.6f} p99={st_final['p99_ms']/1000:.3f}s std={st_final['std_ms']/1000:.3f}s"
        )
        _info("")

    _info("==== CALIBRATED: WITHDRAW RESIDUAL (FINAL) ====")
    _info(f"dispute_ref: {args.dispute_ref_s:.1f} sec")
    _info("OBS : " + fmt_stats_sec("", obs_wd_st).replace(": ", ": ").strip())
    _info("SIM : " + fmt_stats_sec("", st_final).replace(": ", ": ").strip())
    _info(
        f"Withdraw residual model (final): N={wp_final.N} mu_b={wp_final.mu_b:.6f} sigma_b={wp_final.sigma_b:.6f} top_q={wp_final.top_q:.3f} "
        f"meas_sigma_ms={wp_final.meas_sigma_ms:.3f} clip_k={wp_final.clip_k:.2f} floor_nonneg={wp_final.floor_nonneg} "
        f"cap_ms={wp_final.cap_ms:.3f} shift_ms={wp_final.shift_ms:.3f} force_min_outlier={wp_final.force_min_outlier} "
        f"force_max_once={wp_final.force_max_once}"
    )
    _info(f"cap_hit_rate(final)={cap_rate_final:.6f} neg_count(final)={neg_final}")
    _info("")

    # ---- WRITE (OVERWRITE) ----
    out_dep_csv = os.path.join(args.out_dir, "validator_env_deposit_fit.csv")
    out_wd_csv  = os.path.join(args.out_dir, "validator_env_withdraw_res_fit.csv")
    out_json    = os.path.join(args.out_dir, "validator_env_best.json")

    write_fit_csv(out_dep_csv, dep_ms, sim_dep)
    write_fit_csv(out_wd_csv, obs_wd_res, sim_wd_final)

    write_json(out_json, {
        "meta": {
            "deposit_csv": args.in_deposit,
            "deposit_col": dep_col,
            "withdraw_csv": args.in_withdraw,
            "withdraw_col": wd_col,
            "dispute_ref_s": float(args.dispute_ref_s),
            "seed": int(args.seed),
        },
        "deposit": asdict(dp),
        "withdraw": asdict(wp_final),
    })

    _info("===== WRITE (OVERWRITE) =====")
    _info(f"Wrote: {out_dep_csv}")
    _info(f"Wrote: {out_wd_csv}")
    _info(f"Wrote: {out_json}")
    _info("")

    # ---- Final compare ----
    _info("============================================================")
    _info("FINAL COMPARE (structured, guarded, ms->sec display)")
    _info("============================================================\n")

    _info("=== DEPOSIT ===")
    _info("OBS : " + fmt_stats_sec("", obs_dep_st).replace(": ", ": ").strip())
    _info("SIM : " + fmt_stats_sec("", sim_dep_st).replace(": ", ": ").strip())
    _info(
        f"Δp99={(sim_dep_st['p99_ms']-obs_dep_st['p99_ms'])/1000:.3f}s "
        f"Δstd={(sim_dep_st['std_ms']-obs_dep_st['std_ms'])/1000:.3f}s "
        f"Δmax={(sim_dep_st['max_ms']-obs_dep_st['max_ms'])/1000:.3f}s"
    )
    _info("")

    _info("=== WITHDRAW RESIDUAL ===")
    _info("OBS : " + fmt_stats_sec("", obs_wd_st).replace(": ", ": ").strip())
    _info("SIM : " + fmt_stats_sec("", st_final).replace(": ", ": ").strip())
    _info(
        f"Δmean={(st_final['mean_ms']-obs_wd_st['mean_ms'])/1000:.3f}s "
        f"Δp99={(st_final['p99_ms']-obs_wd_st['p99_ms'])/1000:.3f}s "
        f"Δstd={(st_final['std_ms']-obs_wd_st['std_ms'])/1000:.3f}s "
        f"Δmax={(st_final['max_ms']-obs_wd_st['max_ms'])/1000:.3f}s"
    )
    _info(f"cap_hit_rate(final)={cap_rate_final:.6f} neg_count(final)={neg_final}")

if __name__ == "__main__":
    main()
