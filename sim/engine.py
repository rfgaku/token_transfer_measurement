import json, math
import numpy as np
from .broadcast import simulate_broadcast

DEFAULTS = {
    "scenario": "withdraw",
    "timeouts": {"quorum_timeout_s": 30.0, "run_timeout_s": 35.0},
    "gossip": {"fanout": 3, "max_hops": 6, "forward_probability": 1.0, "max_events": 120000},
    "latency": {"model": "lognorm", "base_ms": 40.0, "sigma": 0.4, "scale_ms": 40.0, "clip_ms": [1.0, 500.0]},
    "deposit": {"fixed_base_s": 0.0},
    "withdraw": {"fixed_base_s": 200.3},
    "sweep": {
        "trials": 300,
        "byz_fracs": [0.0, 0.1, 0.2, 0.3],
        # default grid chosen to match typical "N=21/24/30 with q sets" output
        "cells": [
            {"N": 21, "q_list": [14, 16, 17]},
            {"N": 24, "q_list": [16, 18, 20]},
            {"N": 30, "q_list": [20, 23, 24]},
        ],
    }
}

def _deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    cfg = json.loads(json.dumps(DEFAULTS))
    _deep_merge(cfg, user)

    # Backward compat defaults
    cfg.setdefault("gossip", {})
    cfg["gossip"].setdefault("fanout", DEFAULTS["gossip"]["fanout"])
    cfg["gossip"].setdefault("max_hops", DEFAULTS["gossip"]["max_hops"])
    cfg["gossip"].setdefault("forward_probability", DEFAULTS["gossip"]["forward_probability"])
    cfg["gossip"].setdefault("max_events", DEFAULTS["gossip"]["max_events"])

    cfg.setdefault("timeouts", {})
    cfg["timeouts"].setdefault("quorum_timeout_s", DEFAULTS["timeouts"]["quorum_timeout_s"])
    cfg["timeouts"].setdefault("run_timeout_s", max(cfg["timeouts"]["quorum_timeout_s"] + 2.0, DEFAULTS["timeouts"]["run_timeout_s"]))

    cfg.setdefault("sweep", {})
    cfg["sweep"].setdefault("trials", DEFAULTS["sweep"]["trials"])
    cfg["sweep"].setdefault("byz_fracs", DEFAULTS["sweep"]["byz_fracs"])

    # Support older key names: grid -> cells
    if "cells" not in cfg["sweep"] and "grid" in cfg["sweep"]:
        cells = []
        for g in cfg["sweep"]["grid"]:
            N = int(g["N"])
            if "q" in g:
                q_list = [int(g["q"])]
            else:
                qr = float(g.get("quorum_ratio", 2/3))
                q_list = [int(math.ceil(qr * N))]
            cells.append({"N": N, "q_list": q_list})
        cfg["sweep"]["cells"] = cells

    cfg["sweep"].setdefault("cells", DEFAULTS["sweep"]["cells"])
    return cfg

def one_run(cfg, N, q, byz_frac, seed, verbose=False):
    scenario = cfg.get("scenario","withdraw")
    fixed_base_s = float(cfg.get(scenario, {}).get("fixed_base_s", 0.0))
    out = simulate_broadcast(cfg, N=N, q=q, byz_frac=byz_frac, seed=seed, verbose=verbose)
    out["t_total"] = float(fixed_base_s + out["t_quorum"])
    return out

def _q(p, xs):
    if len(xs) == 0:
        return float("nan")
    return float(np.quantile(xs, p, method="linear"))

def sweep(cfg, trials=300, seed=1, progress_every=50):
    sweep_cfg = cfg.get("sweep", {})
    cells = sweep_cfg.get("cells", [])
    byz_fracs = list(sweep_cfg.get("byz_fracs", [0.0,0.1,0.2,0.3]))

    scenario = cfg.get("scenario","withdraw")
    fixed_base_s = float(cfg.get(scenario, {}).get("fixed_base_s", 0.0))

    rows = []
    cell_total = sum(len(c.get("q_list", [])) * len(byz_fracs) for c in cells) or 1
    cell_idx = 0

    for c in cells:
        N = int(c["N"])
        q_list = [int(x) for x in c.get("q_list", [])]
        for q in q_list:
            quorum_ratio = q / N if N else float("nan")
            for bf in byz_fracs:
                cell_idx += 1
                outs = []
                fails = 0
                byz_count = int(round(float(bf) * N))
                feasible = (q <= (N - byz_count))

                for t in range(trials):
                    out = simulate_broadcast(cfg, N=N, q=q, byz_frac=float(bf),
                                             seed=seed + 100000*cell_idx + t,
                                             verbose=False)
                    outs.append(out)
                    fails += int(out["fail"] > 0.5)
                    if progress_every and (t+1) % progress_every == 0:
                        print(f"[progress] cell={cell_idx}/{cell_total} N={N} q={q} byz={bf} trials={t+1}/{trials}", flush=True)

                t_quorums = [o["t_quorum"] for o in outs]
                t_totals = [fixed_base_s + o["t_quorum"] for o in outs]
                reach_q = [o["reach_at_quorum"] for o in outs]
                reach_end = [o["reach_at_end"] for o in outs]
                reach_t_p99 = [o["reach_t_p99"] for o in outs]
                hops = [o["reach_hops_mean"] for o in outs]

                rows.append({
                    "N": N,
                    "quorum_ratio": quorum_ratio,
                    "q": q,
                    "byz_frac": float(bf),
                    "feasible": bool(feasible),
                    "fail_rate": fails / trials,
                    "t_quorum_mean": float(np.mean(t_quorums)),
                    "t_quorum_p99": _q(0.99, t_quorums),
                    "t_total_p99": _q(0.99, t_totals),
                    "reach_at_quorum_mean": float(np.mean(reach_q)),
                    "reach_at_end_mean": float(np.mean(reach_end)),
                    "reach_t_p99_mean": float(np.mean(reach_t_p99)),
                    "reach_hops_mean_mean": float(np.mean(hops)),
                })

    header = ["N","quorum_ratio","q","byz_frac","feasible","fail_rate",
              "t_quorum_mean","t_quorum_p99","t_total_p99",
              "reach_at_quorum_mean","reach_at_end_mean","reach_t_p99_mean","reach_hops_mean_mean"]
    return {"header": header, "rows": rows}
