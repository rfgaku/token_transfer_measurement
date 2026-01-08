import random
from dataclasses import dataclass
from typing import Dict

from .broadcast import simulate_broadcast

@dataclass
class ScenarioResult:
    feasible: bool
    fail: bool
    t_quorum: float
    t_total: float
    reach_at_quorum: float
    reach_at_end: float
    reach_t_p99: float
    reach_hops_mean: float
    parts: Dict[str, float]

def _get_scenario_cfg(cfg: dict, scenario: str) -> dict:
    if not isinstance(cfg, dict):
        return {}
    v = cfg.get(scenario, {})
    return v if isinstance(v, dict) else {}

def run_scenario(
    rng: random.Random,
    cfg: dict,
    scenario: str,
    N: int,
    q: int,
    byz_frac: float,
    *,
    progress: bool = False,
) -> ScenarioResult:
    """
    deposit / withdraw structures:

      deposit:
        base_s default 0.0
        stages default 2 ("obs_bcast","sig_bcast")

      withdraw:
        base_s default 200.0  (IMPORTANT: matches HL withdraw long window)
        stages default 3 ("sig_bcast","req_tx_bcast","fin_tx_bcast")

    config override:
      cfg[scenario]["base_s"], cfg[scenario]["stages"], cfg[scenario]["stage_names"]
    """
    scfg = _get_scenario_cfg(cfg, scenario)

    if scenario == "withdraw":
        base_default = 200.0
        stages_default = 3
    elif scenario == "deposit":
        base_default = 0.0
        stages_default = 2
    else:
        base_default = 0.0
        stages_default = 1

    base_s = float(scfg.get("base_s", base_default))
    stages = int(scfg.get("stages", stages_default))

    stage_names = scfg.get("stage_names")
    if not isinstance(stage_names, list) or len(stage_names) != stages:
        if scenario == "withdraw" and stages == 3:
            stage_names = ["sig_bcast", "req_tx_bcast", "fin_tx_bcast"]
        elif scenario == "deposit" and stages == 2:
            stage_names = ["obs_bcast", "sig_bcast"]
        else:
            stage_names = [f"bcast_{i+1}" for i in range(stages)]

    parts: Dict[str, float] = {}
    t_quorum_sum = 0.0
    feasible_all = True
    fail_any = False

    raq, rae, rt99, rh = [], [], [], []

    for i, name in enumerate(stage_names):
        br = simulate_broadcast(rng, N, q, byz_frac, cfg, progress=(progress and i == 0))
        feasible_all = feasible_all and br.feasible
        fail_any = fail_any or br.fail

        t_quorum_sum += br.t_quorum
        parts[name] = br.t_quorum

        raq.append(br.reach_at_quorum)
        rae.append(br.reach_at_end)
        rt99.append(br.reach_t_p99)
        rh.append(br.reach_hops_mean)

    def _mean(xs):
        xs2 = [x for x in xs if x == x]
        return sum(xs2) / len(xs2) if xs2 else float("nan")

    t_total = base_s + t_quorum_sum
    parts["base_s"] = base_s
    parts["t_quorum_sum"] = t_quorum_sum

    return ScenarioResult(
        feasible=feasible_all,
        fail=fail_any or (not feasible_all),
        t_quorum=t_quorum_sum,
        t_total=t_total,
        reach_at_quorum=_mean(raq),
        reach_at_end=_mean(rae),
        reach_t_p99=_mean(rt99),
        reach_hops_mean=_mean(rh),
        parts=parts,
    )
