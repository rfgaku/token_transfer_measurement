import math
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Summary:
    feasible: bool
    fail_rate: float
    t_quorum_mean: float
    t_quorum_p99: float
    t_total_p99: float
    reach_at_quorum_mean: float
    reach_at_end_mean: float
    reach_t_p99_mean: float
    reach_hops_mean_mean: float

def _p99(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(math.ceil(0.99 * len(xs))) - 1))
    return xs[idx]

def summarize(outs: List[Dict[str, Any]]) -> Summary:
    feasible = all(o.get("feasible", True) for o in outs)
    fail_rate = statistics.mean([1.0 if o.get("fail", False) else 0.0 for o in outs]) if outs else float("nan")

    t_quorum = [o.get("t_quorum", float("nan")) for o in outs]
    t_total = [o.get("t_total", float("nan")) for o in outs]

    def _mean_key(k: str) -> float:
        xs = [o.get(k, float("nan")) for o in outs]
        xs = [x for x in xs if x == x]
        return statistics.mean(xs) if xs else float("nan")

    return Summary(
        feasible=feasible,
        fail_rate=fail_rate,
        t_quorum_mean=_mean_key("t_quorum"),
        t_quorum_p99=_p99(t_quorum),
        t_total_p99=_p99(t_total),
        reach_at_quorum_mean=_mean_key("reach_at_quorum"),
        reach_at_end_mean=_mean_key("reach_at_end"),
        reach_t_p99_mean=_mean_key("reach_t_p99"),
        reach_hops_mean_mean=_mean_key("reach_hops_mean"),
    )
