import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class BroadcastResult:
    feasible: bool
    fail: bool
    t_quorum: float
    t_end: float
    reach_at_quorum: float
    reach_at_end: float
    reach_t_p99: float
    reach_hops_mean: float

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def make_latency_sampler(cfg: dict):
    """
    Returns a function rng -> latency_seconds
    Backward compatible: if missing -> fixed 0.08s
    """
    lat = (cfg or {}).get("latency", {}) if isinstance(cfg, dict) else {}
    dist = str(lat.get("dist", "fixed")).lower()
    min_s = float(lat.get("min_s", 0.0))
    max_s = float(lat.get("max_s", 1e9))

    if dist == "gumbel":
        mu = float(lat.get("mu", 0.08))
        beta = float(lat.get("beta", 0.02))
        def sample(rng: random.Random) -> float:
            u = _clamp(rng.random(), 1e-12, 1 - 1e-12)
            x = mu - beta * math.log(-math.log(u))
            return _clamp(x, min_s, max_s)
        return sample

    if dist in ("lognorm", "lognormal"):
        mu = float(lat.get("mu", math.log(0.08)))
        sigma = float(lat.get("sigma", 0.4))
        def sample(rng: random.Random) -> float:
            x = rng.lognormvariate(mu, sigma)
            return _clamp(x, min_s, max_s)
        return sample

    if dist == "fixed":
        v = float(lat.get("value", 0.08))
        def sample(_rng: random.Random) -> float:
            return _clamp(v, min_s, max_s)
        return sample

    def sample(_rng: random.Random) -> float:
        return 0.08
    return sample

def simulate_broadcast(
    rng: random.Random,
    N: int,
    q: int,
    byz_frac: float,
    cfg: dict,
    *,
    progress: bool = False,
) -> BroadcastResult:
    """
    Gossip-style broadcast (no simpy; stable).

    Key properties:
      - quorum is counted over HONEST nodes only (byz do not sign)
      - reach_at_quorum uses earliest scheduled delivery time (tie-safe)
      - TARGET SELECTION FIX: prefer not-seen nodes to avoid wasted fanout
    """
    if N <= 0:
        raise ValueError("N must be positive")
    q = int(q)
    if q <= 0:
        raise ValueError("q must be positive")

    gossip_cfg = (cfg or {}).get("gossip", {}) if isinstance(cfg, dict) else {}

    # stronger defaults (but config overrides)
    fanout = int(gossip_cfg.get("fanout", 3))
    init_fanout = int(gossip_cfg.get("initial_fanout", max(4, fanout)))
    max_hops = int(gossip_cfg.get("max_hops", 10))
    byz_forward_prob = float(gossip_cfg.get("byz_forward_prob", 0.0))

    commit_window = float((cfg or {}).get("commit_window_s", 30.0))
    max_events = int(gossip_cfg.get("max_events", 400000))

    fanout = max(1, min(fanout, max(1, N - 1)))
    init_fanout = max(1, min(init_fanout, max(1, N - 1)))
    max_hops = max(1, max_hops)
    commit_window = max(0.001, commit_window)

    latency = make_latency_sampler(cfg)

    # byz set
    byz_count = int(round(_clamp(byz_frac, 0.0, 1.0) * N))
    all_ids = list(range(N))
    byz_ids = set(rng.sample(all_ids, byz_count)) if byz_count > 0 else set()
    honest_ids = [i for i in all_ids if i not in byz_ids]

    # quorum on honest nodes
    feasible = (q <= len(honest_ids))

    import heapq
    seq = 0
    pq: List[Tuple[float, int, int, int, int]] = []

    # tie-safe reach: earliest scheduled delivery time (even if not yet popped)
    earliest_sched: Dict[int, float] = {0: 0.0}

    recv_time: Dict[int, float] = {0: 0.0}
    recv_hop: Dict[int, int] = {0: 0}
    seen = set([0])

    honest_seen = 1 if 0 in honest_ids else 0
    quorum_time: Optional[float] = 0.0 if q <= honest_seen else None

    # to reduce wasted messages: track "already scheduled" per dst (approx)
    scheduled = set([0])

    def schedule(src: int, dst: int, now: float, hop: int):
        nonlocal seq
        if src == dst:
            return
        t = now + latency(rng)
        if t > commit_window:
            return
        prev = earliest_sched.get(dst)
        if prev is None or t < prev:
            earliest_sched[dst] = t
        seq += 1
        heapq.heappush(pq, (t, seq, src, dst, hop))
        scheduled.add(dst)

    def pick_targets(src: int, k: int) -> List[int]:
        """
        Prefer nodes not yet seen (and not yet scheduled if possible),
        then fall back to any other nodes except self.
        """
        if N <= 1:
            return []
        # strongest preference: not seen & not scheduled
        cand1 = [i for i in all_ids if i != src and (i not in seen) and (i not in scheduled)]
        if cand1:
            return rng.sample(cand1, min(k, len(cand1)))

        # next: not seen
        cand2 = [i for i in all_ids if i != src and (i not in seen)]
        if cand2:
            return rng.sample(cand2, min(k, len(cand2)))

        # fallback: anyone but self
        cand3 = [i for i in all_ids if i != src]
        return rng.sample(cand3, min(k, len(cand3)))

    # initial push from leader(0)
    for dst in pick_targets(0, init_fanout):
        schedule(0, dst, 0.0, 1)

    events = 0
    if progress:
        print(f"[bcast] start N={N} q={q} byz={byz_count} honest={len(honest_ids)} fanout={fanout} init={init_fanout} max_hops={max_hops} window={commit_window}")

    while pq and events < max_events:
        t, _s, src, dst, hop = heapq.heappop(pq)
        events += 1

        if t > commit_window:
            break

        if dst in seen:
            continue

        seen.add(dst)
        recv_time[dst] = t
        recv_hop[dst] = hop

        if dst in honest_ids:
            honest_seen += 1
            if quorum_time is None and honest_seen >= q:
                quorum_time = t

        # forward?
        can_forward = (dst in honest_ids) or (rng.random() < byz_forward_prob)
        if can_forward and hop < max_hops:
            for nd in pick_targets(dst, fanout):
                schedule(dst, nd, t, hop + 1)

        if progress and (events % 5000 == 0):
            qt = quorum_time if quorum_time is not None else float("nan")
            print(f"[bcast] t={t:.3f} seen={len(seen)}/{N} honest_seen={honest_seen}/{len(honest_ids)} qtime={qt}")

    # quorum fail includes infeasible case
    fail = (quorum_time is None) or (not feasible)

    t_quorum = commit_window if quorum_time is None else float(quorum_time)
    t_end = commit_window

    # reach stats use earliest_sched (tie-safe)
    reach_at_end = sum(1 for tt in earliest_sched.values() if tt <= t_end) / N
    reach_at_quorum = sum(1 for tt in earliest_sched.values() if tt <= t_quorum) / N

    reached_times = [tt for tt in earliest_sched.values() if tt <= t_end]
    if len(reached_times) >= 2:
        reached_times.sort()
        idx = max(0, min(len(reached_times) - 1, int(math.ceil(0.99 * len(reached_times))) - 1))
        reach_t_p99 = reached_times[idx]
    elif len(reached_times) == 1:
        reach_t_p99 = reached_times[0]
    else:
        reach_t_p99 = float("nan")

    hop_vals = [h for nid, h in recv_hop.items() if nid != 0 and recv_time.get(nid, 1e18) <= t_end]
    reach_hops_mean = statistics.mean(hop_vals) if hop_vals else float("nan")

    return BroadcastResult(
        feasible=feasible,
        fail=fail,
        t_quorum=t_quorum,
        t_end=t_end,
        reach_at_quorum=reach_at_quorum,
        reach_at_end=reach_at_end,
        reach_t_p99=reach_t_p99,
        reach_hops_mean=reach_hops_mean,
    )
