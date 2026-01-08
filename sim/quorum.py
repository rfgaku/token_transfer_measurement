import simpy
import math


class QuorumCollector:
    """
    Quorum + reach metrics collector.

    - quorum is reached when q HONEST voters are observed.
    - reach metrics are computed from first-seen PROP timestamps among HONEST nodes.
    """

    def __init__(self, env: simpy.Environment, q: int, N: int, timeout_s: float, stop_event: simpy.Event = None):
        self.env = env
        self.q = int(q)
        self.N = int(N)
        self.timeout_s = float(timeout_s)

        self.quorum_event = env.event()
        self.stop_event = stop_event if stop_event is not None else env.event()

        self._seen_votes = set()      # (msg_id, voter_id)
        self._count = {}              # msg_id -> count

        self._honest = None           # set[int] | None
        self._prop_seen_t = {}        # msg_id -> {node_id: t_rel}
        self._prop_seen_h = {}        # msg_id -> {node_id: hops}
        self._t0 = {}                 # msg_id -> t0_rel (float)

        self.t_quorum = None
        self.t0 = None

        # outputs (always set by finalize_reach)
        self.reach_at_quorum = 0.0
        self.reach_at_end = 0.0
        self.reach_t_p99 = 0.0
        self.reach_hops_mean = 0.0

    def set_honest_nodes(self, honest_ids):
        self._honest = set(int(x) for x in honest_ids)

    def note_proposal_seen(self, node_id: int, msg_id: int, t_now: float, hops: int, t0: float):
        mid = int(msg_id)
        nid = int(node_id)
        t0f = float(t0)

        if mid not in self._t0:
            self._t0[mid] = t0f

        # store relative times
        rel = float(t_now) - t0f

        m_t = self._prop_seen_t.get(mid)
        if m_t is None:
            m_t = {}
            self._prop_seen_t[mid] = m_t

        if nid not in m_t:
            m_t[nid] = rel
            m_h = self._prop_seen_h.get(mid)
            if m_h is None:
                m_h = {}
                self._prop_seen_h[mid] = m_h
            m_h[nid] = int(hops)

    def on_vote(self, voter_id: int, msg_id: int, t_now: float, t0: float):
        vid = int(voter_id)
        mid = int(msg_id)

        if self._honest is not None and vid not in self._honest:
            return

        key = (mid, vid)
        if key in self._seen_votes:
            return
        self._seen_votes.add(key)

        c = self._count.get(mid, 0) + 1
        self._count[mid] = c

        if c >= self.q and (not self.quorum_event.triggered):
            self.t_quorum = float(t_now)
            self.t0 = float(t0)
            self.quorum_event.succeed(True)
            if (self.stop_event is not None) and (not self.stop_event.triggered):
                self.stop_event.succeed(True)

    @staticmethod
    def _pctl(xs, p: float):
        if not xs:
            return None
        ys = sorted(xs)
        k = int(math.ceil(p * (len(ys) - 1)))
        return ys[k]

    def finalize_reach(self, msg_id: int, t_end_rel: float, t_quorum_rel=None):
        mid = int(msg_id)
        honest = self._honest
        if honest is None:
            # fallback: treat all observed nodes as "honest universe"
            honest = set(self._prop_seen_t.get(mid, {}).keys())

        honest_total = max(0, len(honest))

        seen_t = self._prop_seen_t.get(mid, {})
        seen_h = self._prop_seen_h.get(mid, {})

        times = [float(seen_t[n]) for n in honest if n in seen_t]
        hops = [int(seen_h[n]) for n in honest if n in seen_h]

        if honest_total == 0:
            self.reach_at_end = 0.0
        else:
            self.reach_at_end = float(len(times) / honest_total)

        if t_quorum_rel is None:
            self.reach_at_quorum = self.reach_at_end
        else:
            tq = float(t_quorum_rel)
            cnt = sum(1 for t in times if t <= tq)
            self.reach_at_quorum = float(cnt / honest_total) if honest_total > 0 else 0.0

        p99 = self._pctl(times, 0.99)
        self.reach_t_p99 = float(p99) if p99 is not None else float(t_end_rel)

        if hops:
            self.reach_hops_mean = float(sum(hops) / len(hops))
        else:
            self.reach_hops_mean = 0.0


QuorumChecker = QuorumCollector
