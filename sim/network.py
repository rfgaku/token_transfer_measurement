import simpy


class Network:
    """
    Event-driven broadcast network.
    Now supports Gumbel distribution for Fujiwara theory.
    """

    def __init__(self, env: simpy.Environment, rng, latency_config: dict, stop_event=None):
        self.env = env
        self.rng = rng
        self.lat_cfg = latency_config
        self.stop_event = stop_event

    def sample_latency_s(self):
        # In v2.0 (Logic-Based), network latency is just the physical P2P delay.
        # Major latencies (Sequencing, Consensus) are handled by their respective logic classes.
        
        base = float(self.lat_cfg.get("p2p_base_s", 0.05))
        jitter = float(self.lat_cfg.get("p2p_jitter_s", 0.01))
        
        # Simple normal distribution for wire latency
        val = self.rng.normalvariate(base, jitter)
        return max(0.001, val)

    def send_unicast(self, src, dst, deliver_fn, payload):
        # v2.0 Network Partition Logic
        partitions = self.lat_cfg.get("partitions", [])
        if partitions:
            # Check if src and dst are in the same partition
            # partitions = [[0, 1, 2], [3, 4, 5]]
            src_part = None
            dst_part = None
            for idx, part in enumerate(partitions):
                if src in part: src_part = idx
                if dst in part: dst_part = idx
            
            # If partitions are defined, nodes must be in the SAME partition to communicate.
            # Nodes not in any partition? Assume they are isolated or in a default group?
            # Assumption: All nodes are assigned to a partition if partitions are active.
            # If different partitions, drop packet (return immediately without delivery)
            if src_part != dst_part:
                # Packet Drop (Infinite Latency)
                # print(f"[DEBUG] DROP src={src}(P{src_part}) dst={dst}(P{dst_part})")
                return
            # else:
            #    print(f"[DEBUG] PASS src={src}(P{src_part}) dst={dst}(P{dst_part})")

        def _proc():
            delay = self.sample_latency_s()
            evt = self.env.timeout(delay)

            if self.stop_event is not None:
                yield evt | self.stop_event
                if self.stop_event.triggered:
                    return
            else:
                yield evt

            deliver_fn(dst, payload)

        self.env.process(_proc())
