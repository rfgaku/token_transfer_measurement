"""
NetworkPhysics: 地理的レイテンシを考慮した物理ネットワークシミュレーション

Phase 2 Digital Twin Simulator の核心コンポーネント。
NetworkXを用いてノード間の接続グラフを構築し、
遅延は「エッジの重み（地理的レイテンシ）+ 動的Jitter」で算出する。

設計決定事項:
- N=21 バリデータ（Hyperliquid Mainnet準拠）
- リージョン: US (7), EU (7), APAC (7)
- 同一リージョン内: 5-20ms
- リージョン間: 100-200ms
"""

import simpy
import random
import math
from typing import Dict, List, Tuple, Callable, Optional
from enum import Enum

# NetworkXがインストールされていない場合のフォールバック
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[WARNING] NetworkX not installed. Using fallback implementation.")


class Region(Enum):
    """
    バリデータの地理的リージョン
    """
    US = "US"       # 北米
    EU = "EU"       # ヨーロッパ
    APAC = "APAC"   # アジア太平洋


# リージョン間レイテンシ行列（ミリ秒）
# [src_region][dst_region] = (min_ms, max_ms)
INTER_REGION_LATENCY_MS = {
    Region.US: {
        Region.US: (5, 20),      # 同一リージョン
        Region.EU: (80, 120),    # US <-> EU
        Region.APAC: (150, 200), # US <-> APAC
    },
    Region.EU: {
        Region.US: (80, 120),
        Region.EU: (5, 20),      # 同一リージョン
        Region.APAC: (100, 150), # EU <-> APAC
    },
    Region.APAC: {
        Region.US: (150, 200),
        Region.EU: (100, 150),
        Region.APAC: (5, 20),    # 同一リージョン
    },
}


class NetworkPhysics:
    """
    物理ネットワークシミュレーション

    - NetworkXグラフによるトポロジー管理
    - 地理的リージョンに基づくエッジ重み
    - SimPyプロセスによるメッセージ伝送
    - ログ出力による物理挙動の証明
    """

    def __init__(
        self,
        env: simpy.Environment,
        N: int = 21,
        rng: Optional[random.Random] = None,
        enable_logging: bool = True,
        jitter_ratio: float = 0.1,
    ):
        """
        初期化

        Args:
            env: SimPy環境
            N: ノード数（デフォルト21 = Hyperliquid Mainnet）
            rng: 乱数生成器
            enable_logging: ログ出力の有効化
            jitter_ratio: 動的Jitterの比率（ベース遅延に対する割合）
        """
        self.env = env
        self.N = N
        self.rng = rng if rng else random.Random()
        self.enable_logging = enable_logging
        self.jitter_ratio = jitter_ratio

        # ノードのリージョン割り当て
        self.node_regions: Dict[int, Region] = {}
        self._assign_regions()

        # グラフ構築
        if HAS_NETWORKX:
            self.graph = nx.complete_graph(N)
            self._initialize_edge_weights()
        else:
            # フォールバック: 辞書ベースの遅延管理
            self.edge_latencies: Dict[Tuple[int, int], float] = {}
            self._initialize_edge_weights_fallback()

        # 通信ログ
        self.message_log: List[Dict] = []

        if self.enable_logging:
            print(f"[NetworkPhysics] 初期化完了: N={N}, Regions={self._get_region_summary()}")

    def _assign_regions(self):
        """
        ノードにリージョンを割り当てる（均等分散）

        N=21の場合: US=7, EU=7, APAC=7
        """
        regions = list(Region)
        nodes_per_region = self.N // len(regions)
        remainder = self.N % len(regions)

        node_id = 0
        for i, region in enumerate(regions):
            # 余りを最初のリージョンに分配
            count = nodes_per_region + (1 if i < remainder else 0)
            for _ in range(count):
                if node_id < self.N:
                    self.node_regions[node_id] = region
                    node_id += 1

    def _get_region_summary(self) -> str:
        """リージョン分布のサマリーを返す"""
        counts = {r: 0 for r in Region}
        for r in self.node_regions.values():
            counts[r] += 1
        return ", ".join(f"{r.value}={c}" for r, c in counts.items())

    def _sample_base_latency(self, src_region: Region, dst_region: Region) -> float:
        """
        地理的特性に基づいたベースレイテンシをサンプリング（秒単位）
        """
        min_ms, max_ms = INTER_REGION_LATENCY_MS[src_region][dst_region]
        latency_ms = self.rng.uniform(min_ms, max_ms)
        return latency_ms / 1000.0  # ミリ秒 -> 秒

    def _initialize_edge_weights(self):
        """
        NetworkXグラフの各エッジに地理的レイテンシを設定
        """
        for (u, v) in self.graph.edges():
            src_region = self.node_regions[u]
            dst_region = self.node_regions[v]
            base_latency = self._sample_base_latency(src_region, dst_region)
            self.graph[u][v]["base_latency"] = base_latency
            self.graph[u][v]["src_region"] = src_region
            self.graph[u][v]["dst_region"] = dst_region

    def _initialize_edge_weights_fallback(self):
        """
        フォールバック実装: NetworkX不使用時のエッジ重み初期化
        """
        for u in range(self.N):
            for v in range(u + 1, self.N):
                src_region = self.node_regions[u]
                dst_region = self.node_regions[v]
                base_latency = self._sample_base_latency(src_region, dst_region)
                self.edge_latencies[(u, v)] = base_latency
                self.edge_latencies[(v, u)] = base_latency

    def get_base_latency(self, src: int, dst: int) -> float:
        """
        src→dst間のベースレイテンシを返す（秒）
        """
        if src == dst:
            return 0.0

        if HAS_NETWORKX:
            return self.graph[src][dst]["base_latency"]
        else:
            return self.edge_latencies.get((src, dst), 0.05)

    def get_latency(self, src: int, dst: int) -> float:
        """
        src→dst間の物理遅延を返す（ベースレイテンシ + 動的Jitter）
        """
        if src == dst:
            return 0.0

        base = self.get_base_latency(src, dst)
        # 動的Jitter: ベース遅延の ±jitter_ratio%
        jitter = self.rng.gauss(0, base * self.jitter_ratio)
        total = base + jitter
        return max(0.001, total)

    def get_node_region(self, node_id: int) -> Region:
        """ノードのリージョンを返す"""
        return self.node_regions.get(node_id, Region.US)

    def send_message(
        self,
        src: int,
        dst: int,
        payload: dict,
        callback: Callable[[int, dict], None],
        message_type: str = "GENERIC",
    ):
        """
        メッセージ送信をSimPyプロセスとして実行

        Args:
            src: 送信元ノードID
            dst: 送信先ノードID
            payload: メッセージペイロード
            callback: 到着時に呼び出すコールバック関数
            message_type: ログ用のメッセージタイプ
        """

        def _transmission_process():
            delay = self.get_latency(src, dst)
            send_time = self.env.now

            if self.enable_logging:
                src_region = self.get_node_region(src).value
                dst_region = self.get_node_region(dst).value
                print(
                    f"[Time: {send_time:.3f}s] Node {src} ({src_region}) -> "
                    f"Node {dst} ({dst_region}): {message_type} sent "
                    f"(Expected Latency: {delay*1000:.1f}ms)"
                )

            yield self.env.timeout(delay)

            arrival_time = self.env.now
            actual_latency = arrival_time - send_time

            if self.enable_logging:
                print(
                    f"[Time: {arrival_time:.3f}s] Node {dst} received {message_type} "
                    f"from Node {src} (Actual Latency: {actual_latency*1000:.1f}ms)"
                )

            # ログ記録
            self.message_log.append({
                "send_time": send_time,
                "arrival_time": arrival_time,
                "src": src,
                "dst": dst,
                "src_region": self.get_node_region(src).value,
                "dst_region": self.get_node_region(dst).value,
                "latency": actual_latency,
                "message_type": message_type,
            })

            # コールバック実行
            callback(dst, payload)

        return self.env.process(_transmission_process())

    def send_rpc_request(
        self,
        requester_id: int,
        rpc_endpoint_id: int,
        request_payload: dict,
    ):
        """
        RPCリクエスト（往復通信）を実行するジェネレータ

        Args:
            requester_id: リクエスト元ノードID
            rpc_endpoint_id: RPCエンドポイントのノードID
            request_payload: リクエストペイロード

        Yields:
            往復通信の完了を待機

        Returns:
            (往路遅延, 復路遅延, 合計遅延)
        """
        start_time = self.env.now

        # 往路遅延
        outbound_delay = self.get_latency(requester_id, rpc_endpoint_id)
        if self.enable_logging:
            print(
                f"[Time: {start_time:.3f}s] Node {requester_id} sent RPC request -> "
                f"Endpoint {rpc_endpoint_id} (Outbound: {outbound_delay*1000:.1f}ms)"
            )
        yield self.env.timeout(outbound_delay)

        # RPC処理時間（エンドポイント側）
        processing_time = self.rng.uniform(0.005, 0.015)  # 5-15ms
        yield self.env.timeout(processing_time)

        # 復路遅延
        inbound_delay = self.get_latency(rpc_endpoint_id, requester_id)
        yield self.env.timeout(inbound_delay)

        end_time = self.env.now
        total_latency = end_time - start_time

        if self.enable_logging:
            print(
                f"[Time: {end_time:.3f}s] Node {requester_id} received RPC response "
                f"(Total RTT: {total_latency*1000:.1f}ms)"
            )

        return (outbound_delay, inbound_delay, total_latency)

    def broadcast_gossip(
        self,
        sender_id: int,
        payload: dict,
        callback: Callable[[int, dict], None],
        fanout: int = 3,
        message_type: str = "GOSSIP",
    ):
        """
        Gossipプロトコルによるブロードキャスト

        Args:
            sender_id: 送信元ノードID
            payload: ブロードキャストペイロード
            callback: 各ノード到着時のコールバック
            fanout: 各ノードが転送する先のノード数
            message_type: ログ用のメッセージタイプ
        """
        # 全ノードへ直接送信（簡易版）
        # TODO: 本格的なGossipプロトコル実装（段階的伝播）

        for dst in range(self.N):
            if dst != sender_id:
                self.send_message(sender_id, dst, payload, callback, message_type)

    def get_statistics(self) -> Dict:
        """
        通信統計を返す
        """
        if not self.message_log:
            return {"total_messages": 0}

        latencies = [log["latency"] for log in self.message_log]
        intra_region = [
            log["latency"] for log in self.message_log
            if log["src_region"] == log["dst_region"]
        ]
        inter_region = [
            log["latency"] for log in self.message_log
            if log["src_region"] != log["dst_region"]
        ]

        def safe_stats(data):
            if not data:
                return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
            import statistics
            return {
                "count": len(data),
                "mean": statistics.mean(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data),
            }

        return {
            "total_messages": len(self.message_log),
            "all_latencies": safe_stats(latencies),
            "intra_region_latencies": safe_stats(intra_region),
            "inter_region_latencies": safe_stats(inter_region),
        }


# === テスト用コード ===
if __name__ == "__main__":
    print("=== NetworkPhysics テスト ===\n")

    env = simpy.Environment()
    rng = random.Random(42)

    # NetworkPhysics初期化
    net = NetworkPhysics(env, N=21, rng=rng, enable_logging=True)

    # リージョン割り当ての確認
    print("\n--- リージョン割り当て ---")
    for node_id, region in net.node_regions.items():
        print(f"  Node {node_id}: {region.value}")

    # エッジ重み（ベースレイテンシ）の確認
    print("\n--- サンプルエッジレイテンシ ---")
    test_pairs = [(0, 1), (0, 7), (0, 14), (7, 14)]
    for src, dst in test_pairs:
        latency = net.get_base_latency(src, dst)
        src_r = net.get_node_region(src).value
        dst_r = net.get_node_region(dst).value
        print(f"  Node {src} ({src_r}) -> Node {dst} ({dst_r}): {latency*1000:.1f}ms")

    # メッセージ送信テスト
    print("\n--- メッセージ送信テスト ---")

    received = []

    def on_receive(dst, payload):
        received.append((dst, payload, env.now))

    # テスト送信
    net.send_message(0, 7, {"test": "intra_region"}, on_receive, "TEST_INTRA")
    net.send_message(0, 14, {"test": "inter_region"}, on_receive, "TEST_INTER")

    # シミュレーション実行
    env.run(until=1.0)

    # 結果確認
    print(f"\n--- 受信結果 ---")
    for dst, payload, time in received:
        print(f"  Node {dst} received at {time:.3f}s: {payload}")

    # 統計
    print(f"\n--- 通信統計 ---")
    stats = net.get_statistics()
    print(f"  Total Messages: {stats['total_messages']}")
    print(f"  All Latencies: {stats['all_latencies']}")
    print(f"  Intra-Region: {stats['intra_region_latencies']}")
    print(f"  Inter-Region: {stats['inter_region_latencies']}")

    print("\n=== テスト完了 ===")
