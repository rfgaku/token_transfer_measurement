"""
HyperliquidValidator: 独立したバリデータエージェント

Phase 2 Digital Twin Simulator のバリデータコンポーネント。
各バリデータが env.process で独立して動作し、
NetworkPhysics を介して RPC リクエスト（Deposit監視）を行う。

設計決定事項:
- N=21 バリデータ（Hyperliquid Mainnet準拠）
- 各バリデータが独立したポーリングループを持つ
- RPCリクエストはネットワーク経由（神の視点禁止）
- ログ出力で物理挙動を証明
"""

import simpy
import random
from typing import Dict, List, Optional, Callable, Any


class HyperliquidValidator:
    """
    Hyperliquidバリデータエージェント

    - 各インスタンスが SimPy プロセスとして独立動作
    - NetworkPhysics 経由で Arbitrum RPC にポーリング
    - Deposit検出時に署名 → Vote送信
    """

    def __init__(
        self,
        env: simpy.Environment,
        node_id: int,
        network_physics,  # NetworkPhysics インスタンス
        consensus_engine,  # PhysicalBridgeConsensus インスタンス
        config: dict,
        rng: Optional[random.Random] = None,
    ):
        """
        初期化

        Args:
            env: SimPy環境
            node_id: バリデータID (0-20)
            network_physics: NetworkPhysicsインスタンス
            consensus_engine: コンセンサスエンジン
            config: 設定辞書
            rng: 乱数生成器
        """
        self.env = env
        self.node_id = node_id
        self.network = network_physics
        self.consensus = consensus_engine
        self.config = config
        self.rng = rng if rng else random.Random()

        # ポーリング間隔パラメータ
        self.poll_interval_mu = float(config.get("bridge_poll_interval_mu", 5.0))
        self.poll_interval_sigma = float(config.get("bridge_poll_interval_sigma", 0.5))

        # RPCエンドポイントID（Arbitrum側）
        # 通常は node_id=0 を Arbitrum RPC とみなすが、専用IDを設定可能
        self.rpc_endpoint_id = int(config.get("_rpc_endpoint_id", 0))

        # 署名処理時間
        self.sign_delay_min = float(config.get("validator_sign_delay_min", 0.01))
        self.sign_delay_max = float(config.get("validator_sign_delay_max", 0.05))

        # ログ設定
        self.enable_logging = config.get("enable_validator_logging", True)

        # 統計用
        self.rpc_requests_count = 0
        self.votes_sent_count = 0
        self.total_rpc_latency = 0.0

        # リージョン（NetworkPhysicsから取得）
        self.region = self.network.get_node_region(node_id).value

        # 処理済みDeposit（重複投票防止）
        self._processed_deposits = set()

        # ポーリングループを起動
        self.env.process(self._polling_loop())

        if self.enable_logging:
            print(
                f"[Validator {self.node_id}] 初期化完了 "
                f"(Region: {self.region}, Poll Interval: {self.poll_interval_mu}s)"
            )

    def _polling_loop(self):
        """
        RPCポーリングループ

        1. 初期オフセット（バリデータ間の非同期化）
        2. RPCリクエスト送信 → レスポンス待ち
        3. 新規Deposit検出時に署名 → Vote送信
        4. 次のポーリングまで待機
        """
        # 初期オフセット（非同期化のため）
        initial_offset = self.rng.uniform(0, self.poll_interval_mu * 2.0)
        yield self.env.timeout(initial_offset)

        if self.enable_logging:
            print(
                f"[Time: {self.env.now:.3f}s] Validator {self.node_id} "
                f"started polling (offset: {initial_offset:.3f}s)"
            )

        while True:
            # RPC リクエスト送信（物理的な通信をシミュレート）
            deposits = yield from self._fetch_deposits_via_rpc()

            # 新規Depositに対して署名 → Vote
            for deposit in deposits:
                tx_id = deposit.get("tx_id")
                if tx_id is not None and tx_id not in self._processed_deposits:
                    self._processed_deposits.add(tx_id)
                    self.env.process(self._sign_and_vote(deposit))

            # 次のポーリングまで待機
            interval = max(0.1, self.rng.gauss(self.poll_interval_mu, self.poll_interval_sigma))
            yield self.env.timeout(interval)

    def _fetch_deposits_via_rpc(self):
        """
        RPCリクエストをネットワーク経由で投げる（神の視点禁止）

        往復通信の物理遅延をシミュレートし、pending depositsを取得する。

        Yields:
            ネットワーク遅延の待機

        Returns:
            List[dict]: 検出されたDepositのリスト
        """
        request_start = self.env.now
        self.rpc_requests_count += 1

        if self.enable_logging:
            print(
                f"[Time: {request_start:.3f}s] Validator {self.node_id} "
                f"sending RPC request to Arbitrum (Request #{self.rpc_requests_count})"
            )

        # 往路遅延
        outbound_delay = self.network.get_latency(self.node_id, self.rpc_endpoint_id)
        yield self.env.timeout(outbound_delay)

        # RPC処理時間（Arbitrumノード側）
        processing_time = self.rng.uniform(0.005, 0.015)  # 5-15ms
        yield self.env.timeout(processing_time)

        # 復路遅延
        inbound_delay = self.network.get_latency(self.rpc_endpoint_id, self.node_id)
        yield self.env.timeout(inbound_delay)

        request_end = self.env.now
        rtt = request_end - request_start
        self.total_rpc_latency += rtt

        if self.enable_logging:
            print(
                f"[Time: {request_end:.3f}s] Validator {self.node_id} "
                f"received RPC response (RTT: {rtt*1000:.1f}ms)"
            )

        # コンセンサスエンジンから pending deposits を取得
        # （この時点でバリデータが「見える」deposits）
        deposits = self.consensus.get_pending_deposits_for_validator(
            self.node_id, request_end
        )
        return deposits

    def _sign_and_vote(self, deposit: dict):
        """
        Deposit に対して署名し、Vote を送信

        1. 署名処理（ローカル処理時間）
        2. Vote を consensus engine に登録（到達時刻を記録）

        Args:
            deposit: Deposit情報
        """
        tx_id = deposit.get("tx_id")
        sign_start = self.env.now

        if self.enable_logging:
            print(
                f"[Time: {sign_start:.3f}s] Validator {self.node_id} "
                f"signing deposit tx_id={tx_id}"
            )

        # 署名処理時間（ローカル計算）
        sign_delay = self.rng.uniform(self.sign_delay_min, self.sign_delay_max)
        yield self.env.timeout(sign_delay)

        vote_time = self.env.now
        self.votes_sent_count += 1

        if self.enable_logging:
            print(
                f"[Time: {vote_time:.3f}s] Validator {self.node_id} "
                f"voted for tx_id={tx_id} (Sign Time: {sign_delay*1000:.1f}ms)"
            )

        # コンセンサスエンジンに Vote を登録
        # （物理的な到達時刻を含めて記録）
        self.consensus.register_vote(
            node_id=self.node_id,
            tx_id=tx_id,
            vote_time=vote_time,
            region=self.region,
        )

    def get_statistics(self) -> Dict:
        """
        バリデータ統計を返す
        """
        avg_rpc_latency = (
            self.total_rpc_latency / self.rpc_requests_count
            if self.rpc_requests_count > 0
            else 0
        )
        return {
            "node_id": self.node_id,
            "region": self.region,
            "rpc_requests": self.rpc_requests_count,
            "votes_sent": self.votes_sent_count,
            "avg_rpc_latency_ms": avg_rpc_latency * 1000,
            "processed_deposits": len(self._processed_deposits),
        }


class PhysicalBridgeConsensus:
    """
    物理的なVote到達時間を計測するコンセンサスエンジン

    BFT物理挙動:
    1. Validators -> Leader: Vote を送信（片道）
    2. Leader: 定足数を集計し、Block/QCを作成（計算時間）
    3. Leader -> Validators: QC をブロードキャスト（復路）
    4. Validators: QCを受け取って初めて Finalized

    - 各Voteの到達時刻を記録
    - 定足数（2/3 + 1）に達した時点でQCブロードキャスト開始
    - 地理的分布による標準偏差の再現
    """

    def __init__(
        self,
        env: simpy.Environment,
        N: int = 21,
        config: Optional[dict] = None,
        network_physics=None,  # NetworkPhysics インスタンス（QCブロードキャスト用）
    ):
        """
        初期化

        Args:
            env: SimPy環境
            N: バリデータ数（デフォルト21）
            config: 設定辞書
            network_physics: NetworkPhysicsインスタンス（QCブロードキャスト遅延計算用）
        """
        self.env = env
        self.N = N
        self.config = config if config else {}
        self.network = network_physics  # QCブロードキャスト用

        # 定足数: 2/3 + 1 = 15 (N=21の場合)
        self.quorum_size = int(N * 2 / 3) + 1

        # リーダーID（ラウンドロビンで選出、初期は0）
        self.leader_id = 0

        # QC作成処理時間（リーダー側）
        self.qc_creation_time = float(self.config.get("qc_creation_time_s", 0.01))  # 10ms

        # Pending Deposits (Sequencerから通知されたがまだ処理中のもの)
        # tx_id -> {"timestamp": float, "payload": dict}
        self.pending_deposits: Dict[int, dict] = {}

        # Vote到着記録
        # tx_id -> {node_id: {"vote_time": float, "region": str}}
        self.vote_arrivals: Dict[int, Dict[int, dict]] = {}

        # 確定済みDeposit
        # tx_id -> {"quorum_time": float, "votes": int}
        self.finalized: Dict[int, dict] = {}

        # コールバック
        self.on_deposit_finalized: Optional[Callable[[int, float], None]] = None

        # ログ設定
        self.enable_logging = self.config.get("enable_consensus_logging", True)

        if self.enable_logging:
            print(f"[PhysicalBridgeConsensus] 初期化完了: N={N}, Quorum={self.quorum_size}")

    def notify_deposit(self, tx_id: int, timestamp: float, payload: dict = None):
        """
        Sequencer からの Deposit 通知

        バリデータがRPCでポーリングした際に「見える」ようになる。

        Args:
            tx_id: トランザクションID
            timestamp: Deposit発生時刻
            payload: 追加ペイロード
        """
        if tx_id not in self.pending_deposits:
            self.pending_deposits[tx_id] = {
                "timestamp": timestamp,
                "payload": payload or {},
            }
            if self.enable_logging:
                print(
                    f"[Time: {self.env.now:.3f}s] [Consensus] "
                    f"New deposit registered: tx_id={tx_id}"
                )

    def get_pending_deposits_for_validator(
        self, node_id: int, query_time: float
    ) -> List[dict]:
        """
        バリデータがRPCで問い合わせた時点で「見える」Depositを返す

        条件:
        - Depositの発生時刻 < query_time
        - まだ確定していない
        - このバリデータがまだVoteしていない

        Args:
            node_id: バリデータID
            query_time: クエリ時刻

        Returns:
            List[dict]: 見えるDepositのリスト
        """
        result = []
        for tx_id, info in self.pending_deposits.items():
            # 確定済みはスキップ
            if tx_id in self.finalized:
                continue

            # Depositの発生時刻より後でないと見えない
            if info["timestamp"] > query_time:
                continue

            # このバリデータがまだVoteしていない
            if tx_id in self.vote_arrivals:
                if node_id in self.vote_arrivals[tx_id]:
                    continue

            result.append({
                "tx_id": tx_id,
                "timestamp": info["timestamp"],
                **info["payload"],
            })

        return result

    def register_vote(
        self,
        node_id: int,
        tx_id: int,
        vote_time: float,
        region: str = "UNKNOWN",
    ):
        """
        Vote の到着を記録

        Args:
            node_id: 投票したバリデータID
            tx_id: トランザクションID
            vote_time: 投票時刻
            region: バリデータのリージョン
        """
        # 既に確定済みならスキップ
        if tx_id in self.finalized:
            return

        # Vote記録を初期化
        if tx_id not in self.vote_arrivals:
            self.vote_arrivals[tx_id] = {}

        # 重複投票チェック
        if node_id in self.vote_arrivals[tx_id]:
            return

        # Vote登録
        self.vote_arrivals[tx_id][node_id] = {
            "vote_time": vote_time,
            "region": region,
        }

        vote_count = len(self.vote_arrivals[tx_id])

        if self.enable_logging:
            print(
                f"[Time: {self.env.now:.3f}s] [Consensus] "
                f"Vote received: tx_id={tx_id}, node={node_id} ({region}), "
                f"count={vote_count}/{self.quorum_size}"
            )

        # 定足数チェック → QCブロードキャスト開始
        if vote_count >= self.quorum_size:
            if tx_id not in self.finalized:
                self.env.process(self._start_qc_broadcast(tx_id))

    def _start_qc_broadcast(self, tx_id: int):
        """
        QCブロードキャストプロセス（BFT物理挙動のStep 2-4）

        1. Leader: QC作成処理時間
        2. Leader -> Validators: QCブロードキャスト遅延
        3. 最後のValidatorがQC受信した時点でFinalized

        Args:
            tx_id: トランザクションID
        """
        if tx_id in self.finalized:
            return

        quorum_reached_time = self.env.now
        vote_count = len(self.vote_arrivals.get(tx_id, {}))

        if self.enable_logging:
            print(
                f"[Time: {quorum_reached_time:.3f}s] [Consensus] "
                f"QUORUM REACHED: tx_id={tx_id}, votes={vote_count}, "
                f"Starting QC broadcast from Leader {self.leader_id}"
            )

        # Step 2: Leader側でQC作成処理
        yield self.env.timeout(self.qc_creation_time)

        # Step 3: Leader -> Validators へのQCブロードキャスト
        # 全バリデータへの最大遅延（最も遠いノードへの到達時間）を計算
        if self.network:
            max_qc_latency = 0.0
            for validator_id in range(self.N):
                if validator_id != self.leader_id:
                    latency = self.network.get_latency(self.leader_id, validator_id)
                    max_qc_latency = max(max_qc_latency, latency)

            if self.enable_logging:
                print(
                    f"[Time: {self.env.now:.3f}s] [Consensus] "
                    f"Leader {self.leader_id} broadcasting QC for tx_id={tx_id} "
                    f"(Max Latency: {max_qc_latency*1000:.1f}ms)"
                )

            # QCが全ノードに届くまで待機
            yield self.env.timeout(max_qc_latency)
        else:
            # NetworkPhysicsがない場合のフォールバック
            fallback_latency = 0.15  # 150ms (リージョン間の平均)
            yield self.env.timeout(fallback_latency)

        # Step 4: Finalized
        self._finalize(tx_id, quorum_reached_time, vote_count)

    def _finalize(self, tx_id: int, quorum_reached_time: float, vote_count: int):
        """
        Deposit確定処理（QCブロードキャスト完了後）

        Args:
            tx_id: 確定するトランザクションID
            quorum_reached_time: 定足数到達時刻
            vote_count: 投票数
        """
        if tx_id in self.finalized:
            return

        finalize_time = self.env.now

        # Vote時刻の統計
        vote_times = [
            v["vote_time"]
            for v in self.vote_arrivals.get(tx_id, {}).values()
        ]
        first_vote = min(vote_times) if vote_times else quorum_reached_time
        last_vote = max(vote_times) if vote_times else quorum_reached_time

        # QCブロードキャスト遅延を含めた統計
        qc_broadcast_latency = finalize_time - quorum_reached_time

        self.finalized[tx_id] = {
            "quorum_time": quorum_reached_time,
            "finalize_time": finalize_time,
            "votes": vote_count,
            "first_vote_time": first_vote,
            "last_vote_time": last_vote,
            "vote_spread": last_vote - first_vote,
            "qc_broadcast_latency": qc_broadcast_latency,
        }

        if self.enable_logging:
            print(
                f"[Time: {finalize_time:.3f}s] [Consensus] "
                f"FINALIZED: tx_id={tx_id}, "
                f"QC Broadcast Latency: {qc_broadcast_latency*1000:.1f}ms, "
                f"vote_spread={self.finalized[tx_id]['vote_spread']*1000:.1f}ms"
            )

        # コールバック（Finalize時刻を使用）
        if self.on_deposit_finalized:
            self.on_deposit_finalized(tx_id, finalize_time)

    def get_statistics(self) -> Dict:
        """
        コンセンサス統計を返す
        """
        if not self.finalized:
            return {"finalized_count": 0}

        spreads = [v["vote_spread"] for v in self.finalized.values()]

        import statistics
        return {
            "finalized_count": len(self.finalized),
            "pending_count": len(self.pending_deposits) - len(self.finalized),
            "vote_spread_mean_ms": statistics.mean(spreads) * 1000,
            "vote_spread_std_ms": statistics.stdev(spreads) * 1000 if len(spreads) > 1 else 0,
        }


# === テスト用コード ===
if __name__ == "__main__":
    from sim.network_physics import NetworkPhysics

    print("=== HyperliquidValidator + PhysicalBridgeConsensus テスト ===\n")

    env = simpy.Environment()
    rng = random.Random(42)

    # 設定
    config = {
        "bridge_poll_interval_mu": 2.0,  # テスト用に短く
        "bridge_poll_interval_sigma": 0.3,
        "_rpc_endpoint_id": 0,
        "enable_validator_logging": True,
        "enable_consensus_logging": True,
    }

    # NetworkPhysics初期化
    net = NetworkPhysics(env, N=21, rng=rng, enable_logging=False)

    # コンセンサスエンジン初期化
    consensus = PhysicalBridgeConsensus(env, N=21, config=config)

    # バリデータ初期化（N=21）
    validators = []
    for i in range(21):
        v = HyperliquidValidator(
            env=env,
            node_id=i,
            network_physics=net,
            consensus_engine=consensus,
            config=config,
            rng=random.Random(42 + i),
        )
        validators.append(v)

    # テスト用Deposit登録
    print("\n--- Deposit登録 ---")
    consensus.notify_deposit(tx_id=1, timestamp=0.5, payload={"amount": 100})
    consensus.notify_deposit(tx_id=2, timestamp=1.0, payload={"amount": 200})

    # シミュレーション実行
    print("\n--- シミュレーション実行（10秒間）---")
    env.run(until=10.0)

    # 結果確認
    print("\n--- コンセンサス統計 ---")
    stats = consensus.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 一部バリデータの統計
    print("\n--- バリデータ統計（サンプル）---")
    for v in validators[:3]:
        vstats = v.get_statistics()
        print(f"  Node {vstats['node_id']} ({vstats['region']}): "
              f"RPC={vstats['rpc_requests']}, Votes={vstats['votes_sent']}, "
              f"Avg RTT={vstats['avg_rpc_latency_ms']:.1f}ms")

    print("\n=== テスト完了 ===")
