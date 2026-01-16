"""
ArbitrumBatcher: Arbitrum Batch Poster の物理シミュレーション

Phase 2.5 Withdraw物理実装のコアコンポーネント。
L2トランザクションをバッファリングし、トリガー条件に基づいてL1に投稿する。

物理要件:
- simpy.Container でバッファを表現
- トリガー条件: データ量超過 OR 時間経過
- L1確定遅延を物理的にシミュレート
"""

import simpy
import random
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum


class BatcherState(Enum):
    """Batcher状態マシンの状態"""
    IDLE = "IDLE"           # 待機中
    BUFFERING = "BUFFERING" # バッファリング中
    POSTING = "POSTING"     # L1投稿中
    CONFIRMING = "CONFIRMING"  # L1確定待ち


class ArbitrumBatcher:
    """
    Arbitrum Batch Poster の物理シミュレーション

    状態遷移:
    IDLE -> BUFFERING: TX受信
    BUFFERING -> POSTING: 閾値超過 OR タイムアウト
    POSTING -> CONFIRMING: L1投稿完了
    CONFIRMING -> IDLE: L1確定

    トリガー条件（OR条件）:
    - データ量: バッファサイズ >= size_threshold_kb
    - 時間経過: 前回Post後 max_wait_s 経過
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: dict,
        network_physics=None,
        rng: Optional[random.Random] = None,
    ):
        """
        初期化

        Args:
            env: SimPy環境
            config: 設定辞書
            network_physics: NetworkPhysicsインスタンス（L1通信遅延計算用）
            rng: 乱数生成器
        """
        self.env = env
        self.config = config
        self.network = network_physics
        self.rng = rng if rng else random.Random()

        # トリガー閾値
        self.size_threshold_kb = float(config.get("batcher_size_threshold_kb", 120))
        self.max_wait_s = float(config.get("batcher_max_wait_s", 300))  # 5分

        # L1確定パラメータ
        self.l1_block_time_s = float(config.get("l1_block_time_s", 12.0))  # Ethereum block time
        self.l1_confirmations = int(config.get("l1_confirmations", 2))  # 確定までのブロック数

        # バッファ（simpy.Container）
        # capacity=inf で無制限、init=0 で空から開始
        self.buffer = simpy.Container(env, capacity=float('inf'), init=0)

        # 状態
        self.state = BatcherState.IDLE
        self.last_post_time = 0.0

        # Pending TXs: (tx_id, submit_time, size_bytes)
        self.pending_txs: List[Tuple[int, float, int]] = []

        # 投稿済みバッチ履歴
        self.posted_batches: List[Dict] = []

        # コールバック
        self.on_batch_posted: Optional[Callable[[int, float, float], None]] = None
        self.on_l1_confirmed: Optional[Callable[[int, float], None]] = None

        # ログ設定
        self.enable_logging = config.get("enable_batcher_logging", True)

        # 監視プロセス起動
        self.env.process(self._monitor_loop())

        if self.enable_logging:
            print(
                f"[ArbitrumBatcher] 初期化完了: "
                f"size_threshold={self.size_threshold_kb}KB, "
                f"max_wait={self.max_wait_s}s"
            )

    def submit_withdrawal(self, tx_id: int, submit_time: float, size_bytes: int = 1500):
        """
        出金TXをバッファに追加

        Args:
            tx_id: トランザクションID
            submit_time: 送信時刻
            size_bytes: TXサイズ（デフォルト1500バイト = 典型的なERC20転送）
        """
        self.pending_txs.append((tx_id, submit_time, size_bytes))

        # バッファにサイズを追加（KB単位）
        size_kb = size_bytes / 1024
        self.buffer.put(size_kb)

        # 状態遷移
        if self.state == BatcherState.IDLE:
            self.state = BatcherState.BUFFERING

        if self.enable_logging:
            print(
                f"[Time: {self.env.now:.3f}s] [Batcher] "
                f"TX {tx_id} buffered (Size: {size_bytes}B, "
                f"Buffer: {self.buffer.level:.1f}KB/{self.size_threshold_kb}KB)"
            )

    def _monitor_loop(self):
        """
        トリガー条件を監視するループ

        チェック間隔: 100ms
        """
        while True:
            yield self.env.timeout(0.1)  # 100ms間隔

            # 状態チェック
            if self.state not in [BatcherState.BUFFERING, BatcherState.IDLE]:
                continue

            if not self.pending_txs:
                continue

            should_post = False
            trigger_reason = ""

            # 条件1: データ量超過
            if self.buffer.level >= self.size_threshold_kb:
                should_post = True
                trigger_reason = f"SIZE_TRIGGER ({self.buffer.level:.1f}KB >= {self.size_threshold_kb}KB)"

            # 条件2: 時間経過
            elif (self.env.now - self.last_post_time) >= self.max_wait_s:
                should_post = True
                trigger_reason = f"TIME_TRIGGER ({self.env.now - self.last_post_time:.1f}s >= {self.max_wait_s}s)"

            if should_post:
                self.env.process(self._post_to_l1(trigger_reason))

    def _post_to_l1(self, trigger_reason: str):
        """
        L1への投稿処理

        1. バッファからバッチを取り出し
        2. L1に投稿（トランザクション送信）
        3. L1確定待ち
        4. 確定通知

        Args:
            trigger_reason: トリガー理由（ログ用）
        """
        if self.state == BatcherState.POSTING:
            return

        self.state = BatcherState.POSTING

        # バッチ作成
        batch = list(self.pending_txs)
        batch_size_kb = self.buffer.level
        batch_tx_count = len(batch)

        # バッファクリア
        self.pending_txs = []
        if self.buffer.level > 0:
            yield self.buffer.get(self.buffer.level)

        post_start_time = self.env.now

        if self.enable_logging:
            print(
                f"[Time: {post_start_time:.3f}s] [Batcher] "
                f"Posting batch to L1: {batch_tx_count} TXs, {batch_size_kb:.1f}KB, "
                f"Reason: {trigger_reason}"
            )

        # L1投稿遅延（ブロックに含まれるまで）
        # 次のブロックまでの待機時間（0〜block_time）
        block_wait = self.rng.uniform(0, self.l1_block_time_s)
        yield self.env.timeout(block_wait)

        self.state = BatcherState.CONFIRMING

        if self.enable_logging:
            print(
                f"[Time: {self.env.now:.3f}s] [Batcher] "
                f"Batch included in L1 block, waiting for {self.l1_confirmations} confirmations..."
            )

        # L1確定待ち（confirmation blocks）
        confirmation_time = self.l1_block_time_s * self.l1_confirmations
        yield self.env.timeout(confirmation_time)

        l1_confirm_time = self.env.now
        self.last_post_time = l1_confirm_time

        # バッチ履歴記録
        batch_info = {
            "post_time": post_start_time,
            "confirm_time": l1_confirm_time,
            "tx_count": batch_tx_count,
            "size_kb": batch_size_kb,
            "trigger_reason": trigger_reason,
        }
        self.posted_batches.append(batch_info)

        if self.enable_logging:
            print(
                f"[Time: {l1_confirm_time:.3f}s] [Batcher] "
                f"Batch confirmed on L1: {batch_tx_count} TXs, "
                f"L1 Latency: {l1_confirm_time - post_start_time:.1f}s"
            )

        # 各TXに対してコールバック
        for tx_id, submit_time, size_bytes in batch:
            if self.on_batch_posted:
                self.on_batch_posted(tx_id, submit_time, l1_confirm_time)
            if self.on_l1_confirmed:
                self.on_l1_confirmed(tx_id, l1_confirm_time)

        # 状態リセット
        self.state = BatcherState.IDLE

    def get_statistics(self) -> Dict:
        """
        Batcher統計を返す
        """
        if not self.posted_batches:
            return {"posted_count": 0}

        latencies = [b["confirm_time"] - b["post_time"] for b in self.posted_batches]
        sizes = [b["size_kb"] for b in self.posted_batches]
        tx_counts = [b["tx_count"] for b in self.posted_batches]

        import statistics
        return {
            "posted_count": len(self.posted_batches),
            "total_txs": sum(tx_counts),
            "avg_batch_size_kb": statistics.mean(sizes),
            "avg_l1_latency_s": statistics.mean(latencies),
            "avg_txs_per_batch": statistics.mean(tx_counts),
        }


class L1ToL2Relayer:
    """
    L1確定後のL2へのリレー

    Dispute Period後にL2（Arbitrum）でWithdrawが確定する。
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: dict,
        network_physics=None,
        rng: Optional[random.Random] = None,
    ):
        """
        初期化

        Args:
            env: SimPy環境
            config: 設定辞書
            network_physics: NetworkPhysicsインスタンス
            rng: 乱数生成器
        """
        self.env = env
        self.config = config
        self.network = network_physics
        self.rng = rng if rng else random.Random()

        # Dispute Period（チャレンジ可能期間）
        self.dispute_period_s = float(config.get("bridge_dispute_period_s", 200.0))

        # L1→L2リレー遅延
        self.relay_latency_mu = float(config.get("relay_latency_mu_s", 1.0))
        self.relay_latency_sigma = float(config.get("relay_latency_sigma_s", 0.3))

        # 完了済みWithdraw
        self.completed_withdrawals: Dict[int, Dict] = {}

        # コールバック
        self.on_withdraw_finalized: Optional[Callable[[int, float], None]] = None

        # ログ設定
        self.enable_logging = config.get("enable_relayer_logging", True)

        if self.enable_logging:
            print(
                f"[L1ToL2Relayer] 初期化完了: "
                f"dispute_period={self.dispute_period_s}s"
            )

    def start_relay(self, tx_id: int, l1_confirm_time: float, submit_time: float):
        """
        リレープロセスを開始

        Args:
            tx_id: トランザクションID
            l1_confirm_time: L1確定時刻
            submit_time: 元の送信時刻
        """
        self.env.process(self._relay_process(tx_id, l1_confirm_time, submit_time))

    def _relay_process(self, tx_id: int, l1_confirm_time: float, submit_time: float):
        """
        リレープロセス

        1. Dispute Period待機
        2. L1→L2メッセージ伝播
        3. Withdraw確定

        Args:
            tx_id: トランザクションID
            l1_confirm_time: L1確定時刻
            submit_time: 元の送信時刻
        """
        dispute_start = self.env.now

        if self.enable_logging:
            print(
                f"[Time: {dispute_start:.3f}s] [Relayer] "
                f"TX {tx_id} entering dispute period ({self.dispute_period_s}s)"
            )

        # 1. Dispute Period
        yield self.env.timeout(self.dispute_period_s)

        # 2. L1→L2メッセージ伝播
        if self.network:
            # NetworkPhysics使用（リージョン間遅延）
            # L1ノード(0) → L2ノード(1) と仮定
            relay_latency = self.network.get_latency(0, 1)
        else:
            # フォールバック: ログ正規分布
            relay_latency = max(0.1, self.rng.gauss(self.relay_latency_mu, self.relay_latency_sigma))

        yield self.env.timeout(relay_latency)

        # 3. 確定
        finalize_time = self.env.now
        total_latency = finalize_time - submit_time

        self.completed_withdrawals[tx_id] = {
            "submit_time": submit_time,
            "l1_confirm_time": l1_confirm_time,
            "finalize_time": finalize_time,
            "total_latency": total_latency,
            "dispute_period": self.dispute_period_s,
            "relay_latency": relay_latency,
        }

        if self.enable_logging:
            print(
                f"[Time: {finalize_time:.3f}s] [Relayer] "
                f"TX {tx_id} FINALIZED on L2: "
                f"Total Latency: {total_latency:.1f}s"
            )

        # コールバック
        if self.on_withdraw_finalized:
            self.on_withdraw_finalized(tx_id, finalize_time)

    def get_statistics(self) -> Dict:
        """
        リレー統計を返す
        """
        if not self.completed_withdrawals:
            return {"completed_count": 0}

        latencies = [w["total_latency"] for w in self.completed_withdrawals.values()]

        import statistics
        return {
            "completed_count": len(self.completed_withdrawals),
            "avg_total_latency_s": statistics.mean(latencies),
            "std_total_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "min_latency_s": min(latencies),
            "max_latency_s": max(latencies),
        }


# === テスト用コード ===
if __name__ == "__main__":
    print("=== ArbitrumBatcher + L1ToL2Relayer テスト ===\n")

    env = simpy.Environment()
    rng = random.Random(42)

    # 設定
    config = {
        "batcher_size_threshold_kb": 50,  # テスト用に小さく
        "batcher_max_wait_s": 30,  # テスト用に短く
        "l1_block_time_s": 12.0,
        "l1_confirmations": 2,
        "bridge_dispute_period_s": 60,  # テスト用に短く
        "enable_batcher_logging": True,
        "enable_relayer_logging": True,
    }

    # Batcher初期化
    batcher = ArbitrumBatcher(env, config, rng=rng)

    # Relayer初期化
    relayer = L1ToL2Relayer(env, config, rng=rng)

    # Batcher→Relayer接続
    def on_l1_confirmed(tx_id, l1_confirm_time):
        # submit_timeを取得するため、記録が必要
        relayer.start_relay(tx_id, l1_confirm_time, env.now - 50)  # 仮のsubmit_time

    batcher.on_l1_confirmed = on_l1_confirmed

    # テスト用TX投入
    def traffic_gen():
        yield env.timeout(1.0)
        for i in range(1, 51):  # 50 TXs
            batcher.submit_withdrawal(tx_id=i, submit_time=env.now, size_bytes=1500)
            yield env.timeout(0.5)  # 0.5秒間隔

    env.process(traffic_gen())

    # シミュレーション実行
    print("\n--- シミュレーション実行（300秒間）---")
    env.run(until=300.0)

    # 結果確認
    print("\n--- Batcher統計 ---")
    bstats = batcher.get_statistics()
    for k, v in bstats.items():
        print(f"  {k}: {v}")

    print("\n--- Relayer統計 ---")
    rstats = relayer.get_statistics()
    for k, v in rstats.items():
        print(f"  {k}: {v}")

    print("\n=== テスト完了 ===")
