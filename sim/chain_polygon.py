import simpy
import random
from collections import deque

class BorBlock:
    def __init__(self, height, producer_id, parent_hash, sprint_idx, timestamp):
        self.height = height
        self.producer_id = producer_id
        self.parent_hash = parent_hash
        self.sprint_idx = sprint_idx
        self.timestamp = timestamp
        # ハッシュの簡易表現
        self.hash = f"{height}:{producer_id}:{random.randint(0,9999)}"

    def __repr__(self):
        return f"Block(H={self.height}, P={self.producer_id}, Sprint={self.sprint_idx})"

class BorNode:
    """
    Polygon Bor (Block Producer) Node.
    - 16ブロック(Sprint)ごとにProducer選出ロジックが変わる。
    - ネットワーク遅延によりForkが発生する。
    """
    def __init__(self, node_id, env, protocol, producers_list):
        self.node_id = node_id
        self.env = env
        self.protocol = protocol
        self.producers_list = producers_list # 全Borノードのリスト
        
        # Local Chain State
        self.local_chain = [] 
        # Genesis Block
        genesis = BorBlock(0, "GENESIS", "0x0", 0, 0)
        self.local_chain.append(genesis)
        
        # Inbox for messages (Blocks)
        self.inbox = simpy.Store(env)
        
        # Start processes
        self.env.process(self._consumer_loop())
        self.env.process(self._producer_loop())

    def get_head(self):
        return self.local_chain[-1]

    def _consumer_loop(self):
        while True:
            msg = yield self.inbox.get()
            if msg["type"] == "BLOCK":
                self._handle_block(msg["payload"])

    def _handle_block(self, block):
        head = self.get_head()
        
        # 基本的なLongest Chainルール
        if block.height > head.height:
            if block.parent_hash == head.hash:
                # Extend chain
                self.local_chain.append(block)
                # フォーク解決や単なる拡張
            else:
                # 親が一致しない -> Fork or Gap
                # 簡易シミュレーションとして、高さが勝っていればReorgして採用する動きを模倣
                # 実際はBlock Requestなどが必要だが、ここでは「強いチェーンに乗り換えた」ことにする
                prev_head_hash = head.hash
                self.local_chain.append(block) # Append anyway for sim simplicity, logic implies "switched head"
                
                # Reorg検知 (親が違うのに高さ更新 = 別ブランチ採用)
                print(f"[Bor:{self.node_id}] Fork Detected! Switched from {prev_head_hash} to {block.hash} (len={block.height})")
                if self.protocol.on_reorg:
                    self.protocol.on_reorg(2.0) # 2秒程度の深いReorgを通知

    def _producer_loop(self):
        """
        自分がProducerの担当スロットであればブロックを生成する。
        """
        while True:
            # 2秒ごとにスロット進行チェック
            current_time = self.env.now
            # スロット計算: Genesisからの経過時間 / Block Time
            slot = int(current_time // self.protocol.block_time) + 1
            
            # 自分がこのスロットの担当か？
            sprint_len = self.protocol.sprint_length
            sprint_idx = (slot - 1) // sprint_len
            
            # GlobalなProducer選出 (Round Robin over producers_list)
            # Sprintごとに選出順序が変わるが、ここではシンプルにSprint Indexを加味してシフトさせる
            num_producers = len(self.producers_list)
            
            # Sprint 0: [0, 1, 2, 3]
            # Sprint 1: [1, 2, 3, 0] ...
            producer_idx_in_sprint = (slot - 1) % num_producers
            # Sprintごとに開始オフセットをずらす
            target_node_idx = (producer_idx_in_sprint + sprint_idx) % num_producers
            target_node = self.producers_list[target_node_idx]

            # Sprint切り替えログ
            if (slot - 1) % sprint_len == 0 and slot > 1:
                # 自分が代表してログ出力 (Node 0 only)
                if self.node_id == self.producers_list[0].node_id:
                     print(f"[Bor] Sprint {sprint_idx-1} ended. Producer shuffle for Sprint {sprint_idx}.")

            if target_node.node_id == self.node_id:
                # It's my turn!
                self._produce_block(slot, sprint_idx)
            
            # Wait for next slot boundary
            next_slot_time = slot * self.protocol.block_time
            delay = next_slot_time - self.env.now
            if delay < 0: delay = 0 # 遅れてる場合は即時
            yield self.env.timeout(delay + 0.01) # 少しずらして無限ループ防止

    def _produce_block(self, slot, sprint_idx):
        head = self.get_head()
        new_block = BorBlock(
            height=slot,
            producer_id=self.node_id,
            parent_hash=head.hash,
            sprint_idx=sprint_idx,
            timestamp=self.env.now
        )
        
        # Self-update
        self.local_chain.append(new_block)
        print(f"[Bor:{self.node_id}] Produced Block #{new_block.height} (Sprint {sprint_idx})")
        
        # Notify Sequencer (Soft Finality)
        if self.protocol.on_soft_finality:
            self.protocol.on_soft_finality(new_block)
            
        # Broadcast to others
        self._broadcast_block(new_block)

    def _broadcast_block(self, block):
        # シミュレートされたネットワーク遅延で他ノードへ送信
        for peer in self.producers_list:
            if peer.node_id == self.node_id: continue
            
            # 基本遅延
            latency = random.uniform(0.05, 0.2)
            
            # INJECT FAULT: Sprint境界でのLatency SpikeによるFork誘発
            # 特定条件下（例えばSprintの変わり目）で遅延を爆増させる
            if (block.height - 1) % self.protocol.sprint_length == 0:
                 if random.random() < 0.4: # 40%で発生
                     latency += 2.5 # 2.5秒遅延 -> 次のスロットとかぶる -> Fork発生
                     print(f"[Network] Latency Spike! Block #{block.height} delayed by {latency:.2f}s to {peer.node_id}")

            self.env.process(self._deliver_msg(peer, {"type": "BLOCK", "payload": block}, latency))

    def _deliver_msg(self, target_node, msg, latency):
        yield self.env.timeout(latency)
        target_node.inbox.put(msg)


class HeimdallValidator:
    """
    Heimdall Validator Node.
    - Checkpointを作成し、Borチェーンの状態をL1へ確定させる。
    """
    def __init__(self, node_id, env, protocol, bor_nodes):
        self.node_id = node_id
        self.env = env
        self.protocol = protocol
        self.bor_nodes = bor_nodes
        self.last_checkpoint_height = 0
        
        self.env.process(self._checkpoint_loop())

    def _checkpoint_loop(self):
        while True:
            # Checkpoint Interval wait
            interval_s = self.protocol.checkpoint_interval * self.protocol.block_time
            yield self.env.timeout(interval_s)
            
            # Borノード（例えばNode0）からcanonical chainを取得
            # 実際はHeimdallノード自体がBorブロックを同期しているが、ここでは省略
            target_bor = self.bor_nodes[0]
            head = target_bor.get_head()
            
            if head.height > self.last_checkpoint_height:
                # Checkpoint Process Start
                print(f"[Heimdall] Checkpoint #{head.height // self.protocol.checkpoint_interval} Pending... (Waiting for signatures)")
                
                # Signature Aggregation Delay (Simulated)
                yield self.env.timeout(1.0) 
                
                # L1 Commit Delay (Ethereum is slow)
                yield self.env.timeout(10.0)
                
                self.last_checkpoint_height = head.height
                print(f"[Heimdall] Checkpoint Committed for Block #{head.height} (Finalized).")
                
                # Notify Hard Finality
                if self.protocol.on_hard_finality:
                    self.protocol.on_hard_finality(head.height)


class PolygonProtocol:
    """
    Manager for the Polygon Simulation.
    Initializes nodes and connects them.
    """
    def __init__(self, env, num_bor=4):
        self.env = env
        self.block_time = 2.0
        self.sprint_length = 16
        self.checkpoint_interval = 32 # Sim用に短縮 (実際はもっと長い)
        
        # Hooks
        self.on_soft_finality = None
        self.on_hard_finality = None
        self.on_reorg = None
        
        # Create Bor Nodes
        self.bor_nodes = []
        for i in range(num_bor):
            node = BorNode(f"Node-{i}", env, self, self.bor_nodes) # pass list ref
            self.bor_nodes.append(node)
            
        # Create Heimdall Validator (Simplified to 1 entity representing the set)
        self.heimdall = HeimdallValidator("Heimdall-Set", env, self, self.bor_nodes)

    def start(self):
        # Nodes start automatically in __init__
        pass
