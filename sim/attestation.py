import simpy
import random

class AttestationService:
    """
    Simulates Circle CCTP Attestation Service.
    Instead of a decentralized consensus (HyperBFT), this relies on a trusted set of Attesters.
    
    Flow:
    1. Listen to Sequencer (Source Chain)
    2. Wait for N confirmations (Finality) or just observe Soft Finality?
       CCTP requires "Finality" on source. Arbitrum Finality takes time (L1 batching).
       But for this sim, we might simulate "Circle's local confirmation" or wait for L1.
       Let's assume Attesters wait for L1 Finality (Safe) or Soft Finality (Fast).
       We'll simulate "Fast Mode" where they trust Sequencer Soft Finality + minimal delay.
    3. Aggregate Signatures (simulated delay).
    4. Emit Attestation (Used for Minting).
    """
    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        # Attestation delay parameters
        # CCTP is reasonably fast, but has API latency + signing latency.
        self.attestation_delay_mu = float(config.get("cctp_attestation_delay_mu", 2.0))
        self.attestation_delay_sigma = float(config.get("cctp_attestation_delay_sigma", 0.5))
        
        # Safety Margin: Extra wait time before signing (to mitigate reorg risk)
        self.safety_margin = float(config.get("cctp_safety_margin_s", 0.0))
        
        # Callback to trigger Mint (Relayer)
        self.on_attestation_complete = None
        
        # Internal state
        self.pending_txs = set()

    def on_sequencer_event(self, payload):
        """
        Called when Sequencer emits an event (Soft Finality).
        """
        tx_id = payload["tx_id"]
        # In reality, CCTP might wait for L1 Finality. 
        # Here we simulate the processing time starting from Soft Finality.
        self.env.process(self._process_attestation(tx_id))

    def _process_attestation(self, tx_id):
        # Simulate Attestation Latency
        # - Observing block
        # - Validating
        # - Signing by multiple attesters (API calls)
        delay = random.gauss(self.attestation_delay_mu, self.attestation_delay_sigma)
        if delay < 0.1: delay = 0.1
        
        # Add Safety Margin
        delay += self.safety_margin
        
        yield self.env.timeout(delay)
        
        # Attestation Ready
        # print(f"[DEBUG] Attestation ready for {tx_id} at {self.env.now}")
        if self.on_attestation_complete:
            self.on_attestation_complete(tx_id, self.env.now)

    def set_callback(self, callback_fn):
        self.on_attestation_complete = callback_fn
