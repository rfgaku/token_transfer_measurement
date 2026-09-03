#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""withdraw 用残高確認: Arbitrum / HyperEVM / HyperCore(spot+perp) の USDC 残高を一覧表示。

withdraw（HyperCore → Arbitrum）の前後で全レイヤの残高を見て、
HyperCore から引落 → Arbitrum へ着金（net）したことを目視確認するための小スクリプト。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from decimal import Decimal

import requests
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

from t4_cctp.withdraw import config as cfg

load_dotenv()

ERC20 = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
]


def main():
    addr = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    low = addr.lower()
    print("=" * 64)
    print(f" T4 withdraw 残高確認  user={addr}")
    print("=" * 64)

    # --- Arbitrum ---
    try:
        w3 = Web3(Web3.HTTPProvider(cfg.ARB_RPC_URL))
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        usdc = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), abi=ERC20)
        bal = usdc.functions.balanceOf(addr).call()
        eth = w3.eth.get_balance(addr)
        print("[Arbitrum]")
        print(f"  connected={w3.is_connected()} chainId={w3.eth.chain_id}")
        print(f"  USDC : {Decimal(bal)/Decimal(10**6)} USDC")
        print(f"  ETH  : {Decimal(eth)/Decimal(10**18)} ETH")
    except Exception as e:
        print(f"[Arbitrum] [WARN] {e}")

    # --- HyperEVM ---
    try:
        hl = Web3(Web3.HTTPProvider(cfg.HL_EVM_RPC_URL,
                  request_kwargs={"timeout": 10, "headers": {"User-Agent": "Mozilla/5.0"}}))
        hl.middleware_onion.inject(geth_poa_middleware, layer=0)
        husdc = hl.eth.contract(address=Web3.to_checksum_address(cfg.HYPEREVM_USDC_ADDRESS), abi=ERC20)
        hbal = husdc.functions.balanceOf(addr).call()
        print("[HyperEVM]")
        print(f"  connected={hl.is_connected()} block={hl.eth.block_number}")
        print(f"  USDC : {Decimal(hbal)/Decimal(10**6)} USDC")
    except Exception as e:
        print(f"[HyperEVM] [WARN] {e}")

    # --- HyperCore spot ---
    try:
        r = requests.post(cfg.HL_INFO_URL, json={"type": "spotClearinghouseState", "user": low}, timeout=10).json()
        print("[HyperCore spot]")
        for b in r.get("balances", []) or []:
            if b.get("coin") in ("USDC",) or Decimal(str(b.get("total") or 0)) > 0:
                print(f"  {b.get('coin'):<6} total={b.get('total')} hold={b.get('hold')}")
    except Exception as e:
        print(f"[HyperCore spot] [WARN] {e}")

    # --- HyperCore perp ---
    try:
        r = requests.post(cfg.HL_INFO_URL, json={"type": "clearinghouseState", "user": low}, timeout=10).json()
        ms = r.get("marginSummary", {}) or {}
        print("[HyperCore perp]")
        print(f"  accountValue={ms.get('accountValue')} withdrawable={r.get('withdrawable')}")
    except Exception as e:
        print(f"[HyperCore perp] [WARN] {e}")

    print("=" * 64)


if __name__ == "__main__":
    main()
