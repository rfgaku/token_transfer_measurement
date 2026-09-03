#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sender の USDC残高 / allowance / nonce / ETH残高 を確認する小スクリプト。"""
import os
import sys

# --- 修正0: リポジトリルートを sys.path に追加（PYTHONPATH 無し・任意ディレクトリから実行可能に） ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware

from t4_cctp.deposit import config as cfg

load_dotenv()

ERC20 = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "o", "type": "address"}, {"name": "sp", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]


def main():
    w3 = Web3(Web3.HTTPProvider(cfg.ARB_RPC_URL))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    s = Web3.to_checksum_address(os.environ["ARB_SENDER_ADDRESS"])
    u = w3.eth.contract(address=Web3.to_checksum_address(cfg.ARB_USDC_ADDRESS), abi=ERC20)
    tm = Web3.to_checksum_address(cfg.ARB_TOKEN_MESSENGER_V2)

    print("connected            :", w3.is_connected(), "chainId", w3.eth.chain_id)
    print("sender               :", s)
    print("USDC balance         :", u.functions.balanceOf(s).call() / 1e6, "USDC")
    print("allowance -> TokenMsgr:", u.functions.allowance(s, tm).call() / 1e6, "USDC")
    print("tx count (nonce)     :", w3.eth.get_transaction_count(s))
    print("ETH balance          :", w3.eth.get_balance(s) / 1e18, "ETH")


if __name__ == "__main__":
    main()
