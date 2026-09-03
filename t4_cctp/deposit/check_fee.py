#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_fee.py — CCTP V2 deposit の正しい maxFee を fee API 実値で確定する（送金しない）。

一次ソース（developers.circle.com / cctp-finality-and-fees, forwarding-service）:
  (a) GET /v2/burn/USDC/fees/{src}/{dst}              → finalityThreshold ごとの minimumFee(bps)
  (b) GET /v2/burn/USDC/fees/{src}/{dst}?forward=true → 追加で forwardFee{low,med,high}(atomic)
  - minimumFee の単位は basis points(bps), 1bps=0.01%。base = ceil(amount * bps / 10000)。
  - HyperCore へ自動配送(Forwarding)するには base に forwardFee を加えた額を maxFee が満たす必要。
  - maxFee = ceil( (base + forwardFee) * (1 + buffer) )。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from decimal import Decimal

from t4_cctp.deposit import config as cfg
from t4_cctp.deposit.deposit_cctp_measure import compute_deposit_max_fee, FORWARD_FEE_FALLBACK_ATOMIC


def usdc(atomic):
    return Decimal(atomic) / Decimal(10**6) if atomic is not None else None


def main():
    amount_usdc = Decimal(str(cfg.DEPOSIT_AMOUNT_USDC))
    amount_raw = int(amount_usdc * (10 ** 6))

    print("=" * 72)
    print(" CCTP V2 deposit maxFee 確定（DRY: 送金しない）")
    print("=" * 72)
    print(f" route   : src domain {cfg.ARB_DOMAIN_ID} (Arbitrum) -> dest domain {cfg.HYPEREVM_DOMAIN_ID} (HyperEVM)")
    print(f" amount  : {amount_usdc} USDC ({amount_raw} atomic)")

    try:
        info = compute_deposit_max_fee(amount_raw, buffer_pct=0.05, forward_tier="med")
    except Exception as e:
        print(f"[ERROR] fee API 取得失敗: {e}")
        print("→ maxFee を特定できないため送信に進まない。")
        sys.exit(1)

    print("\n--- 段1a: GET " + info["url_a"] + " ---")
    print(json.dumps(info["raw_a"], indent=2))
    print("\n--- 段1b: GET " + info["url_b"] + " ---")
    print(json.dumps(info["raw_b"], indent=2))

    print("\n" + "-" * 72)
    bps = info["fast_bps"]
    base = info["base_fee_atomic"]
    fwd = info["forward_fee_atomic"]
    mx = info["max_fee_atomic"]
    print(" 段2: maxFee の内訳（積み上げ）")
    print(f"   Fast minimumFee(bps)       : {bps}  (1bps=0.01%)")
    print(f"   ① CCTP基本手数料 base       : {base} atomic ({usdc(base)} USDC) = ceil({amount_raw}*{bps}/10000)")
    print(f"   forwardFee(API low/med/high): {info['forward_fee_obj']}")
    print(f"   ② Forwarding手数料           : {fwd} atomic ({usdc(fwd)} USDC)  [{info['forward_source']}]")
    print(f"   ③ buffer                    : +{int(info['buffer_pct']*100)}%")
    print(f"   => maxFee = ceil(({base}+{fwd})*{1+info['buffer_pct']}) = {mx} atomic ({usdc(mx)} USDC)")
    print("-" * 72)

    # 安全弁: 算出 maxFee が forwardFee[med] を下回ったら送らない
    fwd_med = info["forward_fee_med"]
    if bps is None or base is None or not mx or not fwd_med or mx < fwd_med:
        print(f"[STOP] maxFee={mx} < forwardFee[med]={fwd_med} または取得不可。送信に進まない。")
        sys.exit(2)

    print("[OK] maxFee を確定。送信時:")
    print(f"     maxFee={mx} (atomic, {usdc(mx)} USDC), minFinalityThreshold={cfg.DEPOSIT_MIN_FINALITY_THRESHOLD}")
    print(f"     ※feeExecuted は forwarder が maxFee 満額を徴収する挙動（実測 exp_id=3）。"
          f"HyperCore credit は約 {amount_usdc - usdc(mx)} USDC 見込み。")


if __name__ == "__main__":
    main()
