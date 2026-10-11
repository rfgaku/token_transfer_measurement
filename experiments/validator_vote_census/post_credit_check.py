"""Fetch the 15 blocks after the credit block for the 6 deposits whose credit block had no 4th vote,
and record every VoteEthDepositAction found there (target deposit or not, error or not).

Waits (one probe per hour) until the explorer's daily archived-block quota is free; once fetching
has started, a quota hit stops the run (partial results are kept and reported)."""
import csv, gzip, json, os, sys, time
from datetime import datetime
import pandas as pd
import fetch_votes as fv  # same pacing (2.2 s, 429 -> 10 s), same cache/blocks

HERE = fv.HERE
EXPS = [23, 24, 32, 64, 101, 102]
AFTER = 15
FOUR = {"0x263294039413b96d25e4173a5f7599f8b3801504", "0xda6816df552c3f9e0fb64979fb357800d690d79b",
        "0x58e1b0e63c905d5982324fcd9108582623b8132e", "0xef2364db5db6f5539aa0bc111771a94ee47637fc"}
OUT = os.path.join(HERE, "post_credit_votes.csv")
STATUS = os.path.join(HERE, "logs", "post_credit_status.json")


class Quota(Exception):
    pass


def get(h):
    """One block; raise Quota on the daily-cap message instead of waiting."""
    p = os.path.join(fv.CACHE, f"{h}.json.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt") as f:
            return json.load(f)
    while True:
        wait = fv._last[0] + fv.INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        fv._last[0] = time.time()
        try:
            r = fv._s.post(fv.URL, json={"type": "blockDetails", "height": int(h)}, timeout=60)
            st = r.status_code
        except Exception as e:  # noqa
            st = repr(e)[:80]
        if st == 200:
            if "archived blocks" in r.text:
                raise Quota(r.text[:200])
            try:
                d = r.json()["blockDetails"]
            except Exception:  # noqa
                fv.log("bad 200 body", h, r.text[:200]); time.sleep(fv.WAIT429); continue
            with gzip.open(p + ".tmp", "wt") as f:
                json.dump(d, f)
            os.replace(p + ".tmp", p)
            return d
        fv.log("status", h, st)
        time.sleep(fv.WAIT429)


def main():
    df = pd.read_csv(fv.SRC).set_index("experiment_id")
    cb = pd.read_csv(fv.CREDIT).set_index("experiment_id")
    todo = [(e, int(cb.loc[e, "credit_l1_height"]) + k) for e in EXPS for k in range(1, AFTER + 1)]
    # wait for quota: probe the first uncached block once per hour
    while True:
        first = next(((e, h) for e, h in todo if not os.path.exists(os.path.join(fv.CACHE, f"{h}.json.gz"))), None)
        if first is None:
            break
        try:
            get(first[1]); break
        except Quota as q:
            fv.log("quota still active; retry in 1 h:", q)
            time.sleep(3600)
    fv.log("quota free; fetching", len(todo), "blocks")
    rows, fetched, stopped = [], set(), None
    try:
        for e, h in todo:
            b = get(h); fetched.add((e, h))
            tgt = df.loc[e, "arb_tx_hash"].lower()
            credit_t = int(cb.loc[e, "credit_block_time_ms"])
            for tx in b["txs"]:
                a = tx.get("action") or {}
                if a.get("type") != "VoteEthDepositAction":
                    continue
                eh = str((a.get("ethId") or {}).get("h", "")).lower()
                rows.append({"experiment_id": e, "offset_from_credit": h - int(cb.loc[e, "credit_l1_height"]),
                             "hl_block": h, "hl_block_time_ms": tx["time"], "dt_from_credit_ms": tx["time"] - credit_t,
                             "validator": tx["user"], "is_target_deposit": eh == tgt, "eth_tx_hash": eh,
                             "in_known_4": tx["user"].lower() in FOUR, "tx_error": tx.get("error") or ""})
            fv.log(f"exp {e} +{h - int(cb.loc[e, 'credit_l1_height'])} ok")
    except Quota as q:
        stopped = str(q)
        fv.log("STOP: quota hit mid-run:", q)
    pd.DataFrame(rows, columns=["experiment_id", "offset_from_credit", "hl_block", "hl_block_time_ms",
                                "dt_from_credit_ms", "validator", "is_target_deposit", "eth_tx_hash",
                                "in_known_4", "tx_error"]).to_csv(OUT, index=False)
    json.dump({"blocks_planned": len(todo), "blocks_fetched": len(fetched), "stopped_by_quota": stopped,
               "finished_at": datetime.now().isoformat(timespec="seconds"),
               "per_exp_fetched": {str(e): sum(1 for x, _ in fetched if x == e) for e in EXPS}},
              open(STATUS, "w"), indent=1)
    fv.log("done", len(fetched), "/", len(todo), "votes rows", len(rows))


if __name__ == "__main__":
    main()
