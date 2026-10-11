"""Collect VoteEthDepositAction txs per deposit from the Hyperliquid explorer RPC.

Resumable: processes order.csv from the first experiment not in progress.csv,
stops after TARGET deposits are done and writes done_<TARGET>.flag.

Credit block: deposit_latency.csv's hl_block_number is a HyperEVM block number,
not an L1 height, so the L1 credit block is located as the L1 block whose
blockTime == hl_ledger_time (t3), via interpolation search on blockDetails.
From there we walk down until 16 consecutive blocks contain no vote for the
deposit, never below the height of t1. Blocks after the credit block are not fetched.
"""
import csv, gzip, json, os, sys, time
from datetime import datetime, timezone
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
# result/validator_vote_census_*/ (working tree) or experiments/validator_vote_census/ (public repo)
SRC = next(p for p in (os.path.join(HERE, "..", "deposit_latency.csv"),
                       os.path.join(HERE, "..", "..", "result", "deposit_latency.csv")) if os.path.exists(p))
CACHE = os.path.join(HERE, "cache", "blocks")
ORDER = os.path.join(HERE, "order.csv")
VOTES = os.path.join(HERE, "votes_raw.csv")
PROGRESS = os.path.join(HERE, "progress.csv")
CREDIT = os.path.join(HERE, "credit_blocks.csv")
URL = "https://rpc.hyperliquid.xyz/explorer"
TARGET = int(os.environ.get("TARGET", "30"))
INTERVAL, WAIT429 = 2.2, 10.0
QUOTA_WAIT = 3600.0  # explorer caps archived-block queries per day
GAP_STOP = 16
MAX_LAT_EXP = 77  # latency max (47.3 s); must be within the first 30
os.makedirs(CACHE, exist_ok=True)


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- RPC
_s = requests.Session()
_last = [0.0]
STATS = {"req": 0, "ok": 0, "429": 0, "err": 0}


def fetch(h):
    p = os.path.join(CACHE, f"{h}.json.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt") as f:
            return json.load(f)
    while True:
        wait = _last[0] + INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        STATS["req"] += 1
        try:
            r = _s.post(URL, json={"type": "blockDetails", "height": int(h)}, timeout=60)
            st = r.status_code
        except Exception as e:  # noqa
            st = repr(e)[:80]
        if st == 200:
            try:
                d = r.json()["blockDetails"]
            except Exception:  # noqa  200 without blockDetails: retry like an error
                STATS["err"] += 1
                log("bad 200 body", h, r.text[:200].replace("\n", " "))
                # daily archive quota ("More than 100 archived blocks queried in one day"): probe hourly
                time.sleep(QUOTA_WAIT if "archived blocks" in r.text else WAIT429)
                continue
            STATS["ok"] += 1
            with gzip.open(p + ".tmp", "wt") as f:
                json.dump(d, f)
            os.replace(p + ".tmp", p)
            return d
        if st == 429:
            STATS["429"] += 1
            time.sleep(WAIT429)
        else:
            STATS["err"] += 1
            log("error", h, st)
            time.sleep(WAIT429)


_times = {}


def btime(h):
    if h not in _times:
        _times[h] = fetch(h)["blockTime"]
    return _times[h]


def height_at(t_ms):
    """Smallest height with blockTime >= t_ms, by interpolation then bracketing."""
    known = sorted(_times.items())
    lo = max([kv for kv in known if kv[1] < t_ms], default=(800_000_000, btime(800_000_000)))
    hi = min([kv for kv in known if kv[1] >= t_ms], default=(900_000_000, btime(900_000_000)))
    while hi[0] - lo[0] > 1:
        span = hi[0] - lo[0]
        g = lo[0] + int(round((t_ms - lo[1]) / max(1, hi[1] - lo[1]) * span))
        g = min(max(g, lo[0] + 1), hi[0] - 1)
        if span > 40:  # nudge off the edges so a bracket collapses quickly
            g = min(max(g, lo[0] + 2), hi[0] - 2)
        t = btime(g)
        if t < t_ms:
            lo = (g, t)
        else:
            hi = (g, t)
    return hi[0]


def load_cached_times():
    for f in os.listdir(CACHE):
        if f.endswith(".json.gz"):
            h = int(f.split(".")[0])
            try:
                with gzip.open(os.path.join(CACHE, f), "rt") as fh:
                    _times[h] = json.load(fh)["blockTime"]
            except Exception:  # noqa  (partial file) -> refetch later
                os.remove(os.path.join(CACHE, f))


def votes_in(block, txh):
    out = []
    for tx in block["txs"]:
        a = tx.get("action") or {}
        if a.get("type") == "VoteEthDepositAction" and str((a.get("ethId") or {}).get("h", "")).lower() == txh:
            out.append(tx)
    return out


# ---------------------------------------------------------------- order
def make_order(df):
    df = df.copy()
    df["date"] = df["experiment_start_iso"].str[:10]
    queues = {}
    for d, g in df.sort_values("experiment_id").groupby("date"):
        ids = list(g.experiment_id)
        if MAX_LAT_EXP in ids:  # pull the max-latency deposit to the front of its day
            ids.remove(MAX_LAT_EXP); ids.insert(0, MAX_LAT_EXP)
        queues[d] = ids
    rows, rnd = [], 0
    while any(queues.values()):
        for d in sorted(queues):
            if queues[d]:
                rows.append({"order": len(rows) + 1, "experiment_id": queues[d].pop(0), "date": d, "round": rnd + 1})
        rnd += 1
    o = pd.DataFrame(rows)
    assert MAX_LAT_EXP in set(o.experiment_id[:30])
    o.to_csv(ORDER, index=False)
    return o


def append_csv(path, rows, header):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------- main
def main():
    df = pd.read_csv(SRC)
    order = pd.read_csv(ORDER) if os.path.exists(ORDER) else make_order(df)
    done = set(pd.read_csv(PROGRESS).experiment_id) if os.path.exists(PROGRESS) else set()
    load_cached_times()
    log(f"start: done={len(done)} target={TARGET} cached_blocks={len(_times)}")
    for _, o in order.iterrows():
        if len(done) >= TARGET:
            break
        eid = int(o.experiment_id)
        if eid in done:
            continue
        r = df[df.experiment_id == eid].iloc[0]
        txh = r["arb_tx_hash"].lower()
        t1, t3 = float(r["arb_block_timestamp(ms)"]), float(r["hl_ledger_time(ms)"])
        req0 = STATS["req"]
        hc = height_at(t3)
        if btime(hc) != t3:
            log(f"exp {eid}: WARNING no block with time == t3 (nearest {hc} {btime(hc)} vs {t3})")
        ht1 = height_at(t1)
        found, nblocks, gap, h = [], 0, 0, hc
        while h >= ht1:
            b = fetch(h); nblocks += 1
            _times[h] = b["blockTime"]
            v = votes_in(b, txh)
            if v:
                gap = 0
                for tx in v:
                    found.append({"experiment_id": eid, "arb_tx_hash": txh, "validator": tx["user"],
                                  "hl_block": h, "hl_block_time_ms": tx["time"],
                                  "tx_error": tx.get("error") or ""})
            else:
                gap += 1
                if gap >= GAP_STOP:
                    break
            h -= 1
        stop = "gap16" if gap >= GAP_STOP else "t1_floor"
        append_csv(VOTES, found, ["experiment_id", "arb_tx_hash", "validator", "hl_block", "hl_block_time_ms", "tx_error"])
        append_csv(CREDIT, [{"experiment_id": eid, "credit_l1_height": hc, "credit_block_time_ms": btime(hc),
                             "t1_l1_height": ht1, "lowest_block_scanned": h, "stop_reason": stop,
                             "requests_incl_locate": STATS["req"] - req0}],
                   ["experiment_id", "credit_l1_height", "credit_block_time_ms", "t1_l1_height",
                    "lowest_block_scanned", "stop_reason", "requests_incl_locate"])
        append_csv(PROGRESS, [{"experiment_id": eid, "blocks_fetched": nblocks, "n_votes": len(found),
                               "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}],
                   ["experiment_id", "blocks_fetched", "n_votes", "completed_at"])
        done.add(eid)
        log(f"exp {eid} done ({len(done)}/{TARGET}): credit={hc} votes={len(found)} "
            f"voters={len({x['validator'] for x in found})} blocks={nblocks} stop={stop} "
            f"req={STATS['req'] - req0} stats={STATS}")
    if len(done) >= TARGET:
        open(os.path.join(HERE, f"done_{TARGET}.flag"), "w").write(datetime.now(timezone.utc).isoformat() + "\n")
        log(f"target {TARGET} reached; flag written")


if __name__ == "__main__":
    main()
