"""Hyperliquid explorer RPC helper with on-disk JSON cache and exponential backoff."""
import json, os, time, random, gzip
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "blocks")
os.makedirs(CACHE, exist_ok=True)
URL = "https://rpc.hyperliquid.xyz/explorer"
_s = requests.Session()


MIN_INTERVAL = float(os.environ.get("HL_MIN_INTERVAL", "6"))  # explorer RPC allows only a few req/min per IP
_last = [0.0]
STATS = {"ok": 0, "429": 0, "err": 0}


def _post(body, tries=12):
    err = None
    for i in range(tries):
        wait = _last[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = _s.post(URL, json=body, timeout=60)
            if r.status_code == 200:
                STATS["ok"] += 1
                return r.json()
            err = f"HTTP {r.status_code}"
            STATS["429" if r.status_code == 429 else "err"] += 1
        except Exception as e:  # noqa
            err = repr(e)
            STATS["err"] += 1
        time.sleep(min(120, 5 * 2 ** i) + random.random())  # exponential backoff
    raise RuntimeError(f"failed {body}: {err}")


def block(h):
    """Full blockDetails (cached gzipped JSON)."""
    p = os.path.join(CACHE, f"{h}.json.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt") as f:
            return json.load(f)
    d = _post({"type": "blockDetails", "height": int(h)})["blockDetails"]
    tmp = p + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(d, f)
    os.replace(tmp, p)
    return d


_tcache = {}
TPATH = os.path.join(HERE, "cache", "block_times.json")
if os.path.exists(TPATH):
    _tcache = {int(k): v for k, v in json.load(open(TPATH)).items()}


def block_time(h):
    if h not in _tcache:
        _tcache[h] = block(h)["blockTime"]
    return _tcache[h]


def save_times():
    json.dump(_tcache, open(TPATH + ".tmp", "w"))
    os.replace(TPATH + ".tmp", TPATH)


def height_at(t_ms, lo=800_000_000, hi=900_000_000):
    """Smallest height with blockTime >= t_ms (binary search, uses lightweight time cache)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if _blocktime_light(mid) < t_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _blocktime_light(h):
    # time-only lookups for binary search: cache only the time, not the full block
    if h in _tcache:
        return _tcache[h]
    p = os.path.join(CACHE, f"{h}.json.gz")
    if os.path.exists(p):
        return block_time(h)
    d = _post({"type": "blockDetails", "height": int(h)})["blockDetails"]
    _tcache[h] = d["blockTime"]
    return _tcache[h]


def vote_hash(tx):
    a = tx.get("action") or {}
    if a.get("type") != "VoteEthDepositAction":
        return None
    for k in ("ethTxHash", "tx_hash", "txHash"):
        if k in a:
            return a[k]
    e = a.get("ethId") or {}
    for k in ("tx_hash", "txHash", "ethTxHash"):
        if k in e:
            return e[k]
    return None
