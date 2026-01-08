# sim/utils.py
from __future__ import annotations

import math
from typing import Iterable, List


def safe_mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


def pctl(xs: List[float], p: float) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return float("nan")
    vals.sort()
    if p <= 0:
        return float(vals[0])
    if p >= 100:
        return float(vals[-1])
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    d0 = vals[int(f)] * (c - k)
    d1 = vals[int(c)] * (k - f)
    return float(d0 + d1)
