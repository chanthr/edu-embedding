"""Score normalisation and dense/BM25 fusion."""
from __future__ import annotations

import math
from typing import Mapping


def minmax(scores: Mapping, *, constant: float | None = None) -> dict:
    """Min-max scale to [0, 1].

    When all values are equal the production rule is kept: 1.0 if the value is
    positive, else 0.0 (``constant`` overrides this).
    """
    if not scores:
        return {}
    values = list(scores.values())
    hi, lo = max(values), min(values)
    if math.isclose(hi, lo):
        base = constant if constant is not None else (1.0 if hi > 0 else 0.0)
        return {k: base for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def fuse(
    dense: Mapping[int, float],
    bm25: Mapping[int, float],
    *,
    dense_weight: float,
    bm25_weight: float,
) -> dict[int, tuple[float, float, float]]:
    """Weighted sum of min-max scaled scores over the union of candidates.

    Returns {index: (fused, dense_norm, bm25_norm)}. A candidate missing from
    one list gets 0 for that part (production rule).
    """
    dn = minmax(dense)
    bn = minmax(bm25)
    out = {}
    for idx in set(dense) | set(bm25):
        d = dn.get(idx, 0.0)
        b = bn.get(idx, 0.0)
        out[idx] = (dense_weight * d + bm25_weight * b, d, b)
    return out
