"""Part D: embedding space health.

mean_cos_random_pairs   average cosine of random passage pairs (lower means
                        the space is less crowded into one direction)
top_pc_variance_share   share of total variance on the first principal
                        component (close to 1 means the space has collapsed
                        onto one axis)
"""
from __future__ import annotations

import numpy as np

from .config import RANDOM_PAIRS, SEED


def space_health(emb: np.ndarray, *, pairs: int = RANDOM_PAIRS, seed: int = SEED) -> dict:
    x = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), size=pairs)
    j = rng.integers(0, len(x) - 1, size=pairs)
    j = j + (j >= i)  # never pair a row with itself
    cos = (x[i] * x[j]).sum(1)
    centred = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    var = s ** 2
    total = float(var.sum())
    return {
        "dim": int(x.shape[1]),
        "mean_cos_random_pairs": float(cos.mean()),
        "std_cos_random_pairs": float(cos.std()),
        "top_pc_variance_share": float(var[0] / total) if total > 0 else 1.0,
        "top10_pc_variance_share": float(var[:10].sum() / total) if total > 0 else 1.0,
    }
