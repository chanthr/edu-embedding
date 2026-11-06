"""Maximal marginal relevance (MMR) over program candidates.

Standard form: pick argmax  lambda * relevance(i) - (1 - lambda) * max_j sim(i, j)
over the programs already picked. Relevance is the program score min-max scaled
to [0, 1] over the candidate pool, and sim is in [0, 1], so both terms are on
the same scale.

Production added the redundancy term instead of subtracting it, mixed raw
(unbounded) program scores with 0-1 values, and one part of its redundancy
term rewarded programs with a similar mean score.
"""
from __future__ import annotations

from typing import Callable

from .fusion import minmax
from .text import tokenize


def title_tokens(candidate: dict) -> set[str]:
    return set(tokenize(f"{candidate.get('program', '')} {candidate.get('university', '')}"))


def title_overlap(a: dict, b: dict) -> float:
    """Shared title words over the longer title (production measure), in [0, 1]."""
    left, right = title_tokens(a), title_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def mmr(
    candidates: list[dict],
    *,
    lambda_: float = 0.75,
    topk: int = 10,
    similarity: Callable[[dict, dict], float] = title_overlap,
) -> list[dict]:
    if not candidates or topk <= 0:
        return []
    rel = minmax({i: float(c["score"]) for i, c in enumerate(candidates)}, constant=1.0)
    remaining = list(range(len(candidates)))
    selected: list[int] = []
    max_sim = {i: 0.0 for i in remaining}
    while remaining and len(selected) < topk:
        best = max(remaining, key=lambda i: (lambda_ * rel[i] - (1.0 - lambda_) * max_sim[i], -i))
        selected.append(best)
        remaining.remove(best)
        for i in remaining:
            max_sim[i] = max(max_sim[i], similarity(candidates[i], candidates[best]))
    out = []
    for i in selected:
        c = dict(candidates[i])
        c["mmr_relevance"] = rel[i]
        out.append(c)
    return out
