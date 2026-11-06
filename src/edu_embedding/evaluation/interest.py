"""Part C (student-interest queries) and part E (recommender ablation).

Each query has a list of acceptable clusters, written by the repository
author. A program counts as acceptable when one of its clusters is in that
list. Program clusters are every taxonomy cluster whose keywords appear in the
program name. Programs with no keyword match can be labelled by hand in an
optional JSON file (``--cluster-overrides``); without it they never count as
acceptable.

Metrics over the top 10 programs:
  precision@10           acceptable programs / 10 (missing slots count as misses),
                         with a 95% bootstrap interval over queries
  precision_of_returned  acceptable programs / programs returned
  unique_unis@10         distinct universities among the returned programs
  returned               programs returned (the guard can return fewer than 10)
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..corpus import Passage
from ..query import build_query
from ..recommender import Recommender, RecommenderConfig
from ..taxonomy import all_clusters
from .cross_section import ProgramRanker

K = 10

ABLATION_STEPS: list[tuple[str, dict]] = [
    ("Dense only", dict(use_bm25=False, lexical_guard=False, strict_taxonomy=False, per_university_cap=None, mmr=False, rerank=False)),
    ("BM25 only", dict(use_dense=False, lexical_guard=False, strict_taxonomy=False, per_university_cap=None, mmr=False, rerank=False)),
    ("Hybrid (dense + BM25)", dict(lexical_guard=False, strict_taxonomy=False, per_university_cap=None, mmr=False, rerank=False)),
    ("+ lexical guard and taxonomy filter", dict(per_university_cap=None, mmr=False, rerank=False)),
    ("+ per-university cap and MMR", dict(rerank=False)),
    ("+ cross-encoder rerank (production defaults)", dict()),
]


def load_queries(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def program_labels(passages: list[Passage], overrides_path: str | Path | None = None) -> dict[tuple[str, str], set[str]]:
    """Overrides file format: {"labels": {"<university> | <program>": [clusters]}}."""
    overrides = {}
    if overrides_path:
        overrides = json.loads(Path(overrides_path).expanduser().read_text(encoding="utf-8"))["labels"]
    labels = {}
    for p in passages:
        if p.program_key in labels:
            continue
        found = all_clusters(p.program)
        if not found:
            found = set(overrides.get(f"{p.university} | {p.program}", []))
        labels[p.program_key] = found
    return labels


def score_list(programs: list[tuple[str, str]], acceptable: set[str], labels: dict) -> dict:
    top = programs[:K]
    ok = [bool(labels.get(k, set()) & acceptable) for k in top]
    return {
        "precision@10": sum(ok) / K,
        "unique_unis@10": len({u for u, _ in top}),
        "returned": len(top),
        "ok": ok,
    }


def bootstrap_ci(values: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean, resampling queries."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(per_query: list[dict]) -> dict:
    p10 = [q["precision@10"] for q in per_query]
    lo, hi = bootstrap_ci(p10)
    returned = [q for q in per_query if q["returned"]]
    return {
        "n_queries": len(per_query),
        "precision@10": float(np.mean(p10)),
        "precision@10_ci95_low": lo,
        "precision@10_ci95_high": hi,
        "precision_of_returned": float(np.mean([sum(q["ok"]) / q["returned"] for q in returned])) if returned else 0.0,
        "unique_unis@10": float(np.mean([q["unique_unis@10"] for q in per_query])),
        "returned": float(np.mean([q["returned"] for q in per_query])),
    }


def dense_top_programs(encoder, doc_emb: np.ndarray, ranker: ProgramRanker, keys: list, query: str, template: bool) -> list:
    text = build_query([query]) if template else query
    q = encoder.encode([text], is_query=True)
    P = ranker.program_scores(q @ doc_emb.T)[0]
    order = np.argsort(-P, kind="stable")[:K]
    return [keys[j] for j in order]


def run_dense(encoder, doc_emb, passages, queries, labels) -> dict:
    ranker = ProgramRanker(passages)
    keys = [None] * len(ranker.key_index)
    for k, j in ranker.key_index.items():
        keys[j] = k
    out = {}
    for name, template in (("dense, plain query", False), ("dense, production template", True)):
        rows = []
        for q in queries:
            progs = dense_top_programs(encoder, doc_emb, ranker, keys, q["query"], template)
            rows.append({"id": q["id"], "top": progs, **score_list(progs, set(q["acceptable"]), labels)})
        out[name] = rows
    return out


def run_recommender(rec: Recommender, queries, labels, config: RecommenderConfig) -> list[dict]:
    rows = []
    for q in queries:
        res = rec.recommend(q["query"], **asdict(config))
        progs = [(r["university"], r["program"]) for r in res]
        rows.append({"id": q["id"], "top": progs, "mode": res[0]["mode"] if res else None,
                     **score_list(progs, set(q["acceptable"]), labels)})
    return rows


def ablation_configs() -> list[tuple[str, RecommenderConfig]]:
    base = RecommenderConfig(topk=K)
    return [(name, replace(base, **kw)) for name, kw in ABLATION_STEPS]
