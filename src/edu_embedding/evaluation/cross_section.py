"""Part B: cross-section program retrieval on the corpus.

For every program with exactly one passage per section (5 passages), each
passage becomes a query: tags stripped, university and program names removed
where they appear verbatim. The target is the same program. The query's own
passage is removed from the index; every other passage stays. Programs are
ranked by their best passage score (max pooling).
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from ..corpus import SECTIONS, Passage
from ..query import build_query
from ..text import tokenize

_DEGREE_WORDS = (
    r"BSc|BA|BEng|MEng|MSci|MSc|LLB|BBA|MPhys|MMath|MChem|MNurs|BNurs|MPharm|MBBS|BDS|"
    r"Hons|MEarthSci|MEarthSc|BASc|BSocSc|BSSc|BAS|BS|BArch|BFA|BMus|BEd|BCom|MBChB|BEcon|"
    r"Joint|Programme|Program|Double Degree"
)
_BACHELOR_OF = (
    r"Bachelor of (?:Science|Arts|Engineering|Social Sciences?|Business Administration|"
    r"Laws|Education|Nursing|Medicine|Pharmacy|Music|Fine Arts|Architecture)(?: in)?"
)


def core_program_name(program: str) -> str:
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", program)
    s = re.sub(_BACHELOR_OF, " ", s, flags=re.I)
    s = re.sub(rf"\b(?:{_DEGREE_WORDS})\b", " ", s)
    s = re.sub(r"^\s*(?:in|of)\b", " ", s.strip(), flags=re.I)
    s = re.sub(r"[\s/,&\-]+$|^[\s/,&\-]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _names_to_remove(p: Passage) -> list[str]:
    names = {p.university, p.program}
    if p.university.startswith("The "):
        names.add(p.university[4:])
    m = re.search(r"\(([A-Z]{2,})\)", p.university)
    if m:
        names.add(m.group(1))
        names.add(re.sub(r"\s*\([^)]*\)", "", p.university))
    core = core_program_name(p.program)
    if len(core.split()) >= 2:
        names.add(core)
    return sorted((n for n in names if n), key=len, reverse=True)


def remove_names(p: Passage) -> tuple[str, bool]:
    text = p.body
    removed = False
    for name in _names_to_remove(p):
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I)
        text, n = pattern.subn(" ", text)
        removed = removed or n > 0
    return re.sub(r"\s+", " ", text).strip(), removed


def build_queries(passages: list[Passage]) -> list[dict]:
    by_prog: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, p in enumerate(passages):
        by_prog[p.program_key].append(i)
    queries = []
    for key, idxs in by_prog.items():
        secs = [passages[i].section for i in idxs]
        if len(idxs) != len(SECTIONS) or set(secs) != set(SECTIONS):
            continue
        for i in idxs:
            text, removed = remove_names(passages[i])
            queries.append({
                "doc_idx": i,
                "program": key,
                "section": passages[i].section,
                "plain": text,
                "template": build_query([text]),
                "names_removed": removed,
            })
    return queries


class ProgramRanker:
    """Max-pool passage scores into program scores."""

    def __init__(self, passages: list[Passage]) -> None:
        keys = sorted({p.program_key for p in passages})
        self.key_index = {k: j for j, k in enumerate(keys)}
        self.doc_prog = np.array([self.key_index[p.program_key] for p in passages])
        self.cols = [np.where(self.doc_prog == j)[0] for j in range(len(keys))]

    def program_scores(self, S: np.ndarray) -> np.ndarray:
        out = np.empty((S.shape[0], len(self.cols)), dtype=np.float32)
        for j, cols in enumerate(self.cols):
            out[:, j] = S[:, cols].max(axis=1)
        return out


def ranks_of_targets(P: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each target; ties broken by program order."""
    t = P[np.arange(len(targets)), targets][:, None]
    greater = (P > t).sum(1)
    order = np.arange(P.shape[1])[None, :]
    ties_before = ((P == t) & (order < targets[:, None])).sum(1)
    return 1 + greater + ties_before


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    return {
        "n_queries": int(len(ranks)),
        "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()),
        "recall@10": float((ranks <= 10).mean()),
        "mrr@10": float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean()),
    }


def evaluate_dense(doc_emb: np.ndarray, query_emb: np.ndarray, queries: list[dict], ranker: ProgramRanker) -> tuple[dict, np.ndarray]:
    S = query_emb @ doc_emb.T
    S[np.arange(len(queries)), [q["doc_idx"] for q in queries]] = -np.inf
    P = ranker.program_scores(S)
    targets = np.array([ranker.key_index[q["program"]] for q in queries])
    ranks = ranks_of_targets(P, targets)
    return metrics_from_ranks(ranks), ranks


def evaluate_bm25(bm25, queries: list[dict], ranker: ProgramRanker) -> tuple[dict, np.ndarray]:
    S = np.stack([bm25.scores(tokenize(q["plain"])) for q in queries]).astype(np.float32)
    S[np.arange(len(queries)), [q["doc_idx"] for q in queries]] = -np.inf
    P = ranker.program_scores(S)
    targets = np.array([ranker.key_index[q["program"]] for q in queries])
    ranks = ranks_of_targets(P, targets)
    return metrics_from_ranks(ranks), ranks


def per_section(ranks: np.ndarray, queries: list[dict]) -> dict:
    out = {}
    for s in SECTIONS:
        mask = np.array([q["section"] == s for q in queries])
        out[s] = float((ranks[mask] <= 10).mean()) if mask.any() else None
    return out
