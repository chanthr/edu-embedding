"""Query parsing, query mode analysis and the production query template."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .profiles import EXTRA_QUERY_STOPWORDS, PHRASES, QUERY_STOPWORDS, SYNONYMS
from .taxonomy import guess_cluster
from .text import has_term, normalize, tokenize

ALL_STOPWORDS = QUERY_STOPWORDS | EXTRA_QUERY_STOPWORDS

# Fusion weights by mode: (bm25_weight, dense_weight). Production values.
MODE_WEIGHTS: dict[str, tuple[float, float]] = {
    "keyword": (0.65, 0.35),
    "hybrid": (0.5, 0.5),
    "sentence": (0.35, 0.65),
}


@dataclass
class QueryPlan:
    text: str
    mode: str
    tokens: list[str]          # guard terms: stopwords removed, synonyms added
    bm25_tokens: list[str]     # single words for BM25
    phrases: list[str]
    segments: list[str]
    bm25_weight: float
    dense_weight: float
    bm25_topk: int
    dense_topk: int
    rerank: bool
    rerank_weight: float
    rerank_topk: int
    segment_tokens: list[list[str]] = field(default_factory=list)


def extract_phrases(text: str) -> list[str]:
    norm = normalize(text)
    return [p for p in PHRASES if has_term(norm, p)]


def parse_query_terms(text: str, *, stopwords: set[str] = ALL_STOPWORDS) -> tuple[list[str], list[str]]:
    """Return (phrases, tokens). Tokens may include multi-word synonyms."""
    phrases = extract_phrases(text)
    base = [t for t in tokenize(text) if t not in stopwords]
    tokens = list(base)
    for t in base:
        tokens.extend(SYNONYMS.get(t, ()))
    return phrases, tokens


def to_bm25_tokens(tokens: Sequence[str]) -> list[str]:
    """Split multi-word synonyms ("computer science") into single words."""
    out: list[str] = []
    for t in tokens:
        out.extend(tokenize(t))
    return out


def split_segments(text: str) -> list[str]:
    parts = re.split(r"[;,]|(?:\band\b)|(?:\bor\b)", text, flags=re.IGNORECASE)
    segments = [s.strip() for s in parts if s and s.strip()]
    return segments or [text.strip()]


def analyse_query(
    text: str,
    base_k_passages: int = 200,
    *,
    dense_topk: int = 200,
    bm25_topk: int = 200,
    rerank_enabled: bool = True,
    rerank_weight: float = 0.6,
    rerank_topk: int = 120,
) -> QueryPlan:
    """Pick the query mode and fusion weights (production thresholds).

    The mode is decided on the production token count (production stopwords
    only), so the thresholds behave as before. Guard and BM25 terms use the
    extended stopword list.
    """
    lowered = text.lower().strip()
    _, mode_tokens = parse_query_terms(lowered, stopwords=QUERY_STOPWORDS)
    phrases, tokens = parse_query_terms(lowered)
    n = len(mode_tokens)
    has_sentence_punct = bool(re.search(r"[.?!]", lowered))
    has_clause_sep = bool(re.search(r"[;,]", lowered))
    segments = split_segments(text)

    if n <= 4 and not has_sentence_punct and not has_clause_sep:
        mode, rerank, rw, rk = "keyword", False, rerank_weight, min(rerank_topk, base_k_passages)
    elif n >= 10 or has_sentence_punct:
        mode, rerank, rw, rk = "sentence", True, max(0.55, rerank_weight), max(base_k_passages, rerank_topk)
    else:
        mode, rerank, rw, rk = "hybrid", True, rerank_weight, max(base_k_passages, rerank_topk)
    bm25_w, dense_w = MODE_WEIGHTS[mode]

    if has_clause_sep and mode != "sentence":
        mode = "multi"
        bm25_w = max(bm25_w, 0.55)
        dense_w = 1.0 - bm25_w
        rerank = True

    return QueryPlan(
        text=text,
        mode=mode,
        tokens=tokens,
        bm25_tokens=to_bm25_tokens(tokens),
        phrases=phrases,
        segments=segments,
        bm25_weight=bm25_w,
        dense_weight=dense_w,
        bm25_topk=max(base_k_passages, bm25_topk),
        dense_topk=max(base_k_passages, dense_topk),
        rerank=rerank and rerank_enabled,
        rerank_weight=rw,
        rerank_topk=rk,
        segment_tokens=[to_bm25_tokens(parse_query_terms(s.lower())[1]) for s in segments],
    )


def as_keywords(query: str | Sequence[str]) -> list[str]:
    if isinstance(query, str):
        return [query.strip()] if query.strip() else []
    return [k.strip() for k in query if k and k.strip()]


def analysis_text(keywords: Sequence[str]) -> str:
    """Keywords joined with commas, so several keywords can reach "multi" mode.

    Production joined them with spaces before the mode check, which made the
    "multi" branch unreachable from the recommendation entry point.
    """
    return ", ".join(keywords)


def build_query(keywords: Sequence[str], section_hint: str = "overview") -> str:
    """Production query template sent to the embedding model."""
    joined = ", ".join(keywords)
    cluster = guess_cluster(joined) or "unknown"
    return f"[CLUSTER:{cluster}][SECTION:{section_hint}] Recommend undergraduate majors matching: {joined}."
