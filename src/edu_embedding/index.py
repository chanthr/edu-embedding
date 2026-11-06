"""Dense (numpy) and BM25 indexes over the passage corpus.

For a corpus of a few thousand passages a brute-force dot product is fast and
exact. Both indexes are built at runtime; ``build_dense`` can cache the
embedding matrix on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .text import tokenize

if TYPE_CHECKING:  # pragma: no cover
    from .corpus import Passage
    from .encoder import Encoder

# Production BM25 parameters (read from the production index file).
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25


def _top(scores: np.ndarray, topk: int, *, positive_only: bool = False) -> list[tuple[int, float]]:
    n = scores.shape[0]
    topk = max(0, min(topk, n))
    if topk == 0:
        return []
    # Full stable sort: ties are broken by corpus order, so runs are repeatable.
    idx = np.argsort(-scores, kind="stable")[:topk]
    return [(int(i), float(scores[i])) for i in idx if not positive_only or scores[i] > 0.0]


class BM25Index:
    """BM25 over passage text, tokenised with the same function as queries."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.doc_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.doc_tokens, k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)

    def __len__(self) -> int:
        return len(self.doc_tokens)

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        if not query_tokens:
            return np.zeros(len(self), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(list(query_tokens)), dtype=np.float32)

    def search(self, query_tokens: Sequence[str], topk: int) -> list[tuple[int, float]]:
        scores = self.scores(query_tokens)
        if not scores.size or np.all(scores <= 0.0):
            return []
        return _top(scores, topk, positive_only=True)


class DenseIndex:
    """Row-normalised embedding matrix with cosine (dot product) search."""

    def __init__(self, embeddings: np.ndarray) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        return self.embeddings @ q

    def search(self, query_vec: np.ndarray, topk: int) -> list[tuple[int, float]]:
        return _top(self.scores(query_vec), topk)


def default_cache_dir() -> Path:
    return Path(os.getenv("EDU_EMBED_CACHE", Path.home() / ".cache" / "edu-embedding"))


def _cache_key(encoder: "Encoder", texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([encoder.model_id, encoder.revision or ""]).encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:24]


def build_dense(
    encoder: "Encoder",
    passages: Sequence["Passage"],
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    show_progress: bool = False,
) -> DenseIndex:
    """Embed passage texts (tags included, as in production)."""
    texts = [p.text for p in passages]
    path = None
    if use_cache:
        cdir = Path(cache_dir) if cache_dir else default_cache_dir()
        slug = encoder.model_id.replace("/", "__")
        path = cdir / slug / f"{_cache_key(encoder, texts)}.npy"
        if path.exists():
            emb = np.load(path, allow_pickle=False)
            if emb.shape == (len(texts), encoder.dim):
                return DenseIndex(emb)
    emb = encoder.encode(texts, show_progress=show_progress)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        os.replace(tmp, path)
    return DenseIndex(emb)
