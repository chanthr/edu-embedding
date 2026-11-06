"""Optional cross-encoder rerank of the fused passage list (production step)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .fusion import minmax

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str = DEFAULT_RERANK_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 8,
        max_chars: int = 1200,
    ) -> None:
        from sentence_transformers import CrossEncoder

        from .encoder import pick_device

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.model = CrossEncoder(model_id, max_length=512, device=pick_device(device))

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars - 3] + "..."

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        pairs = [(query, self._truncate(t)) for t in texts]
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


def apply_rerank(query: str, rows: list[dict], reranker: CrossEncoderReranker, *, topk: int, weight: float) -> None:
    """Blend the scaled cross-encoder score into ``row['similarity']`` in place."""
    topn = min(topk, len(rows))
    if topn == 0:
        return
    texts = [" ".join(filter(None, [r["passage"].title, r["passage"].text])) for r in rows[:topn]]
    raw = reranker.score(query, texts)
    norm = minmax({i: float(s) for i, s in enumerate(raw)})
    for i, row in enumerate(rows[:topn]):
        r = norm.get(i, 0.0)
        row["similarity"] = (1.0 - weight) * row["similarity"] + weight * r
        row["rerank"] = r
