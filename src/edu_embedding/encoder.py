"""Sentence embedding wrapper around sentence-transformers."""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

DEFAULT_MODEL = "ongilLabs/edu-v1"


def pick_device(preferred: str | None = None) -> str:
    """Use the requested device, else Apple MPS, else CUDA, else CPU."""
    if preferred:
        return preferred
    env = os.getenv("EDU_EMBED_DEVICE")
    if env:
        return env
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder:
    """Encodes text into L2-normalised float32 vectors.

    ``query_prefix`` is prepended to queries only (for models that expect an
    instruction, such as the optional BGE retrieval prompt).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        revision: str | None = None,
        query_prefix: str = "",
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.revision = revision
        self.device = pick_device(device)
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, device=self.device, revision=revision)

    @property
    def dim(self) -> int:
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(get_dim())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def encode(self, texts: Sequence[str], *, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """One query vector. Empty input gives a zero vector of the model's size."""
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        return self.encode([text], is_query=True)[0]
