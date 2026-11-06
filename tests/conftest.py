from __future__ import annotations

import hashlib

import numpy as np
import pytest

from edu_embedding.corpus import Passage


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder. No model download."""

    model_id = "fake"
    revision = None
    device = "cpu"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        from edu_embedding.text import tokenize

        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, *, is_query=False, show_progress=False):
        return np.stack([self._vec(t) for t in texts]) if len(texts) else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text):
        return self._vec(text)


def make_passage(uni: str, prog: str, section: str, body: str, cluster: str = "unknown", pid: str | None = None) -> Passage:
    return Passage(
        id=pid or f"{uni}_{prog}_{section}".lower().replace(" ", "_"),
        title=f"{uni} | {prog} | {section}",
        text=f"[UNIV:{uni}][MAJOR:{prog}][LANG:EN][SECTION:{section}] {body}",
        university=uni,
        program=prog,
        section=section,
        language="en",
        cluster=cluster,
    )


@pytest.fixture
def tiny_corpus() -> list[Passage]:
    rows = [
        ("Uni A", "Computer Science BSc", "overview", "Programming, algorithms, software engineering and machine learning."),
        ("Uni A", "Computer Science BSc", "core_modules", "Data structures, operating systems, artificial intelligence."),
        ("Uni A", "Data Science BSc", "overview", "Statistics, data analytics and machine learning for data-driven decisions."),
        ("Uni B", "Economics BSc", "overview", "Microeconomics, macroeconomics and econometrics."),
        ("Uni B", "Accounting and Finance BSc", "overview", "Financial accounting, corporate finance and audit."),
        ("Uni C", "Sustainability Studies BA", "overview", "Sustainability, environmental policy and maintaining ecosystems."),
        ("Uni C", "Web Design BA", "overview", "HTML, CSS and visual design for the web."),
        ("Uni D", "Biology BSc", "overview", "Genetics, ecology and cell biology in the lab."),
        ("Uni D", "Law LLB", "overview", "Contract law, criminal law and legal reasoning."),
    ]
    return [make_passage(*r) for r in rows]


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()
