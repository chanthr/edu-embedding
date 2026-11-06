"""Part A: triplet accuracy, cos(anchor, positive) > cos(anchor, negative).

Triplet files are passed in by the caller (``--triplets NAME=PATH``); nothing
is downloaded. A file is JSONL or Parquet with the columns ``sentence_0``
(anchor), ``sentence_1`` (positive) and ``sentence_2`` (negative).

If a matching training file is given (``--train-triplets NAME=PATH``), the
report also measures how much the validation split and the corpus overlap
that training data. Only shares are reported, never counts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..corpus import strip_tags

Triplet = tuple[str, str, str]
COLUMNS = ("sentence_0", "sentence_1", "sentence_2")

# Some training passages still carry chat citation markers that were removed
# from the corpus; drop them before comparing texts.
_MARKER_RE = re.compile(r"\s*:?contentReference(?:\[[^\]]*\]\{[^}]*\})?")


def read_triplets(path: str | Path) -> list[Triplet]:
    path = Path(path).expanduser()
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return [tuple(r) for r in df[list(COLUMNS)].itertuples(index=False, name=None)]
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out.append(tuple(rec[c] for c in COLUMNS))
    return out


def evaluate_triplets(encoder, triplets: list[Triplet]) -> dict:
    texts = sorted({t for tr in triplets for t in tr})
    pos = {t: i for i, t in enumerate(texts)}
    emb = encoder.encode(texts)
    a = emb[[pos[t[0]] for t in triplets]]
    p = emb[[pos[t[1]] for t in triplets]]
    n = emb[[pos[t[2]] for t in triplets]]
    cos_ap = (a * p).sum(1)
    cos_an = (a * n).sum(1)
    return {
        "accuracy": float((cos_ap > cos_an).mean()),
        "mean_cos_pos": float(cos_ap.mean()),
        "mean_cos_neg": float(cos_an.mean()),
        "mean_margin": float((cos_ap - cos_an).mean()),
    }


def _norm_passage(s: str) -> str:
    s = re.sub(r"^\[CLUSTER:[^\]]*\]\s*", "", s)
    s = _MARKER_RE.sub("", s)
    return re.sub(r"\s+", " ", strip_tags(s)).strip()


def overlap_report(train: list[Triplet], val: list[Triplet], corpus_texts: list[str]) -> dict:
    """Shares of the validation split and of the corpus found in the training data."""
    train_set = set(train)
    train_anchors = {t[0] for t in train}
    train_passages = {_norm_passage(x) for t in train for x in t[1:]}
    corpus_norm = [_norm_passage(t) for t in corpus_texts]

    def share(k: int, n: int) -> float:
        return k / n if n else 0.0

    return {
        "validation_triplets_also_in_training": share(sum(t in train_set for t in val), len(val)),
        "validation_anchors_also_in_training": share(sum(t[0] in train_anchors for t in val), len(val)),
        "validation_positives_seen_as_training_passages": share(
            sum(_norm_passage(t[1]) in train_passages for t in val), len(val)),
        "corpus_passages_seen_as_training_passages": share(
            sum(c in train_passages for c in corpus_norm), len(corpus_norm)),
    }
