"""Models and fixed settings for the evaluation. Data paths come from the CLI."""
from __future__ import annotations

from dataclasses import dataclass

SEED = 0
RANDOM_PAIRS = 20_000
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    query_prefix: str = ""
    note: str = ""
    triplets: bool = True   # part A
    health: bool = True     # part D


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("edu-v1", "ongilLabs/edu-v1", "9794538567f6f0ba5ea5b7637d01e06c7ab024d0",
              note="fine-tuned (this repo's default)"),
    ModelSpec("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
              note="baseline"),
    ModelSpec("bge-base-en-v1.5 +instr", "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
              query_prefix=BGE_QUERY_INSTRUCTION, note="baseline, BGE query instruction on queries only",
              triplets=False, health=False),
    ModelSpec("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
              note="baseline"),
    ModelSpec("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
              note="baseline"),
)

RERANK_MODEL = ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")

# Weight comparison (part P): fine-tuned model -> candidate base model.
PROVENANCE = (("edu-v1", "bge-base-en-v1.5"),)

ABLATION_MODELS = ("edu-v1", "bge-base-en-v1.5")


def model_by_key(key: str) -> ModelSpec:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(key)
