"""Part P: compare a fine-tuned model's weights with a candidate base model.

For each block (embeddings, each encoder layer, pooler) report the relative
difference ||W_tuned - W_base|| / ||W_base|| over all tensors in the block.
0.0 means the block is bit-identical to the base model.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


def _weights(repo: str, revision: str) -> dict[str, np.ndarray]:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    return load_file(hf_hub_download(repo, "model.safetensors", revision=revision))


def compare(tuned: tuple[str, str], base: tuple[str, str]) -> dict:
    a, b = _weights(*tuned), _weights(*base)
    blocks: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    matched = 0
    for name, w in a.items():
        key = name if name in b else ("bert." + name if "bert." + name in b else name.removeprefix("bert."))
        if key not in b or b[key].shape != w.shape:
            continue
        matched += 1
        m = re.search(r"layer\.(\d+)\.", name)
        block = f"encoder.layer.{int(m.group(1)):02d}" if m else ".".join(name.removeprefix("bert.").split(".")[:2])
        diff = w.astype(np.float64) - b[key]
        blocks[block][0] += float((diff ** 2).sum())
        blocks[block][1] += float((b[key].astype(np.float64) ** 2).sum())
    total_num = sum(v[0] for v in blocks.values())
    total_den = sum(v[1] for v in blocks.values())
    return {
        "tensors_compared": matched,
        "tensors_in_tuned_model": len(a),
        "overall_relative_diff": (total_num / total_den) ** 0.5,
        "relative_diff_by_block": {k: (v[0] / v[1]) ** 0.5 if v[1] else 0.0 for k, v in sorted(blocks.items())},
        "identical_blocks": sorted(k for k, v in blocks.items() if v[0] == 0.0),
    }
