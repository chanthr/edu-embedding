"""Run every evaluation part and write results/.

Parts: A triplets, B cross-section retrieval, C student-interest queries,
D embedding space health, E recommender ablation, P weight provenance.

All data comes from the caller: the corpus (``--corpus``), triplet files for
part A (``--triplets NAME=PATH``, optional ``--train-triplets NAME=PATH``) and
optional hand labels for part C (``--cluster-overrides``). Per-query top-10
lists contain corpus text, so they are written only to ``--details-dir``.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from ..corpus import load_corpus, resolve_corpus_path
from ..encoder import Encoder, pick_device
from ..index import BM25Index, DenseIndex, build_dense
from ..recommender import Recommender
from ..rerank import CrossEncoderReranker
from . import cross_section, health, interest, provenance, triplets
from .config import ABLATION_MODELS, MODELS, PROVENANCE, RERANK_MODEL, SEED, model_by_key

ALL_PARTS = ("A", "B", "C", "D", "E", "P")


def default_queries_path() -> Path:
    """eval/student_interest_queries.jsonl in the current directory or the repository."""
    candidates = [Path.cwd() / "eval" / "student_interest_queries.jsonl"]
    candidates += [p / "eval" / "student_interest_queries.jsonl" for p in Path(__file__).resolve().parents]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("eval/student_interest_queries.jsonl not found; pass --queries PATH")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "sentence-transformers", "transformers", "rank-bm25", "huggingface-hub", "edu-embedding"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _merge(old: list[dict], new: list[dict]) -> list[dict]:
    """Replace rows of the models that were just run; keep the others."""
    ran = {r["model"] for r in new}
    rows = [r for r in old if r["model"] not in ran] + new
    order = {m.key: i for i, m in enumerate(MODELS)}
    return sorted(rows, key=lambda r: order.get(r["model"], -1))


def _free(obj) -> None:
    del obj
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover
        pass


def run_all(
    out_dir: str | Path = "results",
    *,
    corpus_path: str | Path | None = None,
    triplet_files: dict[str, str] | None = None,
    train_triplet_files: dict[str, str] | None = None,
    cluster_overrides: str | Path | None = None,
    queries_path: str | Path | None = None,
    details_dir: str | Path | None = None,
    device: str | None = None,
    models: list[str] | None = None,
    parts: list[str] | None = None,
    production_npy: str | Path | None = None,
) -> dict:
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    parts = [p.upper() for p in (parts or ALL_PARTS)]
    specs = [model_by_key(k) for k in models] if models else list(MODELS)
    corpus_path = resolve_corpus_path(corpus_path)
    triplet_files = dict(triplet_files or {})
    train_triplet_files = dict(train_triplet_files or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    passages = load_corpus(corpus_path)
    bm25 = BM25Index([p.text for p in passages])
    queries_c = interest.load_queries(queries_path or default_queries_path())
    labels = interest.program_labels(passages, cluster_overrides)
    queries_b = cross_section.build_queries(passages)
    ranker = cross_section.ProgramRanker(passages)

    # Hashes only: they identify the private inputs without revealing them.
    summary["environment"] = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "device": device,
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": _versions(),
        "seed": SEED,
        "corpus_sha256": _sha256(corpus_path),
        "triplet_files_sha256": {n: _sha256(Path(p).expanduser()) for n, p in triplet_files.items()},
        "models": {m.key: {"id": m.model_id, "revision": m.revision, "query_prefix": m.query_prefix} for m in MODELS},
        "rerank_model": {"id": RERANK_MODEL[0], "revision": RERANK_MODEL[1]},
    }

    # ---- triplet files for part A ----
    trip_data: dict[str, list] = {}
    if "A" in parts:
        if not triplet_files:
            print("  [A] skipped: no --triplets NAME=PATH given")
        for name, path in triplet_files.items():
            trip_data[name] = triplets.read_triplets(path)
        overlap = {}
        for name, path in train_triplet_files.items():
            if name in trip_data:
                train = triplets.read_triplets(path)
                overlap[name] = triplets.overlap_report(train, trip_data[name], [p.text for p in passages])
        if overlap:
            summary["data_overlap"] = overlap
            (out / "data_overlap.json").write_text(json.dumps(overlap, indent=2) + "\n")

    reranker = None
    if "C" in parts or "E" in parts:
        reranker = CrossEncoderReranker(RERANK_MODEL[0], device=device)

    res_a, res_b, res_b_sec, res_c, res_d, res_e = [], [], [], [], [], []
    top_lists = []
    if "B" in parts:
        m, ranks = cross_section.evaluate_bm25(bm25, queries_b, ranker)
        res_b.append({"model": "BM25 (no model)", "query": "plain", **m})
        res_b_sec.append({"model": "BM25 (no model)", "query": "plain", **cross_section.per_section(ranks, queries_b)})

    for spec in specs:
        t0 = time.time()
        print(f"[{spec.key}] loading on {device}")
        enc = Encoder(spec.model_id, revision=spec.revision, device=device, query_prefix=spec.query_prefix)
        dense = build_dense(enc, passages, use_cache=True)
        doc_emb = dense.embeddings

        if "A" in parts and spec.triplets:
            for name, rows in trip_data.items():
                r = triplets.evaluate_triplets(enc, rows)
                res_a.append({"model": spec.key, "split": name, **r})
                print(f"  [A] {name}: acc={r['accuracy']:.3f}")

        if "B" in parts:
            for fmt in ("plain", "template"):
                qemb = enc.encode([q[fmt] for q in queries_b], is_query=True)
                m, ranks = cross_section.evaluate_dense(doc_emb, qemb, queries_b, ranker)
                res_b.append({"model": spec.key, "query": fmt, **m})
                res_b_sec.append({"model": spec.key, "query": fmt, **cross_section.per_section(ranks, queries_b)})
                print(f"  [B] {fmt}: R@10={m['recall@10']:.3f} MRR@10={m['mrr@10']:.3f}")

        if "C" in parts:
            for setting, rows in interest.run_dense(enc, doc_emb, passages, queries_c, labels).items():
                res_c.append({"model": spec.key, "setting": setting, **interest.summarise(rows)})
                top_lists += [{"model": spec.key, "setting": setting, **r} for r in rows]
            rec = Recommender(passages, enc, dense=dense, bm25=bm25, reranker=reranker)
            full = interest.ablation_configs()[-1][1]
            rows = interest.run_recommender(rec, queries_c, labels, full)
            res_c.append({"model": spec.key, "setting": "full recommender", **interest.summarise(rows)})
            top_lists += [{"model": spec.key, "setting": "full recommender", **r} for r in rows]
            print(f"  [C] " + ", ".join(f"{r['setting']}: P@10={r['precision@10']:.3f}" for r in res_c if r["model"] == spec.key))

        if "E" in parts and spec.key in ABLATION_MODELS:
            rec = Recommender(passages, enc, dense=dense, bm25=bm25, reranker=reranker)
            for step, cfg in interest.ablation_configs():
                rows = interest.run_recommender(rec, queries_c, labels, cfg)
                res_e.append({"model": spec.key, "step": step, **interest.summarise(rows)})
            print("  [E] done")

        if "D" in parts and spec.health:
            res_d.append({"model": spec.key, **health.space_health(doc_emb)})

        print(f"  {time.time() - t0:.0f}s")
        _free(enc)

    if production_npy:
        emb = np.load(production_npy, allow_pickle=False, mmap_mode="r")
        prod = {
            "model": "Saved production embeddings (old repository, not included)",
            "file_sha256": hashlib.sha256(Path(production_npy).read_bytes()).hexdigest(),
            **health.space_health(np.asarray(emb)),
        }
        (out / "health_production_npy.json").write_text(json.dumps(prod, indent=2) + "\n")

    if "P" in parts:
        prov = {}
        for tuned, base in PROVENANCE:
            a, b = model_by_key(tuned), model_by_key(base)
            prov[f"{tuned} vs {base}"] = provenance.compare((a.model_id, a.revision), (b.model_id, b.revision))
        summary["weight_provenance"] = prov
        (out / "weight_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    for key, rows, fname in (
        ("triplets", res_a, "triplets.csv"),
        ("cross_section", res_b, "cross_section.csv"),
        ("cross_section_by_section_recall@10", res_b_sec, "cross_section_by_section.csv"),
        ("interest", res_c, "interest.csv"),
        ("health", res_d, "health.csv"),
        ("ablation", res_e, "ablation.csv"),
    ):
        if rows:
            summary[key] = _merge(summary.get(key, []), rows)
            _write_csv(out / fname, summary[key])
    if top_lists and details_dir:
        Path(details_dir).expanduser().mkdir(parents=True, exist_ok=True)
        path = Path(details_dir).expanduser() / "interest_top10.jsonl"
        old = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        new = [dict(r, top=[f"{u} | {p}" for u, p in r["top"]]) for r in top_lists]
        with open(path, "w", encoding="utf-8") as fh:
            for r in _merge(old, new):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    from .report import write_report

    write_report(out, summary)
    print(f"wrote {out}/")
    return summary
