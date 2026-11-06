"""Command line interface: ``edu-embedding search | eval | build-index``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

from .encoder import DEFAULT_MODEL


def _quiet() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover
        pass


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id or local path (default: {DEFAULT_MODEL})")
    p.add_argument("--corpus", default=None,
                   help="JSONL corpus file (or set EDU_EMBED_CORPUS). Required; see data/README.md for the format")
    p.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")


def _corpus_or_exit(path: str | None) -> str:
    from .corpus import CorpusNotFoundError, resolve_corpus_path

    try:
        return str(resolve_corpus_path(path))
    except CorpusNotFoundError as exc:
        print(f"edu-embedding: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _pairs(values: list[str] | None, flag: str) -> dict[str, str]:
    out = {}
    for v in values or []:
        name, sep, path = v.partition("=")
        if not sep or not name or not path:
            print(f"edu-embedding: error: {flag} expects NAME=PATH, got {v!r}", file=sys.stderr)
            raise SystemExit(2)
        out[name] = path
    return out


def cmd_search(args: argparse.Namespace) -> int:
    from .recommender import Recommender, RecommenderConfig

    args.corpus = _corpus_or_exit(args.corpus)

    cfg = RecommenderConfig(
        topk=args.topk,
        rerank=not args.no_rerank,
        lexical_guard=not args.no_guard,
        strict_taxonomy=not args.no_guard,
        per_university_cap=args.per_uni_cap,
        mmr=not args.no_mmr,
    )
    rec = Recommender.load(args.model, corpus_path=args.corpus, device=args.device, config=cfg, use_cache=not args.no_cache)
    keywords = args.query if len(args.query) > 1 else args.query[0]
    results = rec.recommend(keywords)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0
    mode = results[0]["mode"] if results else "-"
    print(f"model: {args.model}   query mode: {mode}   results: {len(results)}\n")
    width = max((len(r["university"]) for r in results), default=10)
    for r in results:
        print(f"{r['rank']:>3}. {r['university']:<{width}}  {r['program']}  ({r['score']:.3f})")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from .corpus import load_corpus
    from .encoder import Encoder
    from .index import build_dense, default_cache_dir

    passages = load_corpus(_corpus_or_exit(args.corpus))
    enc = Encoder(args.model, device=args.device)
    idx = build_dense(enc, passages, cache_dir=args.cache_dir, show_progress=True)
    print(f"cached {len(idx)} x {enc.dim} embeddings for {args.model} under {args.cache_dir or default_cache_dir()}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation.runner import run_all

    run_all(
        out_dir=args.out,
        corpus_path=_corpus_or_exit(args.corpus),
        triplet_files=_pairs(args.triplets, "--triplets"),
        train_triplet_files=_pairs(args.train_triplets, "--train-triplets"),
        cluster_overrides=args.cluster_overrides or os.getenv("EDU_EMBED_CLUSTER_OVERRIDES"),
        queries_path=args.queries,
        details_dir=args.details_dir,
        device=args.device,
        models=args.models,
        parts=args.parts,
        production_npy=args.production_npy,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _quiet()
    ap = argparse.ArgumentParser(prog="edu-embedding", description="Match student interests to university programs.")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="Recommend programs for a query")
    s.add_argument("query", nargs="+", help="Free text, or several keywords as separate arguments")
    _add_common(s)
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--per-uni-cap", type=int, default=3, help="Max programs per university (0 = no cap)")
    s.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder rerank")
    s.add_argument("--no-guard", action="store_true", help="Skip the lexical guard and taxonomy filter")
    s.add_argument("--no-mmr", action="store_true", help="Skip MMR diversification")
    s.add_argument("--no-cache", action="store_true", help="Do not read or write the embedding cache")
    s.add_argument("--json", action="store_true", help="Print JSON")
    s.set_defaults(func=cmd_search)

    b = sub.add_parser("build-index", help="Embed the corpus once and cache the matrix")
    _add_common(b)
    b.add_argument("--cache-dir", default=None)
    b.set_defaults(func=cmd_build_index)

    e = sub.add_parser("eval", help="Run the evaluation and write results/")
    e.add_argument("--out", default="results")
    e.add_argument("--corpus", default=None, help="JSONL corpus file (or set EDU_EMBED_CORPUS). Required")
    e.add_argument("--triplets", action="append", metavar="NAME=PATH",
                   help="Validation triplets for part A (JSONL or Parquet, columns sentence_0/1/2). Repeatable")
    e.add_argument("--train-triplets", action="append", metavar="NAME=PATH",
                   help="Training triplets with the same NAME, to measure overlap with the corpus. Repeatable")
    e.add_argument("--cluster-overrides", default=None,
                   help="Optional JSON of hand cluster labels for programs (or EDU_EMBED_CLUSTER_OVERRIDES)")
    e.add_argument("--queries", default=None, help="Student-interest queries (default: eval/student_interest_queries.jsonl)")
    e.add_argument("--details-dir", default=None,
                   help="Where to write per-query top-10 lists. They contain corpus text, so keep this outside the repo")
    e.add_argument("--device", default=None)
    e.add_argument("--models", nargs="*", default=None, help="Subset of model keys (default: all)")
    e.add_argument("--parts", nargs="*", default=None, help="Subset of A B C D E P (default: all)")
    e.add_argument("--production-npy", default=None,
                   help="Optional: the old production corpus_embeddings.npy, to add it to the health table")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
