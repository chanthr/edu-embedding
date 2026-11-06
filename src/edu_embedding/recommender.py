"""End-to-end program recommendation.

Pipeline (same order as production):

1. Analyse the query: mode (keyword, hybrid, sentence, multi) and fusion weights.
2. Dense search with the production query template, BM25 search on query terms.
3. Min-max scale both and fuse with the mode weights.
4. Optional cross-encoder rerank of the top passages.
5. Lexical guard and category title filter.
6. Group passages into programs with section weights, bonuses and the strict
   taxonomy filter.
7. Cap programs per university, then diversify with MMR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from .aggregate import aggregate_programs, cap_per_university
from .corpus import Passage, load_corpus
from .diversify import mmr
from .filters import lexical_guard, passes_title_filter, profile_for_query
from .fusion import fuse
from .index import BM25Index, DenseIndex, build_dense
from .query import analyse_query, analysis_text, as_keywords, build_query
from .rerank import DEFAULT_RERANK_MODEL, CrossEncoderReranker, apply_rerank
from .taxonomy import guess_cluster
from .text import normalize, tokenize


@dataclass(frozen=True)
class RecommenderConfig:
    topk: int = 25
    k_passages: int = 200
    dense_topk: int = 200
    bm25_topk: int = 200
    use_dense: bool = True
    use_bm25: bool = True
    query_template: bool = True
    rerank: bool = True
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_weight: float = 0.6
    rerank_topk: int = 120
    lexical_guard: bool = True
    strict_taxonomy: bool = True
    per_university_cap: int | None = 3
    mmr: bool = True
    mmr_lambda: float = 0.75


def _guard_text(p: Passage) -> str:
    # Production used title + text + json.dumps(meta). The JSON key names
    # ("language", "program_name", ...) then counted as matches, so only the
    # meta values are kept here.
    return normalize(" ".join([p.title, p.text, p.university, p.program, p.section, p.language, p.cluster]))


class Recommender:
    def __init__(
        self,
        passages: Sequence[Passage],
        encoder=None,
        *,
        config: RecommenderConfig | None = None,
        dense: DenseIndex | None = None,
        bm25: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.passages = list(passages)
        self.encoder = encoder
        self.config = config or RecommenderConfig()
        self.bm25 = bm25 or BM25Index([p.text for p in self.passages])
        if dense is None and encoder is not None:
            dense = build_dense(encoder, self.passages)
        self.dense = dense
        self._reranker = reranker
        self._guard_texts: list[str] | None = None

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        corpus_path: str | Path | None = None,
        device: str | None = None,
        config: RecommenderConfig | None = None,
        use_cache: bool = True,
    ) -> "Recommender":
        from .encoder import DEFAULT_MODEL, Encoder

        passages = load_corpus(corpus_path)
        encoder = Encoder(model_id or DEFAULT_MODEL, device=device)
        dense = build_dense(encoder, passages, use_cache=use_cache)
        return cls(passages, encoder, config=config, dense=dense)

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            device = getattr(self.encoder, "device", None)
            self._reranker = CrossEncoderReranker(self.config.rerank_model, device=device)
        return self._reranker

    def _guard_text(self, idx: int) -> str:
        if self._guard_texts is None:
            self._guard_texts = [_guard_text(p) for p in self.passages]
        return self._guard_texts[idx]

    # ------------------------------------------------------------------
    def retrieve(self, query: str | Sequence[str], config: RecommenderConfig | None = None) -> tuple[object, list[dict]]:
        """Steps 1 to 4. Returns (plan, passage rows sorted by similarity)."""
        cfg = config or self.config
        keywords = as_keywords(query)
        keyword_text = " ".join(keywords).strip()
        plan = analyse_query(
            analysis_text(keywords) or "program recommendations",
            cfg.k_passages,
            dense_topk=cfg.dense_topk,
            bm25_topk=cfg.bm25_topk,
            rerank_enabled=cfg.rerank,
            rerank_weight=cfg.rerank_weight,
            rerank_topk=cfg.rerank_topk,
        )

        dense_scores: dict[int, float] = {}
        use_dense = cfg.use_dense and self.dense is not None and self.encoder is not None
        if use_dense:
            qtext = build_query(keywords) if cfg.query_template else ", ".join(keywords)
            qvec = self.encoder.encode_query(qtext)
            all_dense = self.dense.scores(qvec)
            for idx, score in self.dense.search(qvec, plan.dense_topk):
                dense_scores[idx] = score

        bm25_scores: dict[int, float] = {}
        if cfg.use_bm25:
            tokens = plan.bm25_tokens or tokenize(keyword_text)
            for idx, score in self.bm25.search(tokens, plan.bm25_topk):
                bm25_scores[idx] = score
            if plan.mode == "multi":
                per_segment_k = max(50, plan.bm25_topk // max(1, len(plan.segments)))
                for seg_tokens in plan.segment_tokens:
                    if not seg_tokens:
                        continue
                    for idx, score in self.bm25.search(seg_tokens, per_segment_k):
                        if score > bm25_scores.get(idx, 0.0):
                            bm25_scores[idx] = score

        if use_dense:
            for idx in bm25_scores:
                if idx not in dense_scores:
                    dense_scores[idx] = float(all_dense[idx])

        if use_dense and cfg.use_bm25:
            dw, bw = plan.dense_weight, plan.bm25_weight
        elif use_dense:
            dw, bw = 1.0, 0.0
        else:
            dw, bw = 0.0, 1.0
        fused = fuse(dense_scores, bm25_scores, dense_weight=dw, bm25_weight=bw)

        rows = [
            {
                "idx": idx,
                "passage": self.passages[idx],
                "similarity": f,
                "dense_raw": dense_scores.get(idx),
                "bm25_raw": bm25_scores.get(idx),
                "dense_norm": d,
                "bm25_norm": b,
            }
            for idx, (f, d, b) in fused.items()
        ]
        rows.sort(key=lambda r: (-r["similarity"], r["idx"]))
        rows = rows[: max(cfg.k_passages, plan.dense_topk, plan.bm25_topk)]

        if plan.rerank and rows:
            apply_rerank(keyword_text or plan.text, rows, self.reranker, topk=plan.rerank_topk, weight=plan.rerank_weight)
            rows.sort(key=lambda r: (-r["similarity"], r["idx"]))
        return plan, rows

    def recommend(self, query: str | Sequence[str], **overrides) -> list[dict]:
        """Recommend programs. Keyword arguments override ``RecommenderConfig`` fields."""
        cfg = replace(self.config, **overrides) if overrides else self.config
        keywords = as_keywords(query)
        keyword_text = " ".join(keywords).strip()
        plan, rows = self.retrieve(keywords, cfg)

        category, profile = profile_for_query(keyword_text, plan.mode)
        filtered = []
        for row in rows:
            passed, hits = lexical_guard(
                self._guard_text(row["idx"]),
                must=profile.get("must_any"),
                ban=profile.get("ban_any"),
                must_all=profile.get("must_all"),
                min_hits=profile.get("min_hits", 0),
            )
            row["lex_hits"] = hits
            if cfg.lexical_guard:
                if not passed or not passes_title_filter(category, row["passage"].title):
                    continue
            filtered.append(row)
        if not filtered:
            limit = min(max(cfg.topk, min(cfg.k_passages, len(rows))), len(rows))
            filtered = rows[:limit]
            for row in filtered:
                row["lex_hits"] = 0

        agg = dict(profile=profile, category=category, target_cluster=guess_cluster(keyword_text))
        candidates = aggregate_programs(filtered, strict_taxonomy=cfg.strict_taxonomy, **agg)
        if not candidates and cfg.strict_taxonomy:
            # New fallback: production returned nothing when the taxonomy
            # filter removed every program that passed the guard.
            candidates = aggregate_programs(filtered, strict_taxonomy=False, **agg)
        candidates = cap_per_university(candidates, cfg.per_university_cap)
        if cfg.mmr:
            picked = mmr(candidates, lambda_=cfg.mmr_lambda, topk=cfg.topk)
        else:
            picked = candidates[: cfg.topk]

        return [
            {
                "rank": i + 1,
                "university": c["university"],
                "program": c["program"],
                "score": round(c["score"], 4),
                "evidence": c["samples"],
                "mode": plan.mode,
            }
            for i, c in enumerate(picked)
        ]

    def describe(self) -> dict:
        return {
            "model": getattr(self.encoder, "model_id", None),
            "device": getattr(self.encoder, "device", None),
            "passages": len(self.passages),
            "config": asdict(self.config),
        }
