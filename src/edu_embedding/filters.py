"""Lexical guard, category detection and per-query keyword profiles.

Logic follows production. All term checks use word-boundary matching from
``edu_embedding.text`` instead of substring checks.
"""
from __future__ import annotations

from typing import Iterable

from .profiles import CATEGORY_PROFILES, CATEGORY_TRIGGERS, TITLE_FILTERS
from .query import parse_query_terms
from .text import any_term, count_terms, has_term, normalize, query_term


def lexical_guard(
    normalized_text: str,
    *,
    must: Iterable[str] | None = None,
    ban: Iterable[str] | None = None,
    must_all: Iterable[str] | None = None,
    min_hits: int = 0,
) -> tuple[bool, int]:
    """Return (passed, number of ``must`` terms found)."""
    must_list = [t for t in (must or []) if t]
    hits = count_terms(normalized_text, must_list)
    if min_hits and hits < min_hits:
        return False, hits
    for t in must_all or ():
        if t and not has_term(normalized_text, t):
            return False, hits
    if ban and any_term(normalized_text, ban):
        return False, hits
    if must_list and not hits and not min_hits:
        return False, hits
    return True, hits


def pick_category(query: str) -> str | None:
    norm = normalize(query)
    for category, terms in CATEGORY_TRIGGERS:
        if any_term(norm, terms):
            return category
    return None


def passes_title_filter(category: str | None, title: str) -> bool:
    terms = TITLE_FILTERS.get(category or "")
    if not terms:
        return True
    return any_term(normalize(title), terms)


def refine_profile(query: str, base_profile: dict | None = None, *, mode: str | None = None) -> dict:
    """Merge the category profile with terms taken from the query itself.

    Query words raise ``min_hits``. In keyword mode (or when ``mode`` is None)
    two query words must appear, as in production. In longer queries
    production's substring matching let a function word such as "i" count as
    a hit on every passage, so in practice one content word was enough; with
    word-boundary matching that is now stated explicitly as ``min_hits = 1``.
    """
    base_profile = base_profile or {}
    refined = {
        "must_any": set(base_profile.get("must_any", ())),
        "ban_any": set(base_profile.get("ban_any", ())),
        "bonus_any": set(base_profile.get("bonus_any", ())),
        "must_all": set(base_profile.get("must_all", ())),
        "min_hits": int(base_profile.get("min_hits", 0)),
    }
    q = normalize(query)
    phrases, tokens = parse_query_terms(query)

    if phrases:
        refined["must_any"].update(phrases)
        refined["bonus_any"].update(phrases)
    if tokens:
        refined["must_any"].update(query_term(t) for t in tokens)
        significant = [t for t in tokens if len(t) >= 4]
        if significant:
            needed = min(len(significant), 2) if mode in (None, "keyword") else 1
            refined["min_hits"] = max(refined["min_hits"], needed)

    def says(phrase: str) -> bool:
        return has_term(q, phrase) or phrase in phrases

    if says("quantitative finance") or ("quantitative" in tokens and "finance" in tokens) or says("financial engineering"):
        refined["must_all"].update({"quantitative", "finance"})
        refined["min_hits"] = max(refined["min_hits"], 2)
        refined["bonus_any"].update({"quantitative finance", "financial engineering"})
        refined["ban_any"].update({"sport", "sports", "leisure", "hospitality"})

    if says("financial engineering"):
        refined["must_all"].update({"financial", "engineering"})
        refined["bonus_any"].add("financial engineering")

    if says("risk management"):
        refined["must_all"].update({"risk", "management"})
        refined["min_hits"] = max(refined["min_hits"], 2)

    if says("data science") or ("data" in tokens and "science" in tokens):
        refined["must_any"].update({
            "data", "science", "analytics", "statistic*", "statistics", "machine",
            "learning", "ai", "artificial", "intelligence", "computing",
        })
        refined["min_hits"] = max(refined["min_hits"], 2)
        refined["bonus_any"].update({"data science", "machine learning", "artificial intelligence"})
        refined["ban_any"].update({
            "pharmacy", "biotechnology", "biochemistry", "fashion", "merchandising",
            "hospitality", "leisure", "sport", "sports",
        })
        refined["must_all"].update({"data", "science"})

    if says("computer science") or ("computer" in tokens and "science" in tokens):
        refined["must_any"].update({
            "computer", "computing", "software", "programming", "informatics",
            "cybersecurity", "information technology", "information systems", "ai",
            "artificial intelligence",
        })
        refined["min_hits"] = max(refined["min_hits"], 1)
        refined["bonus_any"].update({"computer science", "software engineering", "cybersecurity"})

    if says("machine learning"):
        refined["must_all"].update({"machine", "learning"})
        refined["bonus_any"].add("machine learning")

    if "software engineering" in phrases:
        refined["must_all"].update({"software", "engineering"})
        refined["bonus_any"].add("software engineering")

    if "bioinformatics" in phrases or "computational biology" in phrases:
        refined["bonus_any"].update({"bioinformatics", "computational biology"})

    return refined


def profile_for_query(keyword_text: str, mode: str | None = None) -> tuple[str | None, dict]:
    category = pick_category(keyword_text)
    return category, refine_profile(keyword_text, CATEGORY_PROFILES.get(category or "", {}), mode=mode)
