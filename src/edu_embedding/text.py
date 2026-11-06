"""Shared text handling: one tokenizer and one term matcher for the whole package.

Production used two tokenizers (``str.split`` for BM25 documents and a regex
for queries) and plain substring checks for keyword rules. Both caused silent
misses and false positives. Everything here goes through ``tokenize``.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    """Lowercase and split on anything that is not a letter or digit."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def normalize(text: str | None) -> str:
    """Token stream joined by single spaces. Term matching runs on this form."""
    return " ".join(tokenize(text))


@lru_cache(maxsize=8192)
def _term_pattern(term: str) -> re.Pattern[str] | None:
    stem = term.endswith("*")
    toks = tokenize(term.rstrip("*"))
    if not toks:
        return None
    body = " ".join(re.escape(t) for t in toks)
    if stem:
        # "econom*" matches "economics", "economy", "econometrics".
        tail = r"[a-z0-9]*"
    else:
        # Whole word or phrase, with an optional plural ending.
        tail = r"(?:s|es)?(?![a-z0-9])"
    return re.compile(r"(?<![a-z0-9])" + body + tail)


def query_term(token: str) -> str:
    """Turn a word the user typed into a match term.

    Words of four or more letters match as word prefixes, so "math" finds
    "mathematics" and "computers" (plural dropped) finds "computer". Shorter
    words ("ai", "ml", "art") must match whole words.
    """
    if " " in token or len(token) < 4:
        return token
    if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token + "*"


def has_term(normalized_text: str, term: str) -> bool:
    """True if ``term`` occurs in ``normalized_text`` at word boundaries.

    A plain term must match whole words (a plural "s" or "es" is allowed). A
    term ending in "*" is a stem and matches the start of a word.
    ``normalized_text`` must come from :func:`normalize`.
    """
    pattern = _term_pattern(term)
    return bool(pattern and pattern.search(normalized_text))


def count_terms(normalized_text: str, terms: Iterable[str]) -> int:
    return sum(1 for t in terms if t and has_term(normalized_text, t))


def any_term(normalized_text: str, terms: Iterable[str]) -> bool:
    return any(t and has_term(normalized_text, t) for t in terms)
