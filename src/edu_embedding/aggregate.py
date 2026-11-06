"""Turn passage scores into program scores (production rules)."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .profiles import BUSINESS_PENALTY_TERMS, COMPATIBLE_CLUSTERS, SECTION_WEIGHTS
from .taxonomy import guess_cluster
from .text import any_term, count_terms, normalize


def program_cluster(passage) -> str | None:
    """Production rule: program name first, then title, then passage text."""
    return guess_cluster(passage.program) or guess_cluster(passage.title) or guess_cluster(passage.text)


def aggregate_programs(
    rows: list[dict],
    *,
    profile: dict,
    category: str | None,
    target_cluster: str | None,
    strict_taxonomy: bool = True,
) -> list[dict]:
    """Sum weighted section scores per (university, program).

    program score = sum(similarity * section_weight + lexical bonus)
                    + 0.05 * passages matched + 0.03 * max lexical hits
    With ``strict_taxonomy`` a passage is skipped when its program cluster is
    known and not compatible with the cluster guessed from the query.
    """
    grouped: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    allowed = COMPATIBLE_CLUSTERS.get(target_cluster or "", {target_cluster}) if target_cluster else None
    bonus_terms = sorted(profile.get("bonus_any", ()))

    for row in rows:
        p = row["passage"]
        if strict_taxonomy and allowed is not None:
            pc = program_cluster(p)
            if pc and pc not in allowed:
                continue
        combined = normalize(p.title + " " + p.text)
        lex_bonus = 0.03 * count_terms(combined, bonus_terms)
        if category == "business-finance" and any_term(combined, BUSINESS_PENALTY_TERMS):
            lex_bonus -= 0.07
        weight = SECTION_WEIGHTS.get(p.section, 1.0)
        state = grouped.setdefault(
            p.program_key,
            {"score": 0.0, "hits": 0, "sims": [], "samples": [], "lex_hits": 0},
        )
        state["score"] += row["similarity"] * weight + lex_bonus
        state["hits"] += 1
        state["sims"].append(row["similarity"])
        state["lex_hits"] = max(state["lex_hits"], int(row.get("lex_hits", 0)))
        if len(state["samples"]) < 3:
            state["samples"].append({"section": p.section, "preview": p.body[:220]})

    candidates = []
    for (university, program), s in grouped.items():
        candidates.append({
            "university": university,
            "program": program,
            "score": s["score"] + 0.05 * s["hits"] + 0.03 * s["lex_hits"],
            "sim_mean": float(np.mean(s["sims"])) if s["sims"] else 0.0,
            "passages": s["hits"],
            "samples": s["samples"],
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def cap_per_university(candidates: list[dict], cap: int | None) -> list[dict]:
    """Keep at most ``cap`` programs per university (None or 0 = no cap)."""
    if not cap:
        return list(candidates)
    counts: dict[str, int] = {}
    out = []
    for c in candidates:
        n = counts.get(c["university"], 0)
        if n >= cap:
            continue
        counts[c["university"]] = n + 1
        out.append(c)
    return out
