"""Program clusters (14) and the production keyword rule that guesses them.

``TAXONOMY`` and ``guess_cluster`` are unchanged from production. Clusters are
checked in dictionary order and the first match wins, so the order matters:
for example "Computer Science and Technology" lands in engineering-science
because "technology" is listed there and that cluster comes first.
``all_clusters`` returns every matching cluster and is used by the evaluation.
"""
from __future__ import annotations

import re

TAXONOMY: dict[str, list[str]] = {
    "economics": [
        "economics",
        "econometric",
        "macroeconomics",
        "macro economics",
        "microeconomics",
        "micro economics",
        "quantitative economics",
        "economic policy",
        "development economics",
    ],
    "business-finance": [
        "business",
        "management",
        "finance",
        "accounting",
        "accountancy",
        "marketing",
        "banking",
        "investment",
        "actuarial",
        "economics",
        "econometric",
        "commerce",
        "entrepreneurship",
    ],
    "data-science": [
        "data science",
        "analytics",
        "machine learning",
        "ai",
        "artificial intelligence",
        "statistics",
        "statistical",
        "computational",
        "big data",
        "data analytics",
    ],
    "engineering-science": [
        "engineering",
        "mechanical",
        "electrical",
        "civil",
        "chemical",
        "aerospace",
        "materials",
        "mechatronics",
        "industrial",
        "manufacturing",
        "robotics",
        "geophysical",
        "geophysics",
        "geological engineering",
        "technology",
        "systems engineering",
        "surveying",
        "survey",
        "maritime",
        "naval",
        "geoscience",
        "geosciences",
        "bioengineering",
    ],
    "computer-science": [
        "computer science",
        "software",
        "software engineering",
        "computing",
        "programming",
        "informatics",
        "cybersecurity",
        "information technology",
        "information systems",
        "information security",
    ],
    "math-statistics": [
        "mathematics",
        "maths",
        "statistics",
        "applied mathematics",
        "pure mathematics",
        "quantitative",
        "operations research",
        "mathematical",
    ],
    "psychology-neurosci": [
        "psychology",
        "neuroscience",
        "neuro",
        "cognitive",
        "psychiatry",
        "behavioural",
    ],
    "medicine-health": [
        "medicine",
        "medical",
        "clinical",
        "nursing",
        "pharmacy",
        "pharmacology",
        "biomedical",
        "health",
        "dentistry",
        "surgery",
        "physiotherapy",
        "midwifery",
        "veterinary",
        "public health",
        "anatomy",
        "therapy",
        "occupational therapy",
        "radiography",
        "optometry",
        "nutrition",
        "audiology",
    ],
    "life-sciences": [
        "biology",
        "biological",
        "biological science",
        "biochemistry",
        "biotechnology",
        "genetics",
        "microbiology",
        "bioscience",
        "zoology",
        "botany",
        "ecology",
        "immunology",
        "immunity",
        "infection",
        "biomedical science",
        "anatomical",
        "biomedicine",
        "cancer biomedicine",
        "chemistry",
        "geography",
        "earth sciences",
        "earth science",
        "earth",
        "geology",
        "geoscience",
        "geosciences",
        "environmental science",
        "environmental studies",
        "marine science",
        "natural sciences",
        "natural science",
        "planetary science",
        "planetary",
        "sports science",
        "sport",
        "physics",
        "applied physics",
    ],
    "art-design": [
        "fine art",
        "creative",
        "design",
        "media",
        "film",
        "architecture",
        "visual",
        "performing arts",
        "music",
        "fashion",
        "digital",
        "performance",
        "dance",
        "musical theatre",
    ],
    "social-sciences": [
        "sociology",
        "social science",
        "politics",
        "policy",
        "anthropology",
        "international relations",
        "development studies",
        "development",
        "global development",
        "american studies",
        "criminology",
        "communication",
        "communications",
        "planning",
        "urban planning",
        "urban studies",
        "social work",
        "humanitarian",
        "war studies",
        "journalism",
    ],
    "education": [
        "education",
        "teaching",
        "pedagogy",
        "teacher",
        "curriculum",
        "early childhood",
        "primary education",
        "secondary education",
    ],
    "humanities": [
        "history",
        "literature",
        "linguistics",
        "philosophy",
        "classics",
        "languages",
        "language",
        "english",
        "political economy",
        "cultural studies",
        "religion",
        "theology",
        "archaeology",
        "italian",
        "chinese",
        "japanese",
        "french",
        "spanish",
        "german",
        "dutch",
        "hebrew",
        "jewish",
        "icelandic",
        "norwegian",
        "swedish",
        "viking",
        "old norse",
        "arabic",
        "liberal arts",
        "arts and sciences",
        "humanities and sciences",
    ],
    "law-legal": [
        "law",
        "legal",
        "llb",
        "juris",
        "juridical",
        "justice",
        "criminology and law",
    ],
}

CLUSTERS: tuple[str, ...] = tuple(TAXONOMY)


def _matches(s: str, key: str) -> bool:
    key_l = key.lower().strip()
    if not key_l:
        return False
    # Multi-word or hyphenated keys are matched as substrings (production rule).
    if " " in key_l or "-" in key_l or "/" in key_l:
        return key_l in s
    return re.search(r"\b{}\b".format(re.escape(key_l)), s) is not None


def guess_cluster(name_or_text: str | None) -> str | None:
    """First cluster (in ``TAXONOMY`` order) with a matching keyword."""
    if not name_or_text:
        return None
    s = name_or_text.lower()
    for cluster, keys in TAXONOMY.items():
        if any(_matches(s, k) for k in keys):
            return cluster
    return None


def all_clusters(name_or_text: str | None) -> set[str]:
    """Every cluster with at least one matching keyword."""
    if not name_or_text:
        return set()
    s = name_or_text.lower()
    return {c for c, keys in TAXONOMY.items() if any(_matches(s, k) for k in keys)}
