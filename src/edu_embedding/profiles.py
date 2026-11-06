"""Keyword data used by the query parser and the lexical guard.

Values are copied from the production recommender, with two changes:

* ``EXTRA_QUERY_STOPWORDS`` is new; the reason is given next to it.
* Production matched every term as a substring. Here a term matches whole
  words (plural allowed). Terms that production clearly used as stems
  ("econom", "comput", "statistic") carry a trailing "*" and match word starts.
  See ``edu_embedding.text.has_term``.
"""
from __future__ import annotations

# Section weights used when section scores are summed into a program score.
SECTION_WEIGHTS: dict[str, float] = {
    "overview": 1.25,
    "core_modules": 1.2,
    "skills": 1.1,
    "capstone": 0.95,
    "outcomes": 0.8,
}

# Production stopwords for query parsing.
QUERY_STOPWORDS: set[str] = {
    "and", "or", "for", "the", "with", "into", "from", "about",
    "program", "programs", "degree", "undergraduate", "bachelor", "major",
    "study", "studies", "course", "courses", "students",
}

# New. Production matched query tokens as substrings, so a token such as "i"
# matched almost every passage and function words never blocked anything.
# With word-boundary matching they would start to block results, so they are
# dropped from the guard terms instead. "is" is listed here, which also turns
# off the production synonym "is" -> "information systems" (it fired on the
# verb in sentences like "my passion is ...").
EXTRA_QUERY_STOPWORDS: set[str] = {
    "i", "im", "me", "my", "mine", "we", "our", "you", "your",
    "a", "an", "to", "of", "in", "on", "at", "by", "as", "is", "am", "are",
    "be", "been", "was", "were", "it", "its", "this", "that", "these", "those",
    "but", "so", "not", "no", "do", "does", "doing", "did", "can", "could",
    "would", "should", "will", "want", "wanna", "like", "love", "enjoy",
    "interested", "really", "very", "also", "too", "maybe",
    "things", "thing", "stuff", "something", "some", "lot", "lots", "kind",
    "good", "how", "what", "which", "who", "why", "when", "where", "m", "s",
}

PHRASES: list[str] = [
    "quantitative finance", "financial engineering", "risk management",
    "machine learning", "data science", "software engineering",
    "computer science", "electrical engineering", "mechanical engineering",
    "civil engineering", "aerospace engineering", "bioinformatics",
    "computational biology", "information systems", "business analytics",
    "operations research", "industrial engineering",
]

SYNONYMS: dict[str, list[str]] = {
    "cs": ["computer science"],
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "is": ["information systems"],
    "stats": ["statistics", "statistical"],
    "quant": ["quantitative", "quantitative finance"],
    "fineng": ["financial engineering"],
    "ba": ["business analytics"],
    "or": ["operations research"],
}

CATEGORY_PROFILES: dict[str, dict] = {
    "business-finance": {
        "must_any": {
            "business", "finance", "financial", "account*", "accounting",
            "management", "marketing", "bank", "banking", "investment",
            "corporate", "asset", "capital", "risk", "actuarial", "economics",
            "econometric", "quantitative", "tax", "audit", "strategy",
        },
        "ban_any": {
            "sociology", "social science", "anthropology", "humanities",
            "creative", "fine art", "literature", "linguistics", "education",
            "culture", "policy", "politics", "philosophy",
            "comparative literature", "sport", "sports", "hospitality",
            "leisure",
        },
        "bonus_any": {
            "economics", "finance", "accounting", "management", "marketing",
            "banking", "investment",
        },
        "min_hits": 2,
    },
    "data-science": {
        "must_any": {
            "data", "analytics", "statistics", "statistical",
            "machine learning", "ml", "ai", "artificial",
        },
        "ban_any": {
            "sport", "sports", "exercise", "physiology", "kinesiology",
            "biomechanics", "architecture", "graphic design", "fashion design",
            "interior design", "creative media",
        },
        "min_hits": 2,
    },
    "engineering-cs": {
        "must_any": {
            "engineering", "engineer", "computer", "software", "electrical",
            "mechanical", "civil", "chemical", "aerospace",
        },
        "ban_any": {
            "creative", "fine art", "comparative literature", "education",
            "social science",
        },
        "min_hits": 1,
    },
    "law-legal": {
        "must_any": {
            "law", "legal", "jurisprudence", "juris*", "juridical", "justice",
            "criminology", "llb", "constitutional", "criminal", "civil law",
            "commercial law", "international law",
        },
        "ban_any": {
            "geophysics", "geology", "geography", "physics", "engineering",
            "computer", "mathematics", "statistics", "chemistry", "biology",
            "medicine", "mechanical", "electrical", "civil engineering",
        },
        "min_hits": 1,
    },
}

# Checked in this order; the first category with a matching term wins.
CATEGORY_TRIGGERS: list[tuple[str, list[str]]] = [
    ("law-legal", [
        "law", "legal", "jurisprudence", "juris*", "juridical", "justice",
        "criminology", "llb", "constitutional", "criminal law", "civil law",
        "commercial law", "international law",
    ]),
    ("business-finance", [
        "business", "finance", "account*", "econom*", "management", "marketing",
        "bank*", "invest*",
    ]),
    ("computer-science", [
        "computer", "software", "informatics", "cyber*",
        "information technology", "information systems",
    ]),
    ("data-science", [
        "data", "analytics", "machine learning", "ai", "ml", "statistic*",
    ]),
    ("engineering-cs", [
        "engineering", "engineer*", "mechanical", "electrical", "civil",
        "chemical", "aero*", "robot*",
    ]),
]

# A passage title must contain one of these terms when its category is active.
TITLE_FILTERS: dict[str, list[str]] = {
    "data-science": [
        "data", "analytics", "statistic*", "comput*", "ai", "machine",
        "information", "artificial", "intelligence", "robot*",
    ],
    "business-finance": [
        "finance", "business", "econom*", "account*", "management", "bank*",
        "investment",
    ],
    "engineering-cs": [
        "engineer*", "computer", "software", "electrical", "mechanical",
    ],
    "law-legal": [
        "law", "legal", "jurisprudence", "juris*", "juridical", "justice",
        "criminology", "llb",
    ],
}

# Penalty terms for business-finance queries (score -0.07 when present).
BUSINESS_PENALTY_TERMS: list[str] = [
    "social science", "sociology", "anthropology", "humanities", "culture",
    "policy", "politics",
]

COMPATIBLE_CLUSTERS: dict[str, set[str]] = {
    "business-finance": {"business-finance", "economics"},
    "economics": {"economics", "business-finance", "math-statistics"},
    "data-science": {"data-science", "math-statistics", "computer-science"},
    "engineering-cs": {"engineering-cs", "computer-science", "engineering-science"},
    "computer-science": {
        "computer-science", "engineering-cs", "engineering-science", "data-science",
    },
    "engineering-science": {"engineering-science", "engineering-cs", "computer-science"},
    "law-legal": {
        "law-legal", "business-finance", "social-sciences", "humanities",
        "medicine-health", "engineering-science", "computer-science", "art-design",
    },
}
