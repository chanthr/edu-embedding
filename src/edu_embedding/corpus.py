"""Program passage corpus: loading and schema checks.

The corpus is not shipped with this repository. Pass its path with
``--corpus`` or the ``EDU_EMBED_CORPUS`` environment variable.
Each JSONL record is one section of one university program::

    {"id": str, "title": str, "text": str,
     "meta": {"university": str, "program_name": str, "section": str,
              "language": str, "cluster": str}}

``text`` starts with tags, for example
``[UNIV:...][MAJOR:...][LANG:EN][SECTION:overview]``, followed by the body.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SECTIONS: tuple[str, ...] = ("overview", "core_modules", "skills", "capstone", "outcomes")

# Non-greedy groups with backtracking, so program names that contain brackets
# such as "Biology [BSc][MSci]" still parse.
TAG_RE = re.compile(
    r"^\[UNIV:(?P<univ>.*?)\]\[MAJOR:(?P<major>.*?)\]"
    r"\[LANG:(?P<lang>[^\]]*)\]\[SECTION:(?P<section>[^\]]*)\]\s*"
)

REQUIRED_KEYS = ("id", "title", "text", "meta")
REQUIRED_META = ("university", "program_name", "section", "language", "cluster")


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str
    university: str
    program: str
    section: str
    language: str
    cluster: str

    @property
    def program_key(self) -> tuple[str, str]:
        return (self.university, self.program)

    @property
    def body(self) -> str:
        """Text without the leading tags."""
        return strip_tags(self.text)

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "meta": {
                "university": self.university,
                "program_name": self.program,
                "section": self.section,
                "language": self.language,
                "cluster": self.cluster,
            },
        }


def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text or "", count=1).strip()


def validate_record(rec: dict, line_no: int | None = None) -> None:
    where = f" (line {line_no})" if line_no is not None else ""
    for key in REQUIRED_KEYS:
        if key not in rec:
            raise ValueError(f"missing key '{key}'{where}")
    meta = rec["meta"]
    if not isinstance(meta, dict):
        raise ValueError(f"'meta' must be an object{where}")
    for key in REQUIRED_META:
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"meta.{key} must be a non-empty string{where}")
    if meta["section"] not in SECTIONS:
        raise ValueError(f"unknown section '{meta['section']}'{where}")
    if not isinstance(rec["text"], str) or not rec["text"].strip():
        raise ValueError(f"empty text{where}")


class CorpusNotFoundError(FileNotFoundError):
    pass


CORPUS_ENV = "EDU_EMBED_CORPUS"
NO_CORPUS_MESSAGE = (
    "No corpus given. Pass --corpus PATH or set EDU_EMBED_CORPUS. "
    "The corpus is a JSONL file of program passages; see data/README.md for the format."
)


def resolve_corpus_path(path: str | Path | None = None) -> Path:
    """Return the corpus path from the argument or ``EDU_EMBED_CORPUS``.

    There is no default: the corpus is private and not part of the repository.
    """
    raw = path or os.getenv(CORPUS_ENV)
    if not raw:
        raise CorpusNotFoundError(NO_CORPUS_MESSAGE)
    p = Path(raw).expanduser()
    if not p.is_file():
        raise CorpusNotFoundError(f"Corpus file not found: {p}")
    return p


def read_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                records.append(json.loads(line))
                records[-1]["_line"] = line_no
    return records


def load_corpus(path: str | Path | None = None, *, validate: bool = True) -> list[Passage]:
    """Load a corpus. Ids must be unique (see ``dedupe_ids``)."""
    path = resolve_corpus_path(path)
    passages: list[Passage] = []
    seen: set[str] = set()
    for rec in read_records(path):
        line_no = rec.pop("_line", None)
        if validate:
            validate_record(rec, line_no)
            if rec["id"] in seen:
                raise ValueError(f"duplicate id '{rec['id']}' (line {line_no})")
        seen.add(rec["id"])
        m = rec["meta"]
        passages.append(
            Passage(
                id=rec["id"], title=rec["title"], text=rec["text"],
                university=m["university"], program=m["program_name"],
                section=m["section"], language=m["language"], cluster=m["cluster"],
            )
        )
    return passages


def corpus_fingerprint(passages: Iterable[Passage]) -> str:
    h = hashlib.sha256()
    for p in passages:
        h.update(p.id.encode())
        h.update(b"\x00")
        h.update(p.text.encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def dedupe_ids(records: list[dict]) -> tuple[list[dict], dict]:
    """Make record ids unique.

    A record whose body text equals an earlier record with the same id is
    dropped. A record with the same id but different text is kept, and its id
    gets a suffix ("__2", "__3"). Returns (records, report).
    """
    by_id: dict[str, list[dict]] = defaultdict(list)
    out: list[dict] = []
    report: dict = {"dropped": [], "renamed": []}
    for rec in records:
        rec = dict(rec)
        same_id = by_id[rec["id"]]
        body = _collapse_ws(strip_tags(rec["text"]))
        if any(_collapse_ws(strip_tags(r["text"])) == body for r in same_id):
            report["dropped"].append(rec["id"])
            continue
        if same_id:
            new_id = f"{rec['id']}__{len(same_id) + 1}"
            report["renamed"].append({"from": rec["id"], "to": new_id})
            rec["id"] = new_id
        same_id.append(rec)
        out.append(rec)
    return out, report
