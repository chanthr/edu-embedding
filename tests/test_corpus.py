import json

import pytest

from edu_embedding.cli import main
from edu_embedding.corpus import CorpusNotFoundError, dedupe_ids, load_corpus, resolve_corpus_path, strip_tags


def rec(i, uni, prog, section, body):
    return {
        "id": i,
        "title": f"{uni} | {prog} | {section}",
        "text": f"[UNIV:{uni}][MAJOR:{prog}][LANG:EN][SECTION:{section}] {body}",
        "meta": {"university": uni, "program_name": prog, "section": section, "language": "en", "cluster": "x"},
    }


def test_duplicate_ids_with_different_text_get_suffixes():
    rows = [rec("p_overview_001", "U", "P", "overview", "one"), rec("p_overview_001", "U", "P", "overview", "two")]
    out, report = dedupe_ids(rows)
    assert [r["id"] for r in out] == ["p_overview_001", "p_overview_001__2"]
    assert report["renamed"] == [{"from": "p_overview_001", "to": "p_overview_001__2"}]


def test_duplicate_ids_with_same_body_are_dropped():
    rows = [rec("a", "U", "Neuro BSc / MSci", "overview", "same body"), rec("a", "U", "Neuro BSc/ MSci", "overview", "same  body")]
    out, report = dedupe_ids(rows)
    assert len(out) == 1 and report["dropped"] == ["a"]


def test_strip_tags_handles_brackets_in_program_names():
    text = "[UNIV:Some University][MAJOR:Biology [BSc][MSci]][LANG:EN][SECTION:overview] Genes."
    assert strip_tags(text) == "Genes."


def test_loader_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(rec("same", "U", "P", "overview", "t")) for _ in range(2)))
    with pytest.raises(ValueError):
        load_corpus(p)


def test_loader_validates_schema(tmp_path):
    bad = rec("x", "U", "P", "introduction", "t")  # unknown section
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_corpus(p)


def test_no_corpus_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("EDU_EMBED_CORPUS", raising=False)
    with pytest.raises(CorpusNotFoundError, match="--corpus"):
        resolve_corpus_path(None)


def test_corpus_from_environment(monkeypatch, tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(rec("a", "U", "P", "overview", "t")))
    monkeypatch.setenv("EDU_EMBED_CORPUS", str(p))
    assert [x.id for x in load_corpus()] == ["a"]


def test_cli_search_without_corpus_exits_before_loading_a_model(monkeypatch, capsys):
    monkeypatch.delenv("EDU_EMBED_CORPUS", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["search", "biology"])
    assert exc.value.code == 2
    assert "No corpus given" in capsys.readouterr().err
