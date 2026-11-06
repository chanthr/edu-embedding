from edu_embedding.aggregate import cap_per_university
from edu_embedding.filters import lexical_guard, pick_category
from edu_embedding.recommender import Recommender, RecommenderConfig
from edu_embedding.text import normalize


def test_guard_ignores_substring_hits():
    text = normalize("Sustainability Studies: maintaining ecosystems, HTML basics")
    ok, hits = lexical_guard(text, must={"ai", "ml"}, min_hits=1)
    assert not ok and hits == 0


def test_category_detection_uses_word_boundaries():
    assert pick_category("I want to explain how machines work") is None  # "ai" in "explain"
    assert pick_category("ai and robots") == "data-science"
    assert pick_category("international law") == "law-legal"


def test_per_university_cap_is_applied_by_default(tiny_corpus, fake_encoder):
    assert RecommenderConfig().per_university_cap == 3
    cands = [{"university": "U", "program": str(i), "score": 10 - i} for i in range(5)]
    assert len(cap_per_university(cands, 3)) == 3
    assert len(cap_per_university(cands, None)) == 5


def test_end_to_end_with_fake_encoder(tiny_corpus, fake_encoder):
    rec = Recommender(tiny_corpus, fake_encoder, config=RecommenderConfig(rerank=False, topk=5))
    out = rec.recommend("machine learning")
    assert out, "expected at least one program"
    assert out[0]["program"] in {"Computer Science BSc", "Data Science BSc"}
    assert all(r["university"] for r in out)
    out = rec.recommend(["law"])
    assert out[0]["program"] == "Law LLB"


def test_empty_query_does_not_crash(tiny_corpus, fake_encoder):
    rec = Recommender(tiny_corpus, fake_encoder, config=RecommenderConfig(rerank=False))
    assert isinstance(rec.recommend(""), list)
