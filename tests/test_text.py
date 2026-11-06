from edu_embedding.index import BM25Index
from edu_embedding.query import analyse_query
from edu_embedding.text import has_term, normalize, query_term, tokenize


def test_bm25_documents_and_queries_share_one_tokenizer():
    docs = ["Statistics, data-driven analytics.", "History of art.", "Modern languages.", "Music theory."]
    index = BM25Index(docs)
    # Production split documents on whitespace, so "analytics." never matched
    # the query token "analytics".
    assert index.doc_tokens[0] == ["statistics", "data", "driven", "analytics"]
    plan = analyse_query("analytics")
    assert plan.bm25_tokens == tokenize("analytics")
    scores = index.scores(plan.bm25_tokens)
    assert scores[0] > 0 and not scores[1:].any()


def test_bm25_tokens_split_multiword_synonyms():
    plan = analyse_query("cs")
    assert "computer science" in plan.tokens
    assert {"computer", "science"} <= set(plan.bm25_tokens)


def test_terms_match_whole_words_not_substrings():
    assert not has_term(normalize("Sustainability and maintenance"), "ai")
    assert not has_term(normalize("HTML and CSS"), "ml")
    assert not has_term(normalize("a flawless taxonomy"), "law")
    assert not has_term(normalize("a flawless taxonomy"), "tax")
    assert not has_term(normalize("ancient civilisations"), "civil")
    assert has_term(normalize("AI-driven systems"), "ai")
    assert has_term(normalize("ML, NLP"), "ml")
    assert has_term(normalize("Laws of motion"), "law")
    assert has_term(normalize("small businesses"), "business")
    assert has_term(normalize("machine-learning models"), "machine learning")


def test_stems_and_typed_words_match_word_starts():
    assert has_term(normalize("Econometrics"), "econom*")
    assert not has_term(normalize("Econometrics"), "econom")
    assert not has_term(normalize("socioeconomic"), "econom*")
    assert query_term("math") == "math*"
    assert query_term("computers") == "computer*"
    assert query_term("business") == "business*"
    assert query_term("art") == "art"
    assert has_term(normalize("Mathematics"), query_term("math"))
    assert has_term(normalize("computer science"), query_term("computers"))
    assert not has_term(normalize("computing"), query_term("computers"))


def test_multi_mode_is_reachable():
    plan = analyse_query("computer science, economics")
    assert plan.mode == "multi"
    assert analyse_query("computer science").mode == "keyword"
    assert analyse_query("I would like a course that mixes biology and computing.").mode == "sentence"
