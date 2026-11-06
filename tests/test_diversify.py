from edu_embedding.diversify import mmr, title_overlap


def cand(program, university, score):
    return {"program": program, "university": university, "score": score}


def test_lambda_one_is_plain_ranking():
    cands = [cand("A", "U1", 9.0), cand("B", "U2", 5.0), cand("C", "U3", 7.0)]
    out = mmr(cands, lambda_=1.0, topk=3)
    assert [c["program"] for c in out] == ["A", "C", "B"]


def test_mmr_prefers_a_different_program_over_a_near_duplicate():
    cands = [
        cand("Computer Science BSc", "Uni A", 10.0),
        cand("Computer Science MSci", "Uni A", 9.9),
        cand("History BA", "Uni B", 9.5),
        cand("Music BA", "Uni C", 1.0),
    ]
    out = mmr(cands, lambda_=0.5, topk=2)
    assert [c["program"] for c in out] == ["Computer Science BSc", "History BA"]


def test_mmr_is_scale_free():
    small = [cand("X", "U1", 1.0), cand("X two", "U1", 0.9), cand("Y", "U2", 0.5)]
    big = [dict(c, score=c["score"] * 1000) for c in small]
    assert [c["program"] for c in mmr(small, topk=3)] == [c["program"] for c in mmr(big, topk=3)]


def test_redundancy_is_penalised_not_rewarded():
    first = cand("Physics", "Uni A", 1.0)
    twin = cand("Physics", "Uni A", 0.8)
    other = cand("Music", "Uni B", 0.8)
    assert title_overlap(first, twin) == 1.0
    out = mmr([first, twin, other], lambda_=0.75, topk=2)
    assert out[1]["program"] == "Music"
