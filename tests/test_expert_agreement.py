import pandas as pd

from scs_morph.expert.agreement import compute_human_llm_agreement, compute_icc_2_1, compute_icc_2_k, pairwise_reviewer_correlations


def test_icc_and_correlations():
    matrix = pd.DataFrame({"r1": [1, 2, 3], "r2": [1, 2, 4], "r3": [2, 2, 3]})
    assert compute_icc_2_1(matrix) is not None
    assert compute_icc_2_k(matrix) is not None
    wide = pd.DataFrame({"claim_id": ["c1", "c2"], "reviewer_id": ["r1", "r1"], "dim": [1, 2]})
    corr = pairwise_reviewer_correlations(pd.DataFrame({"claim_id": ["c1", "c1", "c2", "c2"], "reviewer_id": ["r1", "r2", "r1", "r2"], "dim": [1, 1, 2, 3]}), ["dim"])
    assert not corr.empty


def test_human_llm_agreement():
    long = pd.DataFrame({
        "claim_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
        "reviewer_id": ["h1", "h2", "llm", "h1", "h2", "llm"],
        "reviewer_type": ["human", "human", "llm", "human", "human", "llm"],
        "dimension": ["plausibility_1_5"] * 6,
        "score": [4, 5, 5, 2, 3, 2],
    })
    out = compute_human_llm_agreement(long)
    assert out.loc[0, "n_overlap"] == 2

