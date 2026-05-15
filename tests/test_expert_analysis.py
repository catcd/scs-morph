import pandas as pd

from scs_morph.expert.analysis import assign_expert_decision, summarize_claim_scores


CONFIG = {
    "decision_rules": {
        "expert_accept": {"min_mean_plausibility": 4.0, "min_mean_faithfulness": 4.0, "min_mean_usefulness": 3.5},
        "expert_reject": {"max_mean_plausibility": 2.5, "max_mean_faithfulness": 2.5},
        "high_disagreement": {"min_score_sd": 1.25},
    }
}


def test_expert_decision_rules():
    assert assign_expert_decision({"human_n": 1}, CONFIG) == "insufficient_reviews"
    assert assign_expert_decision({"human_n": 3, "high_disagreement_flag": True}, CONFIG) == "expert_disagreement"
    assert assign_expert_decision({"human_n": 3, "human_mean_plausibility": 4.5, "human_mean_faithfulness": 4.2, "human_mean_usefulness": 4, "majority_action_human": "accept"}, CONFIG) == "expert_accepted"
    assert assign_expert_decision({"human_n": 3, "human_mean_plausibility": 2, "human_mean_faithfulness": 4, "majority_action_human": "accept"}, CONFIG) == "expert_rejected"


def test_summarize_claim_scores_flags_acceptance():
    scores = pd.DataFrame({
        "review_id": ["r1", "r1", "r2", "r2"] * 5,
        "claim_id": ["c1"] * 20,
        "claim_family_id": ["f1"] * 20,
        "reviewer_id": ["h1", "h1", "h2", "h2"] * 5,
        "reviewer_type": ["human"] * 20,
        "dimension": ["plausibility_1_5"] * 4 + ["clarity_1_5"] * 4 + ["faithfulness_1_5"] * 4 + ["usefulness_1_5"] * 4 + ["reviewer_confidence_1_5"] * 4,
        "score": [4, 4, 5, 5] * 5,
        "caution_level": ["stable"] * 20,
        "recommended_action": ["accept"] * 20,
    })
    meta = pd.DataFrame({"claim_id": ["c1"], "claim_text": ["text"]})
    summary = summarize_claim_scores(scores, meta, CONFIG)
    assert summary.loc[0, "expert_decision"] == "expert_accepted"
