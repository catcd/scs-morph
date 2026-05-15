import pandas as pd

from scs_morph.expert.selection import enrich_claims_with_evaluation, make_claim_dedup_key, select_main_expert_claims, validate_claim_for_review


def _config(n=2):
    return {
        "selection": {"random_seed": 42, "main_expert_n": n, "exclude_claim_levels": [], "require_criteria_passed_for_main": True},
        "main_expert_composition": {
            "claim_level_targets": {"composite": n},
            "target_type_minimums": {},
            "evaluation_category_minimums": {},
            "feature_family_minimums": {},
            "dataset_minimums": {},
        },
    }


def test_bad_claim_text_excluded():
    row = {"claim_text": "nan", "status": "generated"}
    assert not validate_claim_for_review(row)


def test_duplicate_dedup_key_avoided_and_requested_n_returned():
    claims = pd.DataFrame([
        {"claim_id": "1", "claim_family_id": "f1", "claim_level": "composite", "claim_type": "mmse_association", "target": "mmse", "feature_family": "volume", "claim_text": "Higher MMSE is associated with higher volume.", "status": "supported", "criteria_passed": True, "dataset": "ADNI", "feature_space": "fs8_cortical_volume", "direction": "higher"},
        {"claim_id": "2", "claim_family_id": "f2", "claim_level": "composite", "claim_type": "mmse_association", "target": "mmse", "feature_family": "volume", "claim_text": "Higher MMSE is associated with higher volume.", "status": "supported", "criteria_passed": True, "dataset": "AIBL", "feature_space": "fs8_cortical_volume", "direction": "higher"},
        {"claim_id": "3", "claim_family_id": "f3", "claim_level": "composite", "claim_type": "age_association", "target": "age", "feature_family": "thickness", "claim_text": "Older age is associated with lower thickness.", "status": "supported", "criteria_passed": True, "dataset": "OASIS1", "feature_space": "fs8_thickness_schaefer200_7", "direction": "lower"},
    ])
    evals = pd.DataFrame(columns=["source_claim_family_id", "target_claim_family_id", "overall_label", "stability_label", "transfer_label", "experiment_id"])
    enriched = enrich_claims_with_evaluation(claims, evals)
    selected = select_main_expert_claims(enriched, _config(2))
    assert len(selected) == 2
    assert selected["dedup_key"].nunique() == 2
    assert set(["review_set", "selection_reason"]).issubset(selected.columns)

