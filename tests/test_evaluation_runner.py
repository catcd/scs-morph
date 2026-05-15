import pandas as pd

from scs_morph.claims.family import ClaimFamily
from scs_morph.evaluation.evaluator import evaluate_pair
from scs_morph.evaluation.pairs import EvaluationPair


def _family(setting_id="SRC"):
    return ClaimFamily(
        claim_family_id=f"{setting_id}_fam",
        setting_id=setting_id,
        setting_metadata={
            "dataset": "D1" if setting_id == "SRC" else "D2",
            "feature_space": "fs",
            "claim_type": "mmse_association",
            "target": "mmse",
        },
    )


def _pair():
    return EvaluationPair(
        pair_id="p1",
        experiment_id="e1",
        experiment_type="cross_dataset_transfer",
        source_dataset="D1",
        source_feature_space="fs",
        source_setting_id="SRC",
        source_claim_type="mmse_association",
        source_target="mmse",
        source_filter=None,
        target_dataset="D2",
        target_feature_space="fs",
        target_setting_id="TGT",
        target_claim_type="mmse_association",
        target_target="mmse",
        target_filter=None,
        roi_match_strategy="exact_roi",
        transfer_target_type="none",
    )


def test_evaluate_pair_returns_result(monkeypatch):
    stats = pd.DataFrame({"roi": ["a", "b", "c", "d", "e"], "coef": [1, 2, 3, 4, 5], "t": [1, 2, 3, 4, 5], "q": [0.1] * 5})
    monkeypatch.setattr("scs_morph.evaluation.evaluator.load_roi_statistics_for_setting", lambda setting_id, base_dir: stats)
    result = evaluate_pair(_pair(), [_family("SRC"), _family("TGT")], {"processed_dir": "data/processed"}, {"stability_thresholds": {}, "transfer_thresholds": {}})
    assert result.status == "evaluated"
    assert result.n_matched_features == 5


def test_missing_family_returns_skipped():
    result = evaluate_pair(_pair(), [_family("SRC")], {"processed_dir": "data/processed"}, {})
    assert result.status == "skipped"
    assert "target claim family missing" in result.warnings
