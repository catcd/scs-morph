from scs_morph.claims.cards import ClaimCard
from scs_morph.claims.family import ClaimFamily


def _card(level):
    return ClaimCard(
        claim_id=f"c_{level}",
        claim_family_id="f1",
        claim_level=level,
        claim_type="continuous_association" if level != "ranking" else "feature_ranking",
        source_setting={},
        dataset="toy",
        cohort="pooled",
        feature_space="fs",
        feature_family="volume",
        pipeline="p",
        atlas="a",
        target="y",
        covariates=[],
        n_samples=10,
        n_features=2,
        criteria_passed=True,
        criteria_details={},
        claim_text="text",
        short_claim_text="text",
        machine_readable_statement={},
    )


def test_claim_family_roundtrip_and_all_claims():
    family = ClaimFamily(
        claim_family_id="f1",
        setting_id="s1",
        setting_metadata={"dataset": "toy", "feature_space": "fs", "claim_type": "continuous_association"},
        composite_claims=[_card("composite")],
        atomic_claims=[_card("atomic_roi")],
        ranking_claims=[_card("ranking")],
    )
    assert len(family.all_claims) == 3
    restored = ClaimFamily.from_json(family.to_json())
    assert restored.claim_family_id == "f1"
    assert restored.summary_row()["n_total_claims"] == 3

