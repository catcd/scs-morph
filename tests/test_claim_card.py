import json
from typing import Any, cast

import numpy as np

from scs_morph.claims.cards import ClaimCard


def test_claim_card_serializes_numpy_types():
    card = ClaimCard(
        claim_id="c1",
        claim_family_id="f1",
        claim_level="atomic_roi",
        claim_type="continuous_association",
        source_setting={"setting_id": "s1"},
        dataset="ADNI",
        cohort="pooled",
        feature_space="fs",
        feature_family="volume",
        pipeline="fs8",
        atlas="atlas",
        target="mmse",
        covariates=["age"],
        n_samples=cast(Any, np.int64(10)),
        n_features=cast(Any, np.int64(2)),
        roi="roi1",
        evidence_value=cast(Any, np.float64(2.0)),
        coefficient=cast(Any, np.float64(0.5)),
        standard_error=cast(Any, np.float64(0.1)),
        t_value=cast(Any, np.float64(5.0)),
        p_value=cast(Any, np.float64(0.01)),
        q_value=cast(Any, np.float64(0.02)),
        support_regions=["roi1"],
        support_region_effects=cast(Any, {"roi1": np.float64(0.5)}),
        support_metrics=cast(Any, {"n_fdr_0_05": np.int64(1)}),
        criteria_passed=True,
        criteria_details={"ok": True},
        claim_text="Example claim.",
        short_claim_text="Example claim.",
        machine_readable_statement={"roi": "roi1"},
    )
    encoded = card.to_json()
    json.loads(encoded)
    restored = ClaimCard.from_json(encoded)
    assert restored.claim_id == "c1"
    assert isinstance(restored.coefficient, float)
