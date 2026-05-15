import numpy as np
import pandas as pd

from scs_morph.analysis.settings import AnalysisSetting
from scs_morph.claims.filters import ClaimCriteria
from scs_morph.claims.normative import NormativeDeviationClaimExtractor


def test_normative_claims_generate_composite_and_atomic():
    rng = np.random.default_rng(2)
    n_ref = 40
    n_target = 40
    group = ["CN"] * n_ref + ["AD"] * n_target
    age = np.r_[rng.normal(70, 3, n_ref), rng.normal(72, 3, n_target)]
    X = pd.DataFrame(
        {
            "Left-Hippocampus": np.r_[rng.normal(0, 1, n_ref), rng.normal(-3, 1, n_target)],
            "Right-Amygdala": np.r_[rng.normal(0, 1, n_ref), rng.normal(-3, 1, n_target)],
            "Left-Caudate": np.r_[rng.normal(0, 1, n_ref), rng.normal(-3, 1, n_target)],
        }
    )
    metadata = pd.DataFrame({"diagnosis_3class": group, "age": age, "sex": ["M", "F"] * 40})
    setting = AnalysisSetting(
        setting_id="NORM_TOY",
        dataset="toy",
        cohort="pooled",
        feature_space="fs8_subcortical_volume",
        feature_family="volume",
        pipeline="fs8",
        atlas="subcortical",
        target=None,
        claim_type="normative_deviation",
        covariates=["age", "sex"],
        reference_filter='diagnosis_3class == "CN"',
        reference_group_label="CN",
        target_group_column="diagnosis_3class",
        target_group_values=["AD"],
        min_n=10,
        z_threshold=1.0,
    )
    criteria = ClaimCriteria(normative_z_threshold=1.0, normative_min_support_rois=2)
    family, stats = NormativeDeviationClaimExtractor(criteria).extract_family(X, metadata, setting)
    assert any(card.claim_level == "normative_deviation" for card in family.normative_claims)
    assert any(card.claim_level == "normative_atomic_roi" for card in family.normative_claims)
    assert not stats.empty

