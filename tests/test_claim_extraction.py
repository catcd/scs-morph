import numpy as np
import pandas as pd

from scs_morph.analysis.settings import AnalysisSetting
from scs_morph.claims.continuous import ContinuousAssociationClaimExtractor
from scs_morph.claims.contrast import DiagnosisContrastClaimExtractor


def _toy_data():
    rng = np.random.default_rng(7)
    n = 60
    target = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    X = pd.DataFrame(
        {
            "roi1": target + rng.normal(scale=0.1, size=n),
            "roi2": rng.normal(size=n),
            "roi3": -target + rng.normal(scale=0.1, size=n),
        },
        index=[f"s{i}" for i in range(n)],
    )
    metadata = pd.DataFrame(
        {
            "diagnosis_binary_ad_cn": target,
            "mmse": 30 - target + rng.normal(scale=0.1, size=n),
            "age": rng.normal(72, 5, size=n),
            "sex": ["M", "F"] * (n // 2),
        },
        index=X.index,
    )
    return X, metadata


def test_binary_claim_extraction_returns_card():
    X, metadata = _toy_data()
    setting = AnalysisSetting(
        setting_id="toy_binary",
        dataset="toy",
        cohort="pooled",
        feature_space="toy_space",
        feature_family="volume",
        pipeline="toy_pipeline",
        atlas="toy_atlas",
        target="diagnosis_binary_ad_cn",
        claim_type="diagnosis_contrast",
        covariates=["age", "sex"],
        positive_label="AD",
        negative_label="CN",
        top_k=2,
        min_n=10,
    )
    card = DiagnosisContrastClaimExtractor().extract(X, metadata, setting)
    assert card.claim_type == "diagnosis_contrast"
    assert len(card.support_regions) == 2


def test_continuous_claim_extraction_returns_card():
    X, metadata = _toy_data()
    setting = AnalysisSetting(
        setting_id="toy_mmse",
        dataset="toy",
        cohort="pooled",
        feature_space="toy_space",
        feature_family="volume",
        pipeline="toy_pipeline",
        atlas="toy_atlas",
        target="mmse",
        claim_type="continuous_association",
        covariates=["age", "sex"],
        top_k=5,
        standardize_target=True,
        min_n=10,
    )
    card = ContinuousAssociationClaimExtractor().extract(X, metadata, setting)
    assert card.claim_type == "continuous_association"
    assert len(card.support_regions) == 3


def test_claim_extraction_handles_duplicate_indices():
    X, metadata = _toy_data()
    X.index = ["dup"] * len(X)
    metadata.index = ["dup"] * len(metadata)
    setting = AnalysisSetting(
        setting_id="toy_duplicate_index",
        dataset="toy",
        cohort="pooled",
        feature_space="toy_space",
        feature_family="volume",
        pipeline="toy_pipeline",
        atlas="toy_atlas",
        target="mmse",
        claim_type="continuous_association",
        covariates=["age", "sex"],
        top_k=2,
        standardize_target=True,
        min_n=10,
    )
    card = ContinuousAssociationClaimExtractor().extract(X, metadata, setting)
    assert len(card.support_regions) == 2
