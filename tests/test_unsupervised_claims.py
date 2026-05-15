import numpy as np
import pandas as pd

from scs_morph.analysis.settings import AnalysisSetting
from scs_morph.claims.filters import ClaimCriteria
from scs_morph.claims.unsupervised import LatentMorphometryClaimExtractor


def test_unsupervised_claims_generate_latent_axis_and_association():
    rng = np.random.default_rng(1)
    n = 60
    latent = rng.normal(size=n)
    X = pd.DataFrame({"LH_Default_Temp_1": latent + rng.normal(scale=0.1, size=n), "RH_Cont_PFCl_1": -latent + rng.normal(scale=0.1, size=n), "LH_Vis_1": rng.normal(size=n)})
    metadata = pd.DataFrame({"age": latent * 5 + 70, "sex": ["M", "F"] * 30})
    setting = AnalysisSetting(
        setting_id="LATENT_TOY",
        dataset="toy",
        cohort="pooled",
        feature_space="fs8_thickness_schaefer200_7",
        feature_family="thickness",
        pipeline="fs8",
        atlas="schaefer",
        target=None,
        claim_type="latent_morphometry_axis",
        covariates=[],
        n_components=2,
        metadata_association_targets=["age"],
        min_n=10,
    )
    criteria = ClaimCriteria(latent_min_explained_variance=0.01, latent_min_metadata_abs_corr=0.1)
    family, loadings = LatentMorphometryClaimExtractor(criteria).extract_family(X, metadata, setting)
    assert len(family.latent_claims) >= 1
    assert not loadings.empty

