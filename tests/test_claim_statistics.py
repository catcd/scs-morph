import numpy as np
import pandas as pd

from scs_morph.claims.statistics import add_fdr, benjamini_hochberg, fit_roiwise_ols


def test_fit_roiwise_ols_recovers_known_effect():
    rng = np.random.default_rng(42)
    n = 80
    y = pd.Series(rng.normal(size=n), name="target")
    covariates = pd.DataFrame({"age": rng.normal(70, 5, size=n)})
    X = pd.DataFrame(
        {
            "roi_effect": 2.0 * y + rng.normal(scale=0.2, size=n),
            "roi_null": rng.normal(size=n),
        }
    )

    results = fit_roiwise_ols(
        X,
        y,
        covariates,
        standardize_features=False,
        standardize_target=False,
    )
    effect = results.set_index("roi").loc["roi_effect"]
    assert float(effect["coef"]) > 1.5
    assert np.isfinite(float(effect["p"]))


def test_fdr_outputs_same_length_as_p_values():
    p_values = np.array([0.01, 0.2, np.nan, 0.04])
    q_values = benjamini_hochberg(p_values)
    assert len(q_values) == len(p_values)
    assert np.isnan(q_values[2])

    results = add_fdr(pd.DataFrame({"p": p_values}))
    assert "q" in results.columns
