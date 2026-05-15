from __future__ import annotations

import pandas as pd

from scs_morph.analysis.settings import AnalysisSetting
from scs_morph.claims.base import SupervisedClaimFamilyBuilder
from scs_morph.claims.family import ClaimFamily


class DiagnosisContrastClaimExtractor(SupervisedClaimFamilyBuilder):
    """Claim-family extractor for binary contrasts."""

    def extract_family(self, X: pd.DataFrame, metadata: pd.DataFrame, setting: AnalysisSetting) -> tuple[ClaimFamily, pd.DataFrame]:
        if setting.target in metadata.columns:
            values = set(pd.to_numeric(metadata[setting.target].dropna(), errors="coerce").dropna().unique())
            if not values.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f"Contrast target '{setting.target}' must contain only 0/1 values; found {sorted(values)}")
        return super().extract_family(X, metadata, setting)

