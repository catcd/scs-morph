from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

import numpy as np


CLAIM_LEVELS = {
    "composite",
    "atomic_roi",
    "group_summary",
    "ranking",
    "direction_summary",
    "latent_axis",
    "latent_axis_association",
    "normative_deviation",
    "normative_atomic_roi",
}

CLAIM_TYPES = {
    "diagnosis_contrast",
    "cdr_impairment_contrast",
    "mmse_association",
    "age_association",
    "continuous_association",
    "demographic_contrast",
    "acquisition_contrast",
    "feature_ranking",
    "group_enrichment",
    "direction_summary",
    "latent_morphometry_axis",
    "latent_axis_association",
    "normative_deviation",
    "normative_group_deviation",
}

CLAIM_STATUSES = {"generated", "weak_support", "supported", "criteria_failed", "skipped", "error"}


def _to_native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _to_native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_to_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_native(item) for item in value]
    if isinstance(value, tuple):
        return [_to_native(item) for item in value]
    return value


@dataclass
class ClaimCard:
    """JSON-serializable scientific claim card."""

    claim_id: str
    claim_family_id: str
    claim_level: str
    claim_type: str
    source_setting: dict
    dataset: str
    cohort: str | None
    feature_space: str
    feature_family: str | None
    pipeline: str | None
    atlas: str | None
    target: str | None
    covariates: list[str]
    n_samples: int
    n_features: int
    roi: str | None = None
    hemisphere: str | None = None
    roi_group: str | None = None
    anatomical_group: str | None = None
    network: str | None = None
    lobe: str | None = None
    direction: str | None = None
    effect_size: float | None = None
    evidence_metric: str = "t"
    evidence_value: float | None = None
    coefficient: float | None = None
    standard_error: float | None = None
    t_value: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    support_regions: list[str] = field(default_factory=list)
    support_region_effects: dict[str, float] = field(default_factory=dict)
    support_metrics: dict = field(default_factory=dict)
    criteria_passed: bool = False
    criteria_details: dict = field(default_factory=dict)
    claim_text: str = ""
    short_claim_text: str = ""
    machine_readable_statement: dict = field(default_factory=dict)
    expert_review_hint: str | None = None
    status: str = "generated"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.claim_level not in CLAIM_LEVELS:
            raise ValueError(f"Unsupported claim_level: {self.claim_level}")
        if self.claim_type not in CLAIM_TYPES:
            raise ValueError(f"Unsupported claim_type: {self.claim_type}")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return _to_native(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimCard":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "ClaimCard":
        return cls.from_dict(json.loads(text))

