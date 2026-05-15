from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EXPERIMENT_TYPES = {
    "cross_dataset_transfer",
    "weak_label_transfer",
    "cohort_transfer",
    "subgroup_transfer",
    "pipeline_drift",
    "fastsurfer_drift",
    "atlas_sensitivity",
    "feature_family_sensitivity",
}
ROI_MATCH_STRATEGIES = {"exact_roi", "group_summary", "network_7_like", "lobe", "no_direct_roi_match"}
TRANSFER_TARGET_TYPES = {"binary", "continuous", "none", None}
WEIGHT_METRICS = {"coefficient", "t_value", "evidence_value"}
RESULT_STATUSES = {"evaluated", "skipped", "error"}


@dataclass
class EvaluationPair:
    pair_id: str
    experiment_id: str
    experiment_type: str
    source_dataset: str
    source_feature_space: str
    source_setting_id: str | None
    source_claim_type: str
    source_target: str | None
    source_filter: str | None
    target_dataset: str
    target_feature_space: str
    target_setting_id: str | None
    target_claim_type: str
    target_target: str | None
    target_filter: str | None
    roi_match_strategy: str
    transfer_target: str | None = None
    transfer_target_type: str | None = "none"
    positive_label: str | int | float | None = None
    negative_label: str | int | float | None = None
    use_claim_level: str = "composite"
    use_weight_metric: str = "coefficient"
    top_k: int = 20
    standardize_target_features: bool = True
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"Unsupported experiment_type: {self.experiment_type}")
        if self.roi_match_strategy not in ROI_MATCH_STRATEGIES:
            raise ValueError(f"Unsupported roi_match_strategy: {self.roi_match_strategy}")
        if self.transfer_target_type not in TRANSFER_TARGET_TYPES:
            raise ValueError(f"Unsupported transfer_target_type: {self.transfer_target_type}")
        if self.use_weight_metric not in WEIGHT_METRICS:
            raise ValueError(f"Unsupported use_weight_metric: {self.use_weight_metric}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationPair":
        return cls(**data)


@dataclass
class EvaluationResult:
    pair_id: str
    experiment_id: str
    experiment_type: str
    source_claim_family_id: str | None
    target_claim_family_id: str | None
    source_setting_id: str | None
    target_setting_id: str | None
    source_dataset: str
    target_dataset: str
    source_feature_space: str
    target_feature_space: str
    source_claim_type: str
    target_claim_type: str
    source_target: str | None
    target_target: str | None
    roi_match_strategy: str
    n_source_features: int = 0
    n_target_features: int = 0
    n_matched_features: int = 0
    n_source_support: int = 0
    n_target_support: int = 0
    evidence_metrics: dict = field(default_factory=dict)
    support_metrics: dict = field(default_factory=dict)
    direction_metrics: dict = field(default_factory=dict)
    magnitude_metrics: dict = field(default_factory=dict)
    group_metrics: dict = field(default_factory=dict)
    transfer_metrics: dict = field(default_factory=dict)
    stability_label: str = "insufficient"
    transfer_label: str = "not_applicable"
    overall_label: str = "insufficient"
    warnings: list[str] = field(default_factory=list)
    status: str = "evaluated"

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        return cls(**data)

