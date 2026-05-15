from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class AnalysisSetting:
    setting_id: str
    dataset: str
    cohort: str | None
    feature_space: str
    feature_family: str | None
    pipeline: str | None
    atlas: str | None
    target: str | None
    claim_type: str
    covariates: list[str]
    inclusion_filter: str | None = None
    positive_label: str | int | float | None = None
    negative_label: str | int | float | None = None
    top_k: int = 20
    standardize_features: bool = True
    standardize_target: bool = False
    min_n: int = 30
    generate_composite_claim: bool = True
    generate_atomic_claims: bool = True
    generate_group_claims: bool = True
    generate_ranking_claim: bool = True
    generate_direction_claim: bool = True
    criteria: dict = field(default_factory=dict)
    n_components: int = 3
    metadata_association_targets: list[str] = field(default_factory=list)
    reference_filter: str | None = None
    reference_group_label: str | None = None
    target_group_column: str | None = None
    target_group_values: list | None = None
    z_threshold: float = 1.5
    output_roi_statistics: bool = True


@dataclass
class SourceSetting:
    setting_id: str
    dataset: str
    cohort: str | None
    feature_space: str
    target: str | None
    claim_type: str
    covariates: list[str] = field(default_factory=list)
    inclusion_filter: str | None = None
    positive_label: str | int | float | None = None
    negative_label: str | int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_setting_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", value.upper()).strip("_")


def setting_from_dict(d: dict[str, Any]) -> AnalysisSetting:
    return AnalysisSetting(
        setting_id=_safe_setting_id(str(d["setting_id"])),
        dataset=str(d["dataset"]),
        cohort=d.get("cohort", "pooled"),
        feature_space=str(d["feature_space"]),
        feature_family=d.get("feature_family"),
        pipeline=d.get("pipeline"),
        atlas=d.get("atlas"),
        target=d.get("target"),
        claim_type=str(d["claim_type"]),
        covariates=list(d.get("covariates", [])),
        inclusion_filter=d.get("inclusion_filter"),
        positive_label=d.get("positive_label"),
        negative_label=d.get("negative_label"),
        top_k=int(d.get("top_k", 20)),
        standardize_features=bool(d.get("standardize_features", True)),
        standardize_target=bool(d.get("standardize_target", False)),
        min_n=int(d.get("min_n", 30)),
        generate_composite_claim=bool(d.get("generate_composite_claim", True)),
        generate_atomic_claims=bool(d.get("generate_atomic_claims", True)),
        generate_group_claims=bool(d.get("generate_group_claims", True)),
        generate_ranking_claim=bool(d.get("generate_ranking_claim", True)),
        generate_direction_claim=bool(d.get("generate_direction_claim", True)),
        criteria=dict(d.get("criteria", {})),
        n_components=int(d.get("n_components", 3)),
        metadata_association_targets=list(d.get("metadata_association_targets", [])),
        reference_filter=d.get("reference_filter"),
        reference_group_label=d.get("reference_group_label"),
        target_group_column=d.get("target_group_column"),
        target_group_values=d.get("target_group_values"),
        z_threshold=float(d.get("z_threshold", 1.5)),
        output_roi_statistics=bool(d.get("output_roi_statistics", True)),
    )


def setting_to_dict(setting: AnalysisSetting) -> dict[str, Any]:
    return asdict(setting)


def source_setting_from_analysis(setting: AnalysisSetting) -> SourceSetting:
    return SourceSetting(
        setting_id=setting.setting_id,
        dataset=setting.dataset,
        cohort=setting.cohort,
        feature_space=setting.feature_space,
        target=setting.target,
        claim_type=setting.claim_type,
        covariates=list(setting.covariates),
        inclusion_filter=setting.inclusion_filter,
        positive_label=setting.positive_label,
        negative_label=setting.negative_label,
    )


def expand_settings_from_config(config: dict[str, Any], feature_spaces: dict | None = None) -> list[AnalysisSetting]:
    settings: list[AnalysisSetting] = []
    raw_settings = config.get("settings") or config.get("claims") or []
    for item in raw_settings:
        feature_space_list = item.get("feature_spaces") or [item.get("feature_space")]
        for feature_space in feature_space_list:
            expanded = dict(item)
            expanded.pop("feature_spaces", None)
            expanded["feature_space"] = feature_space
            if "{FEATURE}" in str(expanded.get("setting_id", "")):
                expanded["setting_id"] = str(expanded["setting_id"]).replace("{FEATURE}", str(feature_space).upper())
            if feature_spaces and feature_space in feature_spaces:
                fs = feature_spaces[feature_space]
                expanded.setdefault("feature_family", fs.family)
                expanded.setdefault("pipeline", fs.pipeline)
                expanded.setdefault("atlas", fs.atlas)
            settings.append(setting_from_dict(expanded))
    return settings

