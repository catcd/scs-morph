from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.analysis.design_matrix import prepare_analysis_table
from scs_morph.analysis.filters import apply_cohort_filter, apply_inclusion_filter
from scs_morph.analysis.settings import AnalysisSetting, expand_settings_from_config, setting_to_dict
from scs_morph.claims.base import get_extractor_for_claim_type
from scs_morph.claims.filters import ClaimCriteria
from scs_morph.claims.serialization import (
    save_claim_cards_jsonl,
    save_claim_cards_summary_csv,
    save_claim_families_jsonl,
    save_claim_family_summary_csv,
)
from scs_morph.features.feature_space import load_feature_spaces
from scs_morph.utils.io import ensure_dir, load_yaml, read_table
from scs_morph.utils.logging import get_logger


DATASET_ALIASES = {"ADNI_SPM_MATCHED": "ADNI_SPM"}


def _load_table_pair(processed_dir: Path, dataset: str, feature_space: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder_dataset = DATASET_ALIASES.get(dataset, dataset)
    matrix_dir = processed_dir / "feature_matrices" / folder_dataset / feature_space
    x_path = matrix_dir / "X.parquet"
    meta_path = matrix_dir / "meta.parquet"
    if not x_path.exists():
        x_path = matrix_dir / "X.csv"
    if not meta_path.exists():
        meta_path = matrix_dir / "meta.csv"
    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing feature matrix files in {matrix_dir}")
    return read_table(str(x_path)), read_table(str(meta_path))


def _create_derived_targets(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if "sex" in out.columns:
        sex = out["sex"].astype(str).str.upper()
        out["sex_binary_male_female"] = sex.map({"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0})
    if "field_strength" in out.columns:
        fs = out["field_strength"].astype(str).str.upper().str.replace(" ", "", regex=False)
        out["field_strength_binary_3t_15t"] = fs.map({"3T": 1.0, "1.5T": 0.0, "1.5TESLA": 0.0})
    if "cohort" in out.columns:
        cohort = out["cohort"].astype(str).str.upper().str.replace(" ", "", regex=False)
        out["cohort_binary_adni3_adni1"] = cohort.map({"ADNI3": 1.0, "ADNI1": 0.0})
    return out


def _filter_rows(X: pd.DataFrame, meta: pd.DataFrame, setting: AnalysisSetting) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = apply_cohort_filter(meta, setting.cohort)
    mask &= apply_inclusion_filter(meta, setting.inclusion_filter)
    values = mask.to_numpy(dtype=bool)
    return X.loc[values].copy(), meta.loc[values].copy()


def _drop_unusable_covariates(meta: pd.DataFrame, setting: AnalysisSetting) -> tuple[AnalysisSetting, list[str]]:
    warnings: list[str] = []
    kept = []
    for covariate in setting.covariates:
        if covariate not in meta.columns:
            warnings.append(f"Dropped missing covariate: {covariate}")
            continue
        present_fraction = float(meta[covariate].notna().mean())
        if present_fraction < 0.2:
            warnings.append(f"Dropped mostly missing covariate: {covariate} ({present_fraction:.1%} present)")
            continue
        kept.append(covariate)
    updated = AnalysisSetting(**{**setting_to_dict(setting), "covariates": kept})
    return updated, warnings


def _setting_min_n_count(meta: pd.DataFrame, setting: AnalysisSetting) -> int:
    required = [c for c in [setting.target, *setting.covariates] if c is not None]
    if not required:
        return len(meta)
    present = [column for column in required if column in meta.columns]
    if len(present) != len(required):
        return 0
    return int(meta[present].dropna().shape[0])


def _merge_criteria(generation_config: dict, setting: AnalysisSetting) -> ClaimCriteria:
    defaults = dict(generation_config.get("default_criteria", {}))
    overrides = generation_config.get("feature_space_overrides", {}).get(setting.feature_space, {})
    merged = {**defaults, **overrides, **setting.criteria, "min_n": setting.min_n, "top_k": setting.top_k}
    return ClaimCriteria.from_dict(merged)


def _save_setting_statistics(setting: AnalysisSetting, stats: pd.DataFrame | None, per_setting_dir: Path) -> str | None:
    if stats is None or stats.empty or not setting.output_roi_statistics:
        return None
    if setting.claim_type == "latent_morphometry_axis":
        path = per_setting_dir / f"{setting.setting_id}_latent_loadings.csv"
    elif setting.claim_type == "normative_deviation":
        path = per_setting_dir / f"{setting.setting_id}_normative_z_statistics.csv"
    else:
        path = per_setting_dir / f"{setting.setting_id}_roi_statistics.csv"
    stats.to_csv(path, index=False)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract full claim families from feature matrices.")
    parser.add_argument("--config", default="configs/claim_extraction_all_main.yaml")
    parser.add_argument("--generation-config", default="configs/claim_generation.yaml")
    args = parser.parse_args()

    logger = get_logger("extract_claims")
    paths = load_yaml("configs/paths.yaml")
    feature_spaces = load_feature_spaces("configs/feature_spaces.yaml")
    spm17 = feature_spaces.get("spm_thickness_schaefer200_17")
    if spm17 is None or spm17.feature_names_file != "schaefer200_17network_feats.txt":
        raise ValueError("spm_thickness_schaefer200_17 must use schaefer200_17network_feats.txt")

    generation_config = load_yaml(args.generation_config)
    extraction_config = load_yaml(args.config)
    settings = expand_settings_from_config(extraction_config, feature_spaces)
    processed_dir = Path(paths["processed_dir"])
    output_dir = Path(extraction_config.get("output_dir", processed_dir / "claim_cards"))
    per_setting_dir = output_dir / "per_setting"
    ensure_dir(str(per_setting_dir))

    feature_family_labels = generation_config.get("feature_family_labels", {})
    families = []
    cards = []
    skipped = []

    logger.info("Expanded %d settings", len(settings))
    for setting in settings:
        try:
            X, meta = _load_table_pair(processed_dir, setting.dataset, setting.feature_space)
            meta = _create_derived_targets(meta)
            X, meta = _filter_rows(X, meta, setting)
            setting, warnings = _drop_unusable_covariates(meta, setting)
            if setting.target is not None and setting.target not in meta.columns:
                raise KeyError(f"Missing target column: {setting.target}")
            n_count = _setting_min_n_count(meta, setting)
            if n_count < setting.min_n:
                raise ValueError(f"n={n_count} below min_n={setting.min_n}")
            if setting.target is not None:
                prepare_analysis_table(X, meta, setting.target, setting.covariates)

            criteria = _merge_criteria(generation_config, setting)
            extractor = get_extractor_for_claim_type(setting.claim_type, criteria, feature_family_labels)
            family, stats = extractor.extract_family(X, meta, setting)
            family.warnings.extend(warnings)
            family.roi_statistics_path = _save_setting_statistics(setting, stats, per_setting_dir)
            families.append(family)
            cards.extend(family.all_claims)
            print(
                f"{setting.setting_id} | {setting.dataset} | {setting.feature_space} | {setting.claim_type} | "
                f"n={family.extraction_summary.get('n_samples')} | features={family.extraction_summary.get('n_features')} | "
                f"composite={len(family.composite_claims)} atomic={len(family.atomic_claims)} group={len(family.group_claims)} "
                f"ranking={len(family.ranking_claims)} latent={len(family.latent_claims)} normative={len(family.normative_claims)}"
            )
        except Exception as exc:
            reason = str(exc)
            skipped.append(
                {
                    "setting_id": setting.setting_id,
                    "dataset": setting.dataset,
                    "feature_space": setting.feature_space,
                    "claim_type": setting.claim_type,
                    "target": setting.target,
                    "reason": reason,
                    "details": setting_to_dict(setting),
                }
            )
            print(f"{setting.setting_id} | {setting.dataset} | {setting.feature_space} | {setting.claim_type} | skipped={reason}")

    save_claim_families_jsonl(families, str(output_dir / "claim_families.jsonl"))
    save_claim_cards_jsonl(cards, str(output_dir / "claim_cards.jsonl"))
    save_claim_cards_summary_csv(cards, str(output_dir / "claim_cards_summary.csv"))
    save_claim_family_summary_csv(families, str(output_dir / "claim_family_summary.csv"))
    skipped_columns = ["setting_id", "dataset", "feature_space", "claim_type", "target", "reason", "details"]
    pd.DataFrame(skipped, columns=skipped_columns).to_csv(output_dir / "skipped_settings.csv", index=False)
    logger.info("Generated %d claim families and %d claim cards", len(families), len(cards))


if __name__ == "__main__":
    main()
