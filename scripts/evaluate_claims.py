from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.evaluation.evaluator import evaluate_pairs
from scs_morph.evaluation.loaders import load_claim_families
from scs_morph.evaluation.pairs import EvaluationPair
from scs_morph.evaluation.serialization import save_evaluation_results
from scs_morph.utils.io import ensure_dir, load_yaml


CLAIM_SPECS = {
    "dx_cn_ad": ("diagnosis_contrast", "diagnosis_binary_ad_cn", "AD", "CN"),
    "dx_cn_mci": ("diagnosis_contrast", "diagnosis_binary_mci_cn", "MCI", "CN"),
    "mmse": ("mmse_association", "mmse", None, None),
    "age": ("age_association", "age", None, None),
    "cdr": ("cdr_impairment_contrast", "diagnosis_binary_impairment_cdr", "CDR_impairment", "CDR_0"),
    "ad_to_cdr": ("diagnosis_contrast", "diagnosis_binary_ad_cn", "CDR_impairment", "CDR_0"),
}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", value.upper()).strip("_")


def _setting_id(dataset: str, feature_space: str, claim_key: str, claim_type: str | None = None, target: str | None = None) -> str:
    if dataset == "ADNI_SPM_MATCHED" and feature_space == "fs8_thickness_schaefer200_7":
        special = {
            "dx_cn_ad": "DX_CN_AD",
            "dx_cn_mci": "DX_CN_MCI",
            "mmse": "MMSE",
            "age": "AGE",
            "cdr": "CDR_IMPAIRMENT",
            "ad_to_cdr": "DX_CN_AD",
        }
        return f"ADNI_SPM_MATCHED_FS8_SCHAEFER7_{special.get(claim_key, _safe(target or claim_type or claim_key))}"
    prefix = _safe(dataset)
    feature = _safe(feature_space)
    if claim_key == "dx_cn_ad":
        suffix = "DX_CN_AD"
    elif claim_key == "dx_cn_mci":
        suffix = "DX_CN_MCI"
    elif claim_key == "mmse":
        suffix = "MMSE"
    elif claim_key == "age":
        suffix = "AGE"
    elif claim_key == "cdr":
        suffix = "CDR_IMPAIRMENT"
    elif claim_key == "ad_to_cdr":
        suffix = "DX_CN_AD"
    else:
        suffix = _safe(target or claim_type or claim_key)
    return f"{prefix}_{feature}_{suffix}"


def _claim_for_dataset(claim_key: str, dataset: str, as_target: bool = False) -> tuple[str, str, str | None, str | None, str]:
    if claim_key == "ad_to_cdr" and as_target:
        claim_type, target, pos, neg = CLAIM_SPECS["cdr"]
        return claim_type, target, pos, neg, "cdr"
    claim_type, target, pos, neg = CLAIM_SPECS[claim_key]
    if dataset.startswith("OASIS1") and claim_key in {"dx_cn_ad", "dx_cn_mci"}:
        claim_type, target, pos, neg = CLAIM_SPECS["cdr"]
        return claim_type, target, pos, neg, "cdr"
    return claim_type, target, pos, neg, claim_key


def _target_type(target: str | None) -> str:
    if target in {"mmse", "age"}:
        return "continuous"
    if target:
        return "binary"
    return "none"


def _pair(
    experiment_id: str,
    experiment_type: str,
    source_dataset: str,
    target_dataset: str,
    source_feature_space: str,
    target_feature_space: str,
    claim_key: str,
    strategy: str,
    index: int,
) -> EvaluationPair:
    source_claim_type, source_target, pos, neg, source_setting_key = _claim_for_dataset(claim_key, source_dataset, as_target=False)
    target_claim_type, target_target, target_pos, target_neg, target_setting_key = _claim_for_dataset(claim_key, target_dataset, as_target=True)
    positive = target_pos or pos
    negative = target_neg or neg
    pair_id = _safe(f"{experiment_id}_{index:03d}_{source_dataset}_{source_feature_space}_TO_{target_dataset}_{target_feature_space}_{claim_key}")
    return EvaluationPair(
        pair_id=pair_id,
        experiment_id=experiment_id,
        experiment_type="weak_label_transfer" if claim_key == "ad_to_cdr" else experiment_type,
        source_dataset=source_dataset,
        source_feature_space=source_feature_space,
        source_setting_id=_setting_id(source_dataset, source_feature_space, source_setting_key, source_claim_type, source_target),
        source_claim_type=source_claim_type,
        source_target=source_target,
        source_filter=None,
        target_dataset=target_dataset,
        target_feature_space=target_feature_space,
        target_setting_id=_setting_id(target_dataset, target_feature_space, target_setting_key, target_claim_type, target_target),
        target_claim_type=target_claim_type,
        target_target=target_target,
        target_filter=None,
        roi_match_strategy=strategy,
        transfer_target=target_target,
        transfer_target_type=_target_type(target_target),
        positive_label=positive,
        negative_label=negative,
    )


def expand_pairs(config: dict) -> list[EvaluationPair]:
    pairs: list[EvaluationPair] = []
    for group in config.get("pair_groups", []):
        experiment_id = group["experiment_id"]
        experiment_type = group.get("experiment_type", "cross_dataset_transfer")
        start = len(pairs) + 1
        if group["group_id"] in {"E1", "E2"}:
            for fs in group["feature_spaces"]:
                for claim in group["claims"]:
                    pairs.append(_pair(experiment_id, experiment_type, group["source_dataset"], group["target_dataset"], fs, fs, claim, group["roi_match_strategy"], len(pairs) + 1))
        elif group["group_id"] in {"E3", "E4"}:
            dataset = group["dataset"]
            for source_fs, target_fs, strategy in group["feature_pairs"]:
                for claim in group["claims"]:
                    pairs.append(_pair(experiment_id, experiment_type, dataset, dataset, source_fs, target_fs, claim, strategy, len(pairs) + 1))
        elif group["group_id"] in {"E5", "E6"}:
            for dataset in group["datasets"]:
                claims = group["claims_by_dataset"][dataset]
                for source_fs, target_fs, strategy in group["feature_pairs"]:
                    for claim in claims:
                        pairs.append(_pair(experiment_id, experiment_type, dataset, dataset, source_fs, target_fs, claim, strategy, len(pairs) + 1))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation_pairs.yaml")
    args = parser.parse_args()

    paths = load_yaml("configs/paths.yaml")
    pair_config = load_yaml(args.config)
    thresholds = load_yaml("configs/evaluation_thresholds.yaml")
    output_dir = Path(pair_config.get("output_dir", "data/processed/evaluations"))
    ensure_dir(str(output_dir))

    pairs = expand_pairs(pair_config)
    pd.DataFrame([pair.to_dict() for pair in pairs]).to_csv(output_dir / "evaluation_pairs_expanded.csv", index=False)
    families = load_claim_families(str(Path(paths.get("processed_dir", "data/processed")) / "claim_cards" / "claim_families.jsonl"))
    results = evaluate_pairs(pairs, families, paths, thresholds)
    output_paths = save_evaluation_results(results, str(output_dir))

    evaluated = sum(r.status == "evaluated" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    errors = sum(r.status == "error" for r in results)
    print(f"Total pairs expanded: {len(pairs)}")
    print(f"Evaluated pairs: {evaluated}")
    print(f"Skipped pairs: {skipped}")
    print(f"Errors: {errors}")
    print("Counts by experiment_id:")
    print(pd.Series([r.experiment_id for r in results]).value_counts().to_string())
    print("Counts by overall_label:")
    print(pd.Series([r.overall_label for r in results]).value_counts().to_string())
    print("Output files:")
    for path in output_paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
