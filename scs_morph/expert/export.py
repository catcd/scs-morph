from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from scs_morph.utils.io import ensure_dir


IDENTIFICATION = ["review_id", "claim_id", "claim_family_id", "review_set", "claim_level", "claim_type"]
CONTENT = ["claim_text", "short_claim_text", "target", "feature_family", "feature_space", "pipeline", "atlas", "dataset", "cohort"]
ANATOMY = ["roi", "roi_group", "anatomical_group", "network", "network_7_like", "lobe", "direction"]
EVIDENCE = ["evidence_metric", "evidence_value", "coefficient", "t_value", "p_value", "q_value", "n_samples", "n_features", "support_regions", "support_metrics_short"]
EVAL_CONTEXT = ["overall_label", "stability_label", "transfer_label", "evidence_spearman", "support_jaccard", "direction_retention_source_support", "transfer_auc_oriented", "transfer_spearman_corr", "related_experiment_ids"]
SCORES = ["reviewer_id", "reviewer_type", "reviewer_specialty", "plausibility_1_5", "clarity_1_5", "faithfulness_1_5", "usefulness_1_5", "reviewer_confidence_1_5", "caution_level", "recommended_action", "corrected_claim_text", "reviewer_comments"]


def generate_review_id(row, prefix: str = "REV") -> str:
    return f"{prefix}{int(row.name) + 1:04d}"


def _support_short(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text[:500]


def create_reviewer_table(selected_df: pd.DataFrame, reviewer_id=None, reviewer_type=None, reviewer_specialty=None, include_scores: bool = True, context: str = "full") -> pd.DataFrame:
    df = selected_df.reset_index(drop=True).copy()
    if "review_id" not in df.columns:
        df["review_id"] = df.apply(generate_review_id, axis=1)
    df["support_metrics_short"] = df.get("support_metrics", "").apply(_support_short) if "support_metrics" in df else ""
    rename = {
        "best_overall_label": "overall_label",
        "max_evidence_spearman": "evidence_spearman",
        "min_support_jaccard": "support_jaccard",
        "max_auc_oriented": "transfer_auc_oriented",
        "max_abs_transfer_corr": "transfer_spearman_corr",
    }
    for source, target in rename.items():
        if target not in df and source in df:
            df[target] = df[source]
    for col in [*IDENTIFICATION, *CONTENT, *ANATOMY, *EVIDENCE, *EVAL_CONTEXT]:
        if col not in df:
            df[col] = ""
    columns = [*IDENTIFICATION, *CONTENT, *ANATOMY, *EVIDENCE]
    if context == "full":
        columns += EVAL_CONTEXT
    if include_scores:
        df["reviewer_id"] = reviewer_id or ""
        df["reviewer_type"] = reviewer_type or ""
        df["reviewer_specialty"] = reviewer_specialty or ""
        for col in SCORES:
            if col not in df:
                df[col] = ""
        columns += SCORES
    return df[columns]


def _write(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(str(path.parent))
    df.to_csv(path, index=False)
    return path


def export_review_files(main_df: pd.DataFrame, llm_df: pd.DataFrame, reserve_df: pd.DataFrame, config: dict) -> list[Path]:
    paths = config["paths"]
    to_send = ensure_dir(paths["to_send_dir"])
    processed = ensure_dir(paths["output_dir"])
    written: list[Path] = []
    for name, df in [
        ("selected_main_expert_claims.csv", main_df),
        ("selected_llm_extended_claims.csv", llm_df),
        ("selected_reserve_claims.csv", reserve_df),
        ("selection_candidates_enriched.csv", config.get("_candidates", pd.DataFrame())),
    ]:
        written.append(_write(df, Path(processed) / name))

    exports = [
        ("expert_review_main_master_full_context.csv", main_df, "full"),
        ("expert_review_main_master_minimal_context.csv", main_df, "minimal"),
        ("expert_review_llm_extended_full_context.csv", llm_df, "full"),
        ("expert_review_llm_extended_minimal_context.csv", llm_df, "minimal"),
        ("expert_review_reserve_full_context.csv", reserve_df, "full"),
        ("expert_review_reserve_minimal_context.csv", reserve_df, "minimal"),
    ]
    for filename, df, context in exports:
        written.append(_write(create_reviewer_table(df, context=context), Path(to_send) / filename))

    for reviewer in config["reviewers"]["humans"]:
        table = create_reviewer_table(main_df, reviewer["reviewer_id"], reviewer["reviewer_type"], reviewer["specialty"], context="minimal")
        written.append(_write(table, Path(to_send) / f"{reviewer['reviewer_id']}_expert_review_main_minimal_context.csv"))
    for reviewer in config["reviewers"]["llms"]:
        main = create_reviewer_table(main_df, reviewer["reviewer_id"], reviewer["reviewer_type"], reviewer["specialty"], context="minimal")
        ext = create_reviewer_table(llm_df, reviewer["reviewer_id"], reviewer["reviewer_type"], reviewer["specialty"], context="full")
        written.append(_write(main, Path(to_send) / f"{reviewer['reviewer_id']}_review_main_minimal_context.csv"))
        written.append(_write(ext, Path(to_send) / f"{reviewer['reviewer_id']}_review_extended_full_context.csv"))

    readme = Path(to_send) / "expert_review_README.md"
    readme.write_text("Please score each claim using the blank columns. Detailed scoring guidelines will be provided separately.\n", encoding="utf-8")
    written.append(readme)
    return written

