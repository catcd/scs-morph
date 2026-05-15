from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def _majority(series: pd.Series):
    values = series.dropna().astype(str)
    return values.value_counts().index[0] if not values.empty else None


def assign_expert_decision(row, config) -> str:
    if row.get("human_n", 0) < 2:
        return "insufficient_reviews"
    if bool(row.get("high_disagreement_flag", False)):
        return "expert_disagreement"
    accept = config["decision_rules"]["expert_accept"]
    reject = config["decision_rules"]["expert_reject"]
    action = row.get("majority_action_human")
    if row.get("human_mean_plausibility", 0) <= reject["max_mean_plausibility"] or row.get("human_mean_faithfulness", 0) <= reject["max_mean_faithfulness"] or action == "reject":
        return "expert_rejected"
    if row.get("human_mean_plausibility", 0) >= accept["min_mean_plausibility"] and row.get("human_mean_faithfulness", 0) >= accept["min_mean_faithfulness"] and row.get("human_mean_usefulness", 0) >= accept["min_mean_usefulness"]:
        return "expert_accepted" if action == "accept" else "expert_minor_revision"
    return "expert_major_revision"


def summarize_claim_scores(scores_long: pd.DataFrame, claim_metadata: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or {"decision_rules": {"high_disagreement": {"min_score_sd": 1.25}, "expert_accept": {"min_mean_plausibility": 4, "min_mean_faithfulness": 4, "min_mean_usefulness": 3.5}, "expert_reject": {"max_mean_plausibility": 2.5, "max_mean_faithfulness": 2.5}}}
    if scores_long.empty:
        return pd.DataFrame()
    wide = scores_long.pivot_table(index=["review_id", "claim_id", "reviewer_id", "reviewer_type"], columns="dimension", values="score", aggfunc="mean").reset_index()
    rows = []
    for claim_id, group in wide.groupby("claim_id"):
        human = group[group["reviewer_type"] == "human"]
        llm = group[group["reviewer_type"] == "llm"]
        original = scores_long[scores_long["claim_id"] == claim_id]
        row = {
            "claim_id": claim_id,
            "human_n": int(human["reviewer_id"].nunique()),
            "llm_n": int(llm["reviewer_id"].nunique()),
            "human_mean_plausibility": human.get("plausibility_1_5", pd.Series(dtype=float)).mean(),
            "human_mean_clarity": human.get("clarity_1_5", pd.Series(dtype=float)).mean(),
            "human_mean_faithfulness": human.get("faithfulness_1_5", pd.Series(dtype=float)).mean(),
            "human_mean_usefulness": human.get("usefulness_1_5", pd.Series(dtype=float)).mean(),
            "human_mean_confidence": human.get("reviewer_confidence_1_5", pd.Series(dtype=float)).mean(),
            "human_sd_plausibility": human.get("plausibility_1_5", pd.Series(dtype=float)).std(),
            "human_sd_faithfulness": human.get("faithfulness_1_5", pd.Series(dtype=float)).std(),
            "llm_mean_plausibility": llm.get("plausibility_1_5", pd.Series(dtype=float)).mean(),
            "llm_mean_faithfulness": llm.get("faithfulness_1_5", pd.Series(dtype=float)).mean(),
            "all_mean_plausibility": group.get("plausibility_1_5", pd.Series(dtype=float)).mean(),
            "all_mean_faithfulness": group.get("faithfulness_1_5", pd.Series(dtype=float)).mean(),
            "majority_caution_level_human": _majority(original[original["reviewer_type"] == "human"]["caution_level"]),
            "majority_action_human": _majority(original[original["reviewer_type"] == "human"]["recommended_action"]),
        }
        row["high_disagreement_flag"] = bool(max(row.get("human_sd_plausibility") or 0, row.get("human_sd_faithfulness") or 0) >= config["decision_rules"]["high_disagreement"]["min_score_sd"])
        rows.append(row)
    summary = pd.DataFrame(rows).merge(claim_metadata, on="claim_id", how="left")
    summary["expert_decision"] = summary.apply(lambda row: assign_expert_decision(row, config), axis=1)
    return summary


def summarize_by_claim_group(claim_score_summary: pd.DataFrame) -> pd.DataFrame:
    if claim_score_summary.empty:
        return pd.DataFrame()
    group_cols = ["claim_level", "claim_type", "dataset", "feature_space", "target", "overall_label", "stability_label", "transfer_label"]
    rows = []
    for keys, group in claim_score_summary.groupby(group_cols, dropna=False):
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_claims": len(group),
            "mean_plausibility": group["human_mean_plausibility"].mean(),
            "mean_faithfulness": group["human_mean_faithfulness"].mean(),
            "accept_rate": group["expert_decision"].isin(["expert_accepted", "expert_minor_revision"]).mean(),
            "reject_rate": group["expert_decision"].eq("expert_rejected").mean(),
            "high_disagreement_rate": group["high_disagreement_flag"].mean(),
        })
    return pd.DataFrame(rows)


def create_expert_review_report(scores_long, claim_summary, agreement_summary, human_llm_summary, output_path) -> None:
    lines = ["# Expert Review Report\n"]
    lines.append(f"- Number of scored claims: {claim_summary['claim_id'].nunique() if not claim_summary.empty else 0}\n")
    lines.append(f"- Reviewers included: {', '.join(sorted(scores_long['reviewer_id'].dropna().astype(str).unique())) if not scores_long.empty else 'none'}\n")
    lines.append("## Mean Scores By Dimension\n")
    if not scores_long.empty:
        lines.append(scores_long.groupby("dimension")["score"].mean().to_markdown() + "\n")
    lines.append("## Human Agreement\n")
    lines.append(agreement_summary.to_markdown(index=False) if not agreement_summary.empty else "_No agreement results._")
    lines.append("\n## Human vs LLM Agreement\n")
    lines.append(human_llm_summary.to_markdown(index=False) if not human_llm_summary.empty else "_No LLM agreement results._")
    lines.append("\n## Decision Counts\n")
    if not claim_summary.empty:
        lines.append(claim_summary["expert_decision"].value_counts().to_markdown() + "\n")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")

