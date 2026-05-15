from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.expert.agreement import compute_agreement_by_dimension, compute_human_llm_agreement, pairwise_reviewer_correlations
from scs_morph.expert.analysis import create_expert_review_report, summarize_by_claim_group, summarize_claim_scores
from scs_morph.utils.io import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expert_review.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    analyzed = Path(config["paths"]["analyzed_dir"])
    long_path = analyzed / "expert_scores_long.csv"
    if not long_path.exists() or pd.read_csv(long_path).empty:
        print("No returned review files found. Run export_expert_review.py and collect completed review CSVs first.")
        return
    scores_long = pd.read_csv(long_path)
    claim_meta = pd.read_csv(Path(config["paths"]["output_dir"]) / "selected_llm_extended_claims.csv")
    summary = summarize_claim_scores(scores_long, claim_meta, config)
    agreement = compute_agreement_by_dimension(scores_long, "human")
    human_llm = compute_human_llm_agreement(scores_long)
    wide = pd.read_csv(analyzed / "expert_scores_wide.csv") if (analyzed / "expert_scores_wide.csv").exists() else pd.DataFrame()
    pairwise = pairwise_reviewer_correlations(wide, config["scoring"]["required_score_columns"]) if not wide.empty else pd.DataFrame()
    group_summary = summarize_by_claim_group(summary)
    ensure_dir(str(analyzed))
    summary.to_csv(analyzed / "claim_review_summary.csv", index=False)
    agreement.to_csv(analyzed / "agreement_by_dimension.csv", index=False)
    pairwise.to_csv(analyzed / "pairwise_reviewer_correlations.csv", index=False)
    human_llm.to_csv(analyzed / "human_llm_agreement.csv", index=False)
    group_summary.to_csv(analyzed / "review_group_summary.csv", index=False)
    summary[summary["expert_decision"].isin(["expert_accepted", "expert_minor_revision"])].to_csv(analyzed / "accepted_claims.csv", index=False)
    summary[summary["expert_decision"].eq("expert_rejected")].to_csv(analyzed / "rejected_claims.csv", index=False)
    summary[summary["high_disagreement_flag"]].to_csv(analyzed / "high_disagreement_claims.csv", index=False)
    create_expert_review_report(scores_long, summary, agreement, human_llm, "outputs/expert_review_report.md")
    print(f"Analyzed claims: {summary['claim_id'].nunique() if not summary.empty else 0}")


if __name__ == "__main__":
    main()

