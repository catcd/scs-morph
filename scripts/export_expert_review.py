from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scs_morph.expert.export import create_reviewer_table, export_review_files
from scs_morph.expert.llm import create_llm_response_template, export_llm_prompt_batches
from scs_morph.expert.selection import (
    create_selection_report,
    enrich_claims_with_evaluation,
    load_review_inputs,
    save_selection_report,
    select_llm_extended_claims,
    select_main_expert_claims,
    select_reserve_claims,
)
from scs_morph.utils.io import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expert_review.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    claims, families, evaluation = load_review_inputs(config)
    enriched = enrich_claims_with_evaluation(claims, evaluation)
    config["_candidates"] = enriched
    main = select_main_expert_claims(enriched, config)
    llm = select_llm_extended_claims(enriched, main, config)
    reserve = select_reserve_claims(enriched, main, llm, config)
    written = export_review_files(main, llm, reserve, config)
    report = create_selection_report(main, llm, reserve, enriched)
    to_send = Path(config["paths"]["to_send_dir"])
    save_selection_report(report, str(to_send / "expert_review_selection_report.csv"), str(to_send / "expert_review_selection_report.md"))
    llm_table = create_reviewer_table(llm, context="full")
    export_llm_prompt_batches(llm_table, to_send / "llm_batches", batch_size=20)
    create_llm_response_template(llm_table, to_send / "llm_response_template.csv")

    print(f"Candidate claims: {len(enriched)}")
    print(f"Selected main expert: {len(main)}")
    print(f"Selected LLM extended: {len(llm)}")
    print(f"Selected reserve: {len(reserve)}")
    print("Composition by claim_level:")
    print(main["claim_level"].value_counts().to_string())
    print("Composition by claim_type:")
    print(main["claim_type"].value_counts().to_string())
    print("Composition by dataset:")
    print(main["dataset"].value_counts().to_string())
    print("Exported files:")
    for path in sorted(to_send.glob("*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()

