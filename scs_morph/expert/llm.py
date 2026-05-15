from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

from scs_morph.expert.export import SCORES
from scs_morph.utils.io import ensure_dir


def export_llm_prompt_batches(review_df: pd.DataFrame, output_dir, batch_size: int = 20, context: str = "minimal") -> list[Path]:
    output = ensure_dir(output_dir)
    paths = []
    n_batches = math.ceil(len(review_df) / batch_size) if len(review_df) else 0
    for i in range(n_batches):
        batch = review_df.iloc[i * batch_size : (i + 1) * batch_size]
        lines = [
            f"# LLM Expert Review Batch {i + 1:03d}\n",
            "Score each claim using the requested CSV-compatible columns. Return CSV-compatible rows only in the final response.\n",
            "| review_id | claim_id | claim_text | target | feature_space |\n",
            "| --- | --- | --- | --- | --- |\n",
        ]
        for _, row in batch.iterrows():
            text = str(row.get("claim_text", "")).replace("|", "/")
            lines.append(f"| {row.get('review_id')} | {row.get('claim_id')} | {text} | {row.get('target', '')} | {row.get('feature_space', '')} |\n")
        path = Path(output) / f"llm_batch_{i + 1:03d}.md"
        path.write_text("".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def create_llm_response_template(review_df: pd.DataFrame, output_path) -> Path:
    cols = [c for c in review_df.columns if c in {"review_id", "claim_id", "claim_family_id", "claim_text"}] + SCORES
    template = review_df.copy()
    for col in SCORES:
        if col not in template:
            template[col] = ""
        elif col not in {"reviewer_id", "reviewer_type", "reviewer_specialty"}:
            template[col] = ""
    path = Path(output_path)
    ensure_dir(str(path.parent))
    template[cols].to_csv(path, index=False)
    return path

