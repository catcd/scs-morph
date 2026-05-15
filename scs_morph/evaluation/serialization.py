from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scs_morph.evaluation.pairs import EvaluationResult
from scs_morph.utils.io import ensure_dir


def flatten_metric_dicts(result_dict: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in result_dict.items():
        if isinstance(value, dict):
            for metric_key, metric_value in value.items():
                flat[f"{key}.{metric_key}"] = metric_value
        elif isinstance(value, list):
            flat[key] = "; ".join(str(item) for item in value)
        else:
            flat[key] = value
    return flat


def evaluation_results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([flatten_metric_dicts(result.to_dict()) for result in results])


def _metric_frame(results: list[EvaluationResult], metric_name: str) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = getattr(result, metric_name)
        if metrics:
            rows.append({"pair_id": result.pair_id, "experiment_id": result.experiment_id, **metrics})
    return pd.DataFrame(rows)


def save_evaluation_results(results: list[EvaluationResult], output_dir: str) -> dict[str, Path]:
    output = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    frames = {
        "evaluation_results.csv": evaluation_results_to_dataframe(results),
        "evidence_stability.csv": _metric_frame(results, "evidence_metrics"),
        "support_stability.csv": _metric_frame(results, "support_metrics"),
        "direction_stability.csv": _metric_frame(results, "direction_metrics"),
        "magnitude_stability.csv": _metric_frame(results, "magnitude_metrics"),
        "group_stability.csv": _metric_frame(results, "group_metrics"),
        "transfer_metrics.csv": _metric_frame(results, "transfer_metrics"),
    }
    skipped = [result for result in results if result.status != "evaluated"]
    frames["skipped_pairs.csv"] = pd.DataFrame([
        {
            "pair_id": r.pair_id,
            "experiment_id": r.experiment_id,
            "experiment_type": r.experiment_type,
            "source_dataset": r.source_dataset,
            "target_dataset": r.target_dataset,
            "source_feature_space": r.source_feature_space,
            "target_feature_space": r.target_feature_space,
            "reason": "; ".join(r.warnings),
        }
        for r in skipped
    ], columns=["pair_id", "experiment_id", "experiment_type", "source_dataset", "target_dataset", "source_feature_space", "target_feature_space", "reason"])
    base = evaluation_results_to_dataframe(results)
    if base.empty:
        summary = pd.DataFrame()
    else:
        summary = base.groupby(["experiment_id", "experiment_type", "status", "overall_label"], dropna=False).size().reset_index(name="n_pairs")
    frames["evaluation_summary.csv"] = summary
    for name, frame in frames.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths

