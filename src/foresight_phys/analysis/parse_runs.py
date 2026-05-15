"""Parse benchmark result JSON files into a tidy predictions parquet.

Each run directory under ``docs/<run-name>/`` is expected to contain a
``benchmark_results.json`` file. The per-field rows come from the saved
``per_file[].report_experiments[]`` payload written by the benchmark CLI and
include the per-field uncertainty / scoring values from
``metrics.build_experiment_report``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import AnalysisPaths, default_paths


def _format_report_value(value: Any) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _serialize_optional_mapping(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def parse_summary(model: str, run_name: str, summary: dict[str, Any]) -> list[dict]:
    rows: list[dict] = []
    per_file = summary.get("per_file", [])
    if not isinstance(per_file, list):
        return rows

    for item in per_file:
        if not isinstance(item, dict):
            continue
        paper_title = str(item.get("paper_title") or "")
        file_id = str(item.get("file") or "")
        report_experiments = item.get("report_experiments", [])
        if not isinstance(report_experiments, list):
            continue

        for exp_idx, experiment_summary in enumerate(report_experiments):
            if not isinstance(experiment_summary, dict):
                continue
            exp_desc = str(experiment_summary.get("experiment_description") or "")
            result_rows = experiment_summary.get("result_rows", [])
            if not isinstance(result_rows, list):
                continue

            for row in result_rows:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "run_name": run_name,
                    "model": model,
                    "file": file_id,
                    "paper_title": paper_title,
                    "experiment": exp_idx,
                    "experiment_description": exp_desc,
                    "key": str(row.get("result_key") or ""),
                    "description": str(row.get("description") or ""),
                    "type": str(row.get("type") or ""),
                    "gt": _format_report_value(row.get("ground_truth")),
                    "pred": _format_report_value(row.get("predicted")),
                    "status_class": str(row.get("status_class") or "status-unknown"),
                    "status_text": str(row.get("status_text") or ""),
                    "distribution": row.get("distribution"),
                    "sigma": row.get("sigma"),
                    "z": row.get("z"),
                    "nll": row.get("nll"),
                    "quality": row.get("quality"),
                    "prob_true": row.get("prob_true"),
                    "probabilities_json": _serialize_optional_mapping(row.get("probabilities")),
                    "confidence": row.get("confidence"),
                    "equivalent": row.get("equivalent"),
                    "log_loss": row.get("log_loss"),
                })
    return rows


def discover_run_dirs(docs_dir: Path) -> list[Path]:
    """Find run directories: docs/<name>/ that contain a results JSON."""
    run_dirs: list[Path] = []
    for child in sorted(docs_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "benchmark_results.json").exists():
            run_dirs.append(child)
    return run_dirs


def parse_all_runs(paths: AnalysisPaths | None = None) -> pd.DataFrame:
    """Parse every run directory found under ``docs/`` into a single dataframe."""
    paths = paths or default_paths()
    paths.ensure_dirs()

    run_dirs = discover_run_dirs(paths.docs_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No benchmark run directories found under {paths.docs_dir}. "
            "Expected docs/<run-name>/benchmark_results.json."
        )

    all_rows: list[dict] = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "benchmark_results.json").read_text(encoding="utf-8"))
        model = summary.get("model") or run_dir.name
        all_rows.extend(parse_summary(
            model=model,
            run_name=run_dir.name,
            summary=summary,
        ))

    df = pd.DataFrame(all_rows)
    df.to_parquet(paths.predictions_parquet, index=False)
    return df
