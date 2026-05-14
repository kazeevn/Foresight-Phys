"""Join parsed HTML predictions with ground-truth JSON metadata.

Produces a long dataframe with one row per (model, file, experiment, key). Adds:
- ``result_type`` from the ground truth JSON
- ``numeric_gt`` / ``numeric_pred`` (when parseable as floats)
- ``smape`` / ``normalized_smape_score`` / ``score`` (correctness in [0, 1])
- ``correct`` (binary at the same 0.5 threshold the analysis uses)
- ``leak`` (literal ground-truth value appears in description text)
- ``likely_unit_off`` (factor 1e3/1e6 ratio between pred and gt)
"""
from __future__ import annotations

import json
import math
import re

import pandas as pd

from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
CORRECT_THRESHOLD = 0.5
UNIT_FACTORS = (1e-6, 1e-3, 1e3, 1e6)


def _load_ground_truth(paths: AnalysisPaths) -> pd.DataFrame:
    rows: list[dict] = []
    for jp in sorted(paths.json_dir.glob("*.json")):
        data = json.loads(jp.read_text())
        if not data:
            continue
        for ei, exp in enumerate(data):
            exp_desc = exp["experiment_description"]
            for k, v in exp["experiment_results"].items():
                rows.append({
                    "file_id": jp.stem,
                    "experiment": ei,
                    "key": k,
                    "type": v.get("type"),
                    "gt_value": v.get("result"),
                    "result_description": v.get("description", ""),
                    "experiment_description": exp_desc,
                })
    return pd.DataFrame(rows)


def _parse_numeric(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "n/a", "na"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        m = re.match(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None


def _compute_smape(a, b) -> float | None:
    if a is None or b is None:
        return None
    d = abs(a) + abs(b)
    if d == 0:
        return 0.0
    return 2.0 * abs(a - b) / d


def _score_row(row) -> float | None:
    """Replicate the per-field scoring logic in :func:`metrics.compute_experiment_metrics`."""
    t = row["type"]
    if t in NUMERIC_TYPES or (
        t is None and isinstance(row["gt_value"], (int, float))
    ):
        if row["numeric_gt"] is None:
            return None
        if row["numeric_gt"] == 0.0:
            return 1.0 if row["numeric_pred"] == 0.0 else 0.0
        if row["numeric_pred"] is None:
            return 0.0
        return 1.0 - min(row["smape"], 1.0)
    return 1.0 if row["status_class"] == "status-match" else 0.0


def _leak(row) -> bool:
    t = row["type"]
    if t not in NUMERIC_TYPES or row["numeric_gt"] is None:
        return False
    gt = row["gt_value"]
    s = str(gt)
    if isinstance(gt, float) and gt.is_integer():
        s = str(int(gt))
    return s in (row["experiment_description"] or "") or s in (row["result_description"] or "")


def _unit_off(row) -> bool:
    gt = row["numeric_gt"]
    pr = row["numeric_pred"]
    if gt is None or pr is None or gt == 0 or pr == 0:
        return False
    ratio = abs(pr / gt)
    return any(0.5 < ratio / f < 2 for f in UNIT_FACTORS)


def build_scored_dataset(paths: AnalysisPaths | None = None) -> pd.DataFrame:
    paths = paths or default_paths()
    paths.ensure_dirs()

    predictions = pd.read_parquet(paths.predictions_parquet)
    predictions = predictions.copy()
    predictions["file_id"] = predictions["file"].str.replace(".json", "", regex=False)

    ground_truth = _load_ground_truth(paths)

    df = predictions.merge(
        ground_truth[[
            "file_id", "experiment", "key", "type", "gt_value",
            "result_description", "experiment_description",
        ]],
        on=["file_id", "experiment", "key"],
        how="left",
        suffixes=("_html", ""),
    )

    df["numeric_gt"] = df["gt_value"].apply(_parse_numeric)
    df["numeric_pred"] = df["pred"].apply(_parse_numeric)
    df["smape"] = [_compute_smape(a, b) for a, b in zip(df.numeric_gt, df.numeric_pred)]
    df["normalized_smape_score"] = df.smape.apply(
        lambda x: None if x is None else 1.0 - min(x, 1.0)
    )
    df["score"] = df.apply(_score_row, axis=1)
    df["leak"] = df.apply(_leak, axis=1)
    df["likely_unit_off"] = df.apply(_unit_off, axis=1)
    df["correct"] = df.apply(
        lambda r: (r["score"] is not None and r["score"] >= CORRECT_THRESHOLD),
        axis=1,
    )

    # gt_value may contain mixed types (str / int / float / bool); serialize
    # it so we can write parquet without per-row dtype headaches.
    df["gt_value"] = df["gt_value"].apply(lambda v: json.dumps(v, ensure_ascii=False))

    df.to_parquet(paths.scored_parquet, index=False)
    return df
