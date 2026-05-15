"""Join parsed benchmark predictions with ground-truth JSON metadata.

Produces a long dataframe with one row per (model, file, experiment, key). Adds:
- ``result_type`` / ``gt_value`` from the ground truth JSON
- ``numeric_gt`` / ``numeric_pred`` (when parseable as floats; used for scatter)
- ``correct`` (per-field quality >= ``CORRECT_THRESHOLD``)
- ``likely_unit_off`` (factor 1e3/1e6 ratio between pred and gt)

Field-level scoring (``quality``, ``crps``, ``z``, ``brier`` …) is computed
upstream in ``foresight_phys.metrics`` and merely passed through here.
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
                    "gt_type": v.get("type"),
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
            "file_id", "experiment", "key", "gt_type", "gt_value",
            "result_description", "experiment_description",
        ]],
        on=["file_id", "experiment", "key"],
        how="left",
        suffixes=("_parsed", ""),
    )

    # Prefer the ground-truth type tag; fall back to the parquet's row type when
    # absent (e.g., for legacy fixtures without an explicit `type` column).
    df["type"] = df["gt_type"].fillna(df["type"]) if "type" in df.columns else df["gt_type"]
    df = df.drop(columns=["gt_type"])

    df["numeric_gt"] = df["gt_value"].apply(_parse_numeric)
    df["numeric_pred"] = df["pred"].apply(_parse_numeric)
    df["likely_unit_off"] = df.apply(_unit_off, axis=1)

    if "quality" not in df.columns:
        df["quality"] = None
    df["correct"] = df["quality"].apply(
        lambda v: (v is not None and not pd.isna(v) and float(v) >= CORRECT_THRESHOLD)
    )

    # gt_value may contain mixed types; serialize so we can write parquet.
    df["gt_value"] = df["gt_value"].apply(lambda v: json.dumps(v, ensure_ascii=False))

    df.to_parquet(paths.scored_parquet, index=False)
    return df
