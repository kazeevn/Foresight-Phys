"""Cross-model analysis: pivot per-field, classify difficulty, write summary JSON.

All per-model and per-(model, ...) aggregates use a paper-macro of an
experiment-macro (one experiment = one vote within its paper, one paper = one
vote within a model). This matches the per-run ``aggregate_*`` numbers written
by ``benchmark.run_benchmark`` so the cross-model summary is not dominated by
papers that contribute many redundant fields.

Numeric fields are summarised through proper-scoring-rule quantities: mean
CRPS, mean quality (``exp(-z²/2)``), and coverage at 1σ / 2σ as calibration
diagnostics.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
DISC_TYPES = {"bool", "categorical"}

COVERAGE_BUCKETS = (
    ("frac_within_1sigma", 1.0),
    ("frac_within_2sigma", 2.0),
    ("frac_within_3sigma", 3.0),
)


def _paper_macro(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Per (model, file, exp) row-mean, then per (model, file), then per model.

    Mirrors ``metrics.build_file_report_data`` /
    ``benchmark.average_metric``.
    """
    valid = df[df[value_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    per_exp = valid.groupby(["model", "file_id", "experiment"])[value_col].mean()
    per_file = per_exp.groupby(["model", "file_id"]).mean()
    return per_file.groupby("model").mean()


def _difficulty(n_correct: int, n_models: int) -> str:
    if n_correct == n_models:
        return "trivial (all correct)"
    if n_correct == n_models - 1:
        return "easy (1 wrong)"
    if n_correct == 0:
        return "intractable (none correct)"
    if n_correct == 1:
        return "hard (1 right)"
    return "discriminative"


def _build_wide(scored: pd.DataFrame) -> pd.DataFrame:
    models = sorted(scored["model"].unique())
    wide = scored.pivot_table(
        index=[
            "file_id",
            "experiment",
            "key",
            "type",
            "gt_value",
            "result_description",
            "experiment_description",
        ],
        columns="model",
        values=["correct", "quality", "crps", "numeric_pred", "z"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [f"{a}__{b}" if b else a for a, b in wide.columns]

    correct_cols = [f"correct__{m}" for m in models]
    wide["n_correct"] = wide[correct_cols].sum(axis=1).astype(int)
    wide["n_models"] = len(models)
    wide["difficulty"] = wide.apply(
        lambda r: _difficulty(r["n_correct"], r["n_models"]), axis=1
    )
    return wide


def _per_model_summary(scored: pd.DataFrame) -> list[dict]:
    quality_macro = _paper_macro(scored, "quality")
    correct_macro = _paper_macro(
        scored.assign(correct_f=scored["correct"].astype(float)),
        "correct_f",
    )

    numeric_df = scored[scored["type"].isin(NUMERIC_TYPES)].copy()
    numeric_df["abs_z"] = numeric_df["z"].abs()
    for col, cutoff in COVERAGE_BUCKETS:
        numeric_df[col] = (numeric_df["abs_z"] < cutoff).astype(float)
    crps_macro = _paper_macro(numeric_df, "crps")
    crps_scaled_macro = (
        _paper_macro(numeric_df, "crps_scaled")
        if "crps_scaled" in numeric_df.columns
        else pd.Series(dtype=float)
    )
    numeric_quality_macro = _paper_macro(numeric_df, "quality")
    coverage_macros = {col: _paper_macro(numeric_df, col) for col, _ in COVERAGE_BUCKETS}

    disc_df = (
        scored[scored["type"].isin(DISC_TYPES)]
        .assign(correct_f=lambda d: d["correct"].astype(float))
    )
    disc_macro = _paper_macro(disc_df, "correct_f")
    disc_brier_macro = _paper_macro(disc_df, "brier")

    rows: list[dict] = []
    for m, sub in scored.groupby("model"):
        row = {
            "model": m,
            "n_fields": int(len(sub)),
            "n_papers": int(sub["file_id"].nunique()),
            "mean_quality": float(quality_macro.get(m, float("nan"))) if m in quality_macro.index else None,
            "mean_correct": float(correct_macro.get(m, float("nan"))) if m in correct_macro.index else None,
            "numeric_quality": (
                float(numeric_quality_macro.get(m, float("nan")))
                if m in numeric_quality_macro.index
                else None
            ),
            "numeric_crps": float(crps_macro.get(m, float("nan"))) if m in crps_macro.index else None,
            "numeric_crps_scaled": (
                float(crps_scaled_macro.get(m, float("nan")))
                if m in crps_scaled_macro.index
                else None
            ),
            "bool_categorical_accuracy": (
                float(disc_macro.get(m, float("nan"))) if m in disc_macro.index else None
            ),
            "bool_categorical_brier": (
                float(disc_brier_macro.get(m, float("nan")))
                if m in disc_brier_macro.index
                else None
            ),
        }
        for col, _ in COVERAGE_BUCKETS:
            macro = coverage_macros[col]
            row[col] = float(macro.get(m, float("nan"))) if m in macro.index else None
        rows.append(row)
    return rows


def _numeric_calibration(scored: pd.DataFrame) -> list[dict]:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["z"].notna()
    ].copy()
    num["abs_z"] = num["z"].abs()
    for col, cutoff in COVERAGE_BUCKETS:
        num[col] = (num["abs_z"] < cutoff).astype(float)
    crps_macro = _paper_macro(num, "crps")
    crps_scaled_macro = (
        _paper_macro(num, "crps_scaled")
        if "crps_scaled" in num.columns
        else pd.Series(dtype=float)
    )
    quality_macro = _paper_macro(num, "quality")
    coverage = {col: _paper_macro(num, col) for col, _ in COVERAGE_BUCKETS}

    out: list[dict] = []
    for m, sub in num.groupby("model"):
        row = {
            "model": m,
            "n_fields": int(len(sub)),
            "n_papers": int(sub["file_id"].nunique()),
            "numeric_crps": float(crps_macro.get(m, float("nan"))) if m in crps_macro.index else None,
            "numeric_crps_scaled": (
                float(crps_scaled_macro.get(m, float("nan")))
                if m in crps_scaled_macro.index
                else None
            ),
            "numeric_quality": (
                float(quality_macro.get(m, float("nan")))
                if m in quality_macro.index
                else None
            ),
        }
        for col, _ in COVERAGE_BUCKETS:
            macro = coverage[col]
            row[col] = float(macro.get(m, float("nan"))) if m in macro.index else None
        out.append(row)
    return out


def _categorical_baseline(paths: AnalysisPaths) -> dict:
    cat_k: list[int] = []
    for jp in sorted(paths.json_dir.glob("*.json")):
        data = json.loads(jp.read_text())
        if not data:
            continue
        for exp in data:
            for v in exp["experiment_results"].values():
                if v.get("type") == "categorical":
                    allowed = v.get("allowed_categorial_values")
                    if isinstance(allowed, list) and len(allowed) > 1:
                        cat_k.append(len(allowed))
    if not cat_k:
        return {}
    return {
        "categorical_random_baseline": float(np.mean([1.0 / k for k in cat_k])),
        "categorical_k_distribution": dict(Counter(cat_k)),
    }


def write_summary(paths: AnalysisPaths | None = None) -> dict:
    paths = paths or default_paths()
    paths.ensure_dirs()

    scored = pd.read_parquet(paths.scored_parquet)
    wide = _build_wide(scored)
    wide.to_parquet(paths.per_field_parquet, index=False)

    difficulty_overall = wide["difficulty"].value_counts().to_dict()
    difficulty_by_type = (
        wide.groupby(["type", "difficulty"]).size().unstack(fill_value=0)
        .reset_index().to_dict(orient="records")
    )

    summary = {
        "n_models": int(scored["model"].nunique()),
        "n_papers": int(scored["file_id"].nunique()),
        "n_fields": int(len(wide)),
        "aggregation": "paper-macro of experiment-macro (matches per-run aggregate_*)",
        "per_model": _per_model_summary(scored),
        "numeric_calibration": _numeric_calibration(scored),
        "difficulty_distribution_overall": difficulty_overall,
        "difficulty_by_type": difficulty_by_type,
        **_categorical_baseline(paths),
    }
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
