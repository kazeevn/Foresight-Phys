"""Cross-model analysis: pivot per-field, classify difficulty, write summary JSON.

All per-model and per-(model, ...) aggregates use a paper-macro of an
experiment-macro (one experiment = one vote within its paper, one paper = one
vote within a model). This matches the per-run ``aggregate_*`` numbers written
by ``benchmark.run_benchmark`` so the cross-model summary is not dominated by
papers that contribute many redundant fields.

For numeric fields we report a continuum of accuracy thresholds
(``factor_2``, ``factor_3``, ``factor_10``) instead of collapsing on the single
``score >= 0.5`` (factor 10**0.5 ≈ 3.16) cutoff.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
DISC_TYPES = {"bool", "categorical"}

# Multi-threshold cutoffs reported across all numeric tables / plots. The
# `0.5`-score threshold corresponds to factor 10**0.5 ≈ 3.16, so factor-3 is
# the closest physically-meaningful neighbour.
NUMERIC_THRESHOLDS = (
    ("frac_within_factor_2", float(np.log10(2))),
    ("frac_within_factor_3", float(np.log10(3))),
    ("frac_within_factor_10", 1.0),
)


def _paper_macro(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Return Series indexed by ``model`` of the paper-macro of ``value_col``.

    Aggregation: per (model, file_id, experiment) row-mean of ``value_col``
    (ignoring NaNs), then per (model, file_id) experiment-mean, then per model
    file-mean. Experiments and files with no valid rows drop out at each level,
    matching ``metrics.build_file_report_data`` / ``benchmark.average_metric``.
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
        values=["correct", "score", "log_accuracy", "numeric_pred"],
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
    score_macro = _paper_macro(scored, "score")
    correct_macro = _paper_macro(
        scored.assign(correct_f=scored["correct"].astype(float)),
        "correct_f",
    )
    # Drop infinite log_accuracy (predicted == 0 with non-zero GT) so the mean
    # is well-defined; the threshold-based numeric metrics still see those rows.
    log_acc_clean = scored.copy()
    log_acc_clean["log_accuracy"] = log_acc_clean["log_accuracy"].replace(
        [np.inf, -np.inf], np.nan
    )
    log_acc_macro = _paper_macro(log_acc_clean, "log_accuracy")
    disc = (
        scored[scored["type"].isin(DISC_TYPES)]
        .assign(correct_f=lambda d: d["correct"].astype(float))
    )
    disc_macro = _paper_macro(disc, "correct_f")

    rows: list[dict] = []
    for m, sub in scored.groupby("model"):
        rows.append({
            "model": m,
            "n_fields": int(len(sub)),
            "n_papers": int(sub["file_id"].nunique()),
            "mean_score": float(score_macro.get(m, float("nan"))) if m in score_macro.index else None,
            "mean_correct": float(correct_macro.get(m, float("nan"))) if m in correct_macro.index else None,
            "numeric_log_accuracy_mean": (
                float(log_acc_macro.get(m, float("nan"))) if m in log_acc_macro.index else None
            ),
            "bool_categorical_accuracy": (
                float(disc_macro.get(m, float("nan"))) if m in disc_macro.index else None
            ),
        })
    return rows


def _numeric_thresholds(scored: pd.DataFrame) -> list[dict]:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["numeric_gt"].notna()
        & scored["numeric_pred"].notna()
        & (scored["numeric_gt"] != 0)
        & (scored["numeric_pred"] != 0)
    ].copy()
    num["within_0.1_log_acc"] = (num["log_accuracy"] < 0.1).astype(float)
    for col, cutoff in NUMERIC_THRESHOLDS:
        num[col] = (num["log_accuracy"] < cutoff).astype(float)

    f01 = _paper_macro(num, "within_0.1_log_acc")
    macros = {col: _paper_macro(num, col) for col, _ in NUMERIC_THRESHOLDS}
    # Median of |log10 ratio|: median within experiment, then mean across.
    med_per_exp = num.groupby(["model", "file_id", "experiment"])["log_accuracy"].median()
    med_per_file = med_per_exp.groupby(["model", "file_id"]).mean()
    med_macro = med_per_file.groupby("model").mean()

    out: list[dict] = []
    for m, sub in num.groupby("model"):
        row = {
            "model": m,
            "n_fields": int(len(sub)),
            "n_papers": int(sub["file_id"].nunique()),
            "frac_within_0.1_log_acc": float(f01.get(m, float("nan"))) if m in f01.index else None,
        }
        for col, _ in NUMERIC_THRESHOLDS:
            macro = macros[col]
            row[col] = float(macro.get(m, float("nan"))) if m in macro.index else None
        row["median_abs_log10_ratio"] = (
            float(med_macro.get(m, float("nan"))) if m in med_macro.index else None
        )
        out.append(row)
    # Constant-predictor baseline (geometric median of all GT values), scored
    # the same paper-macro way so it is directly comparable to the model rows.
    unique_gts = num.drop_duplicates(["file_id", "experiment", "key"])
    gt = unique_gts["numeric_gt"].to_numpy()
    if len(gt):
        baseline = float(np.exp(np.median(np.log(np.abs(gt)))))
        ratio = np.abs(np.log10(np.abs(baseline / unique_gts["numeric_gt"])))
        bdf = unique_gts.assign(
            ratio=ratio,
            model="_constant_baseline_",
        )
        bdf["within_0.1_log_acc"] = (bdf["ratio"] < 0.1).astype(float)
        for col, cutoff in NUMERIC_THRESHOLDS:
            bdf[col] = (bdf["ratio"] < cutoff).astype(float)
        baseline_row = {
            "model": "_constant_baseline_",
            "n_fields": int(len(bdf)),
            "n_papers": int(bdf["file_id"].nunique()),
            "constant_value": baseline,
            "frac_within_0.1_log_acc": float(_paper_macro(bdf, "within_0.1_log_acc").iloc[0]),
        }
        for col, _ in NUMERIC_THRESHOLDS:
            baseline_row[col] = float(_paper_macro(bdf, col).iloc[0])
        out.append(baseline_row)
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
        "numeric_thresholds": _numeric_thresholds(scored),
        "difficulty_distribution_overall": difficulty_overall,
        "difficulty_by_type": difficulty_by_type,
        **_categorical_baseline(paths),
    }
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
