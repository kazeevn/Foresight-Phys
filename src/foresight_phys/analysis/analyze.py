"""Cross-model analysis: pivot per-field, classify difficulty, write summary JSON."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
DISC_TYPES = {"bool", "categorical"}


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
            "leak",
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
    rows: list[dict] = []
    for m, sub in scored.groupby("model"):
        rows.append({
            "model": m,
            "n": int(len(sub)),
            "mean_score": float(sub["score"].mean()),
            "mean_correct": float(sub["correct"].mean()),
            "numeric_log_accuracy_mean": float(sub["log_accuracy"].mean()),
            "bool_categorical_accuracy": float(
                sub.loc[sub["type"].isin(DISC_TYPES), "correct"].mean()
            ) if sub["type"].isin(DISC_TYPES).any() else None,
        })
    return rows


def _numeric_thresholds(scored: pd.DataFrame) -> list[dict]:
    num = scored[
        scored["type"].isin(NUMERIC_TYPES)
        & scored["numeric_gt"].notna()
        & scored["numeric_pred"].notna()
        & (scored["numeric_gt"] != 0)
        & (scored["numeric_pred"] != 0)
    ]
    out: list[dict] = []
    for m, sub in num.groupby("model"):
        ratio = sub["log_accuracy"]
        out.append({
            "model": m,
            "n": int(len(sub)),
            "frac_within_0.1_log_acc": float((ratio < 0.1).mean()),
            "frac_within_factor_2": float((ratio < np.log10(2)).mean()),
            "frac_within_decade": float((ratio < 1).mean()),
            "median_abs_log10_ratio": float(ratio.median()),
        })
    # Constant-predictor baseline (geometric median of all GT values).
    gt = num.drop_duplicates(["file_id", "experiment", "key"])["numeric_gt"].to_numpy()
    if len(gt):
        baseline = float(np.exp(np.median(np.log(np.abs(gt)))))
        ratio = np.abs(np.log10(np.abs(baseline / gt)))
        out.append({
            "model": "_constant_baseline_",
            "n": int(len(gt)),
            "constant_value": baseline,
            "frac_within_0.1_log_acc": float((ratio < 0.1).mean()),
            "frac_within_factor_2": float((ratio < np.log10(2)).mean()),
            "frac_within_decade": float((ratio < 1).mean()),
        })
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

    leakage_impact = (
        scored.groupby(["model", "leak"])
        .agg(n=("score", "count"),
             mean_score=("score", "mean"),
             accuracy=("correct", "mean"))
        .reset_index().to_dict(orient="records")
    )

    summary = {
        "n_models": int(scored["model"].nunique()),
        "n_papers": int(scored["file_id"].nunique()),
        "n_fields": int(len(wide)),
        "per_model": _per_model_summary(scored),
        "numeric_thresholds": _numeric_thresholds(scored),
        "difficulty_distribution_overall": difficulty_overall,
        "difficulty_by_type": difficulty_by_type,
        "leakage_impact": leakage_impact,
        **_categorical_baseline(paths),
    }
    paths.summary_json.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
