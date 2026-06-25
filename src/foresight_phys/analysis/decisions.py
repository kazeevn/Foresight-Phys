"""Decision-usefulness metrics derived from cached predictions + annotations.

The benchmark scores isolated field values, but the motivating use case is a
*decision*: which experiment/condition to run. Decisions live in the
``comparison_sets`` produced by the annotation pass (a swept series or a
baseline/intervention contrast). For each set we reconstruct each member's
predicted distribution from the stored quantiles and score the *decision* a
researcher would take:

- ``argmax_select``: pick the condition that maximises the quantity — scored by
  top-1 selection accuracy and normalised regret against the oracle, plus the
  decision value over a no-model (random-pick) baseline.
- ``monotonic_direction`` / ``sign_vs_baseline``: get the direction/sign of an
  effect right — scored by directional accuracy and a proper sign Brier from the
  predicted distributions.
- ``order_of_magnitude``: get the scale right — scored by the order-of-magnitude
  hit rate over the set members.

Everything here is computed from ``scored.parquet`` (no re-forecasting needed).
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .paths import AnalysisPaths, default_paths

NUMERIC_TYPES = {"float", "integer", "int", "number"}
DIRECTION_QUESTION_TYPES = {"monotonic_direction", "sign_vs_baseline"}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _as_float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return True
        if value.lower() in {"false", "0"}:
            return False
    return None


def _slope_sign(values: list[float]) -> int:
    """Sign of the least-squares slope of ``values`` against position index."""
    n = len(values)
    if n < 2:
        return 0
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    cov = float(((x - x.mean()) * (y - y.mean())).sum())
    if cov > 1e-12:
        return 1
    if cov < -1e-12:
        return -1
    return 0


def _prob_a_greater_b(row_a: dict[str, Any], row_b: dict[str, Any]) -> float | None:
    """P(Y_a > Y_b) from two fitted predictive Gaussians, when comparable.

    Same-family pairs are scored exactly (``normal`` in linear units,
    ``log_normal`` in log10 units, where ordering is preserved). Mixed-family
    pairs return ``None`` (excluded from the sign Brier).
    """
    dist_a, dist_b = row_a.get("distribution"), row_b.get("distribution")
    if dist_a != dist_b or dist_a not in {"normal", "log_normal"}:
        return None
    p50_a, p50_b = _as_float_or_none(row_a.get("p50")), _as_float_or_none(row_b.get("p50"))
    sa, sb = _as_float_or_none(row_a.get("sigma")), _as_float_or_none(row_b.get("sigma"))
    if None in (p50_a, p50_b, sa, sb):
        return None
    if dist_a == "log_normal":
        if p50_a <= 0 or p50_b <= 0:
            return None
        center_a, center_b = math.log10(p50_a), math.log10(p50_b)
    else:
        center_a, center_b = p50_a, p50_b
    denom = math.sqrt(sa * sa + sb * sb)
    if denom <= 0:
        return 1.0 if center_a > center_b else (0.0 if center_a < center_b else 0.5)
    return _normal_cdf((center_a - center_b) / denom)


def load_comparison_sets(paths: AnalysisPaths) -> dict[str, list[dict[str, Any]]]:
    sets_by_file: dict[str, list[dict[str, Any]]] = {}
    if not paths.annotations_dir.exists():
        return sets_by_file
    for ann_path in sorted(paths.annotations_dir.glob("*.json")):
        record = json.loads(ann_path.read_text(encoding="utf-8"))
        sets_by_file[ann_path.stem] = record.get("comparison_sets", [])
    return sets_by_file


def _member_rows(
    model_block: pd.DataFrame, experiment: int, member_keys: list[str]
) -> list[dict[str, Any]] | None:
    """Ordered member rows for one model; None if any member is missing/non-scorable."""
    rows: list[dict[str, Any]] = []
    for key in member_keys:
        match = model_block[(model_block["experiment"] == experiment) & (model_block["key"] == key)]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        if str(row.get("type")) not in NUMERIC_TYPES:
            return None
        point = _as_float_or_none(row.get("p50"))
        truth = _as_float_or_none(row.get("numeric_gt"))
        if point is None or truth is None:
            return None
        row["_point"] = point
        row["_truth"] = truth
        rows.append(row)
    return rows if len(rows) >= 2 else None


def _score_set(question_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = [r["_point"] for r in rows]
    truths = [r["_truth"] for r in rows]
    out: dict[str, Any] = {
        "decision_correct": None,
        "regret": None,
        "baseline_regret": None,
        "decision_value": None,
        "direction_correct": None,
        "sign_brier": None,
        "oom_factor3": None,
        "oom_decade": None,
    }

    if question_type == "argmax_select":
        pred_idx = int(np.argmax(points))
        true_best = max(truths)
        true_worst = min(truths)
        spread = true_best - true_worst
        out["decision_correct"] = float(truths[pred_idx] == true_best)
        if spread > 0:
            out["regret"] = (true_best - truths[pred_idx]) / spread
            out["baseline_regret"] = (true_best - float(np.mean(truths))) / spread
            out["decision_value"] = out["baseline_regret"] - out["regret"]
        else:
            out["regret"] = 0.0
            out["baseline_regret"] = 0.0
            out["decision_value"] = 0.0
    elif question_type in DIRECTION_QUESTION_TYPES:
        if question_type == "monotonic_direction":
            pred_dir = _slope_sign(points)
            true_dir = _slope_sign(truths)
        else:  # sign_vs_baseline: first member is the baseline, last the intervention
            pred_dir = int(np.sign(points[-1] - points[0]))
            true_dir = int(np.sign(truths[-1] - truths[0]))
        if true_dir != 0:
            out["decision_correct"] = float(pred_dir == true_dir)
            out["direction_correct"] = out["decision_correct"]
            prob_up = _prob_a_greater_b(rows[-1], rows[0])
            if prob_up is not None:
                outcome = 1.0 if true_dir > 0 else 0.0
                out["sign_brier"] = (prob_up - outcome) ** 2
    elif question_type == "order_of_magnitude":
        f3 = [_as_bool_or_none(r.get("within_factor_3")) for r in rows]
        dec = [_as_bool_or_none(r.get("within_decade")) for r in rows]
        f3 = [v for v in f3 if v is not None]
        dec = [v for v in dec if v is not None]
        if f3:
            out["oom_factor3"] = float(np.mean([1.0 if v else 0.0 for v in f3]))
            out["decision_correct"] = float(out["oom_factor3"] >= 0.5)
        if dec:
            out["oom_decade"] = float(np.mean([1.0 if v else 0.0 for v in dec]))
    return out


def build_decision_rows(
    scored: pd.DataFrame, sets_by_file: dict[str, list[dict[str, Any]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_block in scored.groupby("model"):
        for file_id, comparison_sets in sets_by_file.items():
            file_block = model_block[model_block["file_id"] == file_id]
            if file_block.empty:
                continue
            for set_index, cset in enumerate(comparison_sets):
                question_type = cset.get("question_type")
                experiment = int(cset.get("experiment_index", -1))
                member_keys = list(cset.get("member_keys", []))
                members = _member_rows(file_block, experiment, member_keys)
                base = {
                    "model": model,
                    "file_id": file_id,
                    "experiment": experiment,
                    "set_index": set_index,
                    "question_type": question_type,
                    "ordering_variable": cset.get("ordering_variable", ""),
                    "n_members": len(member_keys),
                    "scorable": members is not None,
                }
                if members is None:
                    rows.append(base)
                    continue
                rows.append({**base, **_score_set(question_type, members)})
    return pd.DataFrame(rows)


def _paper_macro(df: pd.DataFrame, value_col: str) -> pd.Series:
    valid = df[df[value_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    per_file = valid.groupby(["model", "file_id"])[value_col].mean()
    return per_file.groupby("model").mean()


def summarize_decisions(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"n_sets_total": 0, "per_model": []}

    scorable = rows[rows["scorable"]].copy()
    macro_decision = _paper_macro(scorable, "decision_correct")
    argmax = scorable[scorable["question_type"] == "argmax_select"]
    direction = scorable[scorable["question_type"].isin(DIRECTION_QUESTION_TYPES)]
    oom = scorable[scorable["question_type"] == "order_of_magnitude"]

    per_model: list[dict[str, Any]] = []
    for model in sorted(rows["model"].unique()):
        sm = scorable[scorable["model"] == model]
        am = argmax[argmax["model"] == model]
        dm = direction[direction["model"] == model]
        om = oom[oom["model"] == model]

        def _mean(frame: pd.DataFrame, col: str) -> float | None:
            vals = frame[col].dropna()
            return float(vals.mean()) if len(vals) else None

        per_model.append(
            {
                "model": model,
                "n_scorable_sets": int(len(sm)),
                "decision_accuracy_micro": _mean(sm, "decision_correct"),
                "decision_accuracy_paper_macro": (
                    float(macro_decision.get(model)) if model in macro_decision.index else None
                ),
                "selection_accuracy": _mean(am, "decision_correct"),
                "mean_regret": _mean(am, "regret"),
                "baseline_regret": _mean(am, "baseline_regret"),
                "decision_value": _mean(am, "decision_value"),
                "direction_accuracy": _mean(dm, "decision_correct"),
                "sign_brier": _mean(dm, "sign_brier"),
                "oom_set_factor3": _mean(om, "oom_factor3"),
                "oom_set_decade": _mean(om, "oom_decade"),
                "n_argmax_sets": int(len(am)),
                "n_direction_sets": int(len(dm)),
                "n_oom_sets": int(len(om)),
            }
        )

    return {
        "n_sets_total": int(len(rows)),
        "n_sets_scorable": int(len(scorable)),
        "n_sets_unscorable": int((~rows["scorable"]).sum()),
        "question_type_distribution": scorable["question_type"].value_counts().to_dict(),
        "per_model": per_model,
    }


def write_decisions(paths: AnalysisPaths | None = None) -> dict[str, Any]:
    paths = paths or default_paths()
    paths.ensure_dirs()
    scored = pd.read_parquet(paths.scored_parquet)
    if "ablation" in scored.columns:
        scored = scored[scored["ablation"] == "none"]
    sets_by_file = load_comparison_sets(paths)
    rows = build_decision_rows(scored, sets_by_file)
    if not rows.empty:
        rows.to_parquet(paths.decisions_parquet, index=False)
    return summarize_decisions(rows)
