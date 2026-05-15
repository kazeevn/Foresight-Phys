from __future__ import annotations

import math
from typing import Any

from .formula_judging import FormulaJudge, FormulaJudgment
from .json_payloads import get_at_path, iter_result_paths


# Cap NLL / log-loss at this value so individual catastrophic predictions don't
# swamp the run aggregates and infinities don't propagate.
NLL_CAP = 30.0
# Floor for sigma and probabilities to keep log/division stable.
SIGMA_FLOOR = 1e-9
PROB_FLOOR = 1e-9
LN10 = math.log(10.0)
LOG_2PI = math.log(2.0 * math.pi)


def coerce_bool(value: Any) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return True, value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True, True
        if lowered in {"false", "no", "0"}:
            return True, False
    return False, False


def coerce_numeric(value: Any) -> tuple[bool, float]:
    if isinstance(value, bool):
        return False, 0.0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False, 0.0

    if not math.isfinite(numeric_value):
        return False, 0.0
    return True, numeric_value


def normalize_comparison_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def values_match(expected_value: Any, actual_value: Any) -> bool:
    if expected_value is None and actual_value is None:
        return True
    if isinstance(expected_value, bool):
        ok, actual_bool = coerce_bool(actual_value)
        return ok and actual_bool == expected_value
    return normalize_comparison_value(expected_value) == normalize_comparison_value(actual_value)


def normalize_formula_text(value: str) -> str:
    return "".join(value.split())


def get_result_type(payload: Any, result_path: tuple[Any, ...]) -> str | None:
    if not result_path or result_path[-1] != "result":
        return None
    parent_path = result_path[:-1]
    found, parent = get_at_path(payload, parent_path)
    if not found or not isinstance(parent, dict):
        return None
    result_type = parent.get("type")
    if isinstance(result_type, str):
        return result_type.lower()
    return None


def get_result_metadata(payload: Any, result_path: tuple[Any, ...]) -> dict[str, Any]:
    parent_path = result_path[:-1]
    found, metadata = get_at_path(payload, parent_path)
    if found and isinstance(metadata, dict):
        return metadata
    return {}


def get_experiment_description(payload: Any, result_path: tuple[Any, ...]) -> str:
    try:
        experiment_results_index = result_path.index("experiment_results")
    except ValueError:
        return ""

    description_path = (*result_path[:experiment_results_index], "experiment_description")
    found, experiment_description = get_at_path(payload, description_path)
    if found and isinstance(experiment_description, str):
        return experiment_description
    return ""


def is_numeric_result(result_type: str | None, expected_value: Any) -> bool:
    return result_type in {"float", "int", "integer", "number"} or (
        result_type is None
        and isinstance(expected_value, (int, float))
        and not isinstance(expected_value, bool)
    )


def is_formula_result(result_type: str | None) -> bool:
    return result_type == "formula"


def is_bool_or_categorical_result(result_type: str | None, expected_value: Any) -> bool:
    return result_type in {"bool", "boolean", "categorical"} or (
        result_type is None and (isinstance(expected_value, bool) or isinstance(expected_value, str))
    )


def is_bool_result(result_type: str | None, expected_value: Any) -> bool:
    return result_type in {"bool", "boolean"} or (
        result_type is None and isinstance(expected_value, bool)
    )


def is_categorical_result(result_type: str | None, expected_value: Any) -> bool:
    return result_type == "categorical" or (
        result_type is None
        and isinstance(expected_value, str)
        and not isinstance(expected_value, bool)
    )


def _clip_nll(value: float) -> float:
    if not math.isfinite(value):
        return NLL_CAP
    return min(max(value, -NLL_CAP), NLL_CAP)


def score_numeric(
    *,
    expected_value: Any,
    actual_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score a numeric prediction under the model's chosen normal / log_normal."""
    expected_ok, expected_value_f = coerce_numeric(expected_value)
    result: dict[str, Any] = {
        "expected_ok": expected_ok,
        "expected_value": expected_value_f if expected_ok else None,
        "predicted_value": None,
        "distribution": None,
        "sigma": None,
        "z": None,
        "nll": NLL_CAP,
        "quality": 0.0,
        "within_1sigma": None,
        "within_2sigma": None,
        "missing": actual_meta is None,
    }
    if not expected_ok or actual_meta is None:
        return result

    distribution = actual_meta.get("distribution")
    if distribution not in {"normal", "log_normal"}:
        # Treat malformed predictions as missing-uncertainty: maximum penalty.
        return result

    pred_ok, predicted_value = coerce_numeric(actual_meta.get("result"))
    if not pred_ok:
        return result

    sigma_raw = actual_meta.get("sigma")
    try:
        sigma = float(sigma_raw)
    except (TypeError, ValueError):
        return result
    if not math.isfinite(sigma) or sigma <= 0.0:
        return result
    sigma = max(sigma, SIGMA_FLOOR)

    result["predicted_value"] = predicted_value
    result["distribution"] = distribution
    result["sigma"] = sigma

    if distribution == "log_normal":
        # Log-normal is undefined at zero in either argument; treat as max-penalty.
        if expected_value_f == 0.0 or predicted_value == 0.0:
            return result
        # Sign mismatch is also undefined under log_normal; treat as max-penalty.
        if (expected_value_f > 0) != (predicted_value > 0):
            return result
        z = (math.log10(abs(expected_value_f)) - math.log10(abs(predicted_value))) / sigma
    else:  # normal
        z = (expected_value_f - predicted_value) / sigma

    nll = 0.5 * z * z + math.log(sigma) + 0.5 * LOG_2PI
    quality = math.exp(-0.5 * z * z)
    result["z"] = z
    result["nll"] = _clip_nll(nll)
    result["quality"] = quality
    result["within_1sigma"] = bool(abs(z) < 1.0)
    result["within_2sigma"] = bool(abs(z) < 2.0)
    return result


def score_bool(
    *,
    expected_value: Any,
    actual_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_ok, expected_bool = coerce_bool(expected_value)
    result: dict[str, Any] = {
        "expected_ok": expected_ok,
        "expected_value": expected_bool if expected_ok else None,
        "prob_true": None,
        "argmax": None,
        "correct": False,
        "log_loss": NLL_CAP,
        "brier": 1.0,
        "quality": 0.0,
        "missing": actual_meta is None,
    }
    if not expected_ok or actual_meta is None:
        return result

    prob_raw = actual_meta.get("prob_true")
    try:
        prob_true = float(prob_raw)
    except (TypeError, ValueError):
        return result
    if not math.isfinite(prob_true):
        return result
    prob_true = min(max(prob_true, 0.0), 1.0)

    argmax_ok, argmax_bool = coerce_bool(actual_meta.get("result"))
    if not argmax_ok:
        argmax_bool = prob_true >= 0.5

    result["prob_true"] = prob_true
    result["argmax"] = argmax_bool
    result["correct"] = argmax_bool == expected_bool

    y = 1.0 if expected_bool else 0.0
    p = min(max(prob_true, PROB_FLOOR), 1.0 - PROB_FLOOR)
    log_loss = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    brier = (prob_true - y) ** 2
    result["log_loss"] = _clip_nll(log_loss)
    result["brier"] = brier
    result["quality"] = 1.0 - brier
    return result


def _coerce_probability_map(value: Any) -> dict[str, float] | None:
    """Accept either {value: prob} or [{value, probability}] shapes."""
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                return None
        return out
    if isinstance(value, list):
        out = {}
        for entry in value:
            if not isinstance(entry, dict):
                return None
            key = entry.get("value")
            prob = entry.get("probability")
            if key is None or prob is None:
                return None
            try:
                out[str(key)] = float(prob)
            except (TypeError, ValueError):
                return None
        return out
    return None


def score_categorical(
    *,
    expected_value: Any,
    expected_meta: dict[str, Any],
    actual_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_str = str(expected_value) if expected_value is not None else None
    allowed = expected_meta.get("allowed_categorial_values")
    if isinstance(allowed, list) and allowed:
        allowed_values = [str(v) for v in allowed]
    elif expected_str is not None:
        allowed_values = [expected_str]
    else:
        allowed_values = []

    result: dict[str, Any] = {
        "expected_ok": expected_str is not None,
        "expected_value": expected_str,
        "allowed_values": allowed_values,
        "probabilities": None,
        "argmax": None,
        "correct": False,
        "log_loss": NLL_CAP,
        "brier": 1.0,
        "quality": 0.0,
        "missing": actual_meta is None,
    }
    if expected_str is None or actual_meta is None:
        return result

    prob_map = _coerce_probability_map(actual_meta.get("probabilities"))
    if prob_map is None:
        return result

    if allowed_values:
        # Restrict to allowed values; missing entries get a tiny probability.
        probabilities = {
            value: max(float(prob_map.get(value, 0.0)), 0.0) for value in allowed_values
        }
    else:
        probabilities = {k: max(float(v), 0.0) for k, v in prob_map.items()}

    total = sum(probabilities.values())
    if total <= 0.0:
        # No useful information; treat as uniform.
        n = max(len(probabilities), 1)
        probabilities = {k: 1.0 / n for k in probabilities}
    else:
        probabilities = {k: v / total for k, v in probabilities.items()}

    argmax_value = max(probabilities, key=lambda k: probabilities[k])
    argmax_meta = actual_meta.get("result")
    if isinstance(argmax_meta, str) and argmax_meta in probabilities:
        argmax_value = argmax_meta

    prob_truth = probabilities.get(expected_str, 0.0)
    clipped_p = min(max(prob_truth, PROB_FLOOR), 1.0)
    log_loss = -math.log(clipped_p)
    brier = sum(
        (probabilities.get(value, 0.0) - (1.0 if value == expected_str else 0.0)) ** 2
        for value in (set(probabilities) | {expected_str})
    )

    result["probabilities"] = probabilities
    result["argmax"] = argmax_value
    result["correct"] = argmax_value == expected_str
    result["log_loss"] = _clip_nll(log_loss)
    result["brier"] = brier
    result["quality"] = max(0.0, 1.0 - 0.5 * brier)
    return result


def judge_formula_values(
    *,
    expected_formula: Any,
    actual_formula: Any,
    experiment_description: str,
    result_key: str,
    result_description: str,
    formula_judge: FormulaJudge | None = None,
) -> FormulaJudgment:
    if not isinstance(expected_formula, str):
        return FormulaJudgment(
            equivalent=False,
            explanation="Reference formula is not a string.",
        )
    if not isinstance(actual_formula, str):
        return FormulaJudgment(
            equivalent=False,
            explanation="Predicted formula is missing or not a string.",
        )

    if normalize_formula_text(expected_formula) == normalize_formula_text(actual_formula):
        return FormulaJudgment(
            equivalent=True,
            explanation="Exact formula match after whitespace normalization.",
        )

    if formula_judge is None:
        return FormulaJudgment(
            equivalent=False,
            explanation="Formula judge unavailable.",
        )

    return formula_judge.judge(
        experiment_description=experiment_description,
        result_key=result_key,
        result_description=result_description,
        reference_formula=expected_formula,
        predicted_formula=actual_formula,
    )


def score_formula(
    *,
    expected_value: Any,
    expected_meta: dict[str, Any],
    actual_meta: dict[str, Any] | None,
    experiment_description: str,
    result_key: str,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "expected_value": expected_value,
        "predicted_value": None,
        "confidence": None,
        "equivalent": False,
        "explanation": "",
        "log_loss": NLL_CAP,
        "brier": 1.0,
        "quality": 0.0,
        "missing": actual_meta is None,
    }

    actual_value = actual_meta.get("result") if isinstance(actual_meta, dict) else None
    result["predicted_value"] = actual_value

    judgment = judge_formula_values(
        expected_formula=expected_value,
        actual_formula=actual_value,
        experiment_description=experiment_description,
        result_key=result_key,
        result_description=str(expected_meta.get("description", "")),
        formula_judge=formula_judge,
    )
    result["equivalent"] = judgment.equivalent
    result["explanation"] = judgment.explanation

    if actual_meta is None:
        return result

    try:
        confidence = float(actual_meta.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.5 if judgment.equivalent else 0.5
    if not math.isfinite(confidence):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)
    result["confidence"] = confidence

    y = 1.0 if judgment.equivalent else 0.0
    p = min(max(confidence, PROB_FLOOR), 1.0 - PROB_FLOOR)
    log_loss = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    brier = (confidence - y) ** 2
    result["log_loss"] = _clip_nll(log_loss)
    result["brier"] = brier
    result["quality"] = 1.0 - brier
    return result


def has_bool_or_categorical_targets(payload: Any) -> bool:
    for path, expected_value in iter_result_paths(payload):
        result_type = get_result_type(payload, path)
        if is_bool_or_categorical_result(result_type, expected_value):
            return True
    return False


METRIC_KEYS = (
    "prediction_quality",
    "numeric_quality",
    "numeric_nll",
    "coverage_1sigma",
    "coverage_2sigma",
    "bool_log_loss",
    "bool_quality",
    "bool_categorical_accuracy",
    "categorical_log_loss",
    "categorical_quality",
    "formula_accuracy",
    "formula_log_loss",
    "formula_quality",
)

COUNT_KEYS = (
    "result_count",
    "numeric_count",
    "bool_count",
    "categorical_count",
    "bool_categorical_count",
    "formula_count",
    "missing_predictions",
)


def _empty_experiment_metrics() -> dict[str, Any]:
    return {key: None for key in METRIC_KEYS} | {key: 0 for key in COUNT_KEYS}


def _format_numeric_status(score: dict[str, Any]) -> tuple[str, str, str]:
    if score["missing"]:
        return "missing prediction", "status-mismatch", ""
    if score["distribution"] is None:
        return "no uncertainty", "status-mismatch", ""
    if score["z"] is None:
        return "incomparable", "status-mismatch", "Reference or prediction undefined under log_normal."
    return (
        f"z = {score['z']:+.2f} (σ = {score['sigma']:.3g} {score['distribution']})",
        "status-numeric",
        f"NLL = {score['nll']:.3f}; quality = {score['quality']:.3f}",
    )


def _format_bool_status(score: dict[str, Any]) -> tuple[str, str, str]:
    if score["missing"]:
        return "missing prediction", "status-mismatch", ""
    correct = score["correct"]
    pct = score["prob_true"] * 100.0 if score["prob_true"] is not None else float("nan")
    text = f"P(true) = {pct:.0f}% ({'match' if correct else 'mismatch'})"
    return text, ("status-match" if correct else "status-mismatch"), (
        f"log-loss = {score['log_loss']:.3f}; brier = {score['brier']:.3f}"
    )


def _format_categorical_status(score: dict[str, Any]) -> tuple[str, str, str]:
    if score["missing"]:
        return "missing prediction", "status-mismatch", ""
    correct = score["correct"]
    truth = score["expected_value"]
    prob_truth = (
        score["probabilities"].get(truth, 0.0) if score["probabilities"] and truth else 0.0
    )
    return (
        f"P({truth}) = {prob_truth * 100.0:.0f}% ({'match' if correct else 'mismatch'})",
        "status-match" if correct else "status-mismatch",
        f"log-loss = {score['log_loss']:.3f}; brier = {score['brier']:.3f}",
    )


def _format_formula_status(score: dict[str, Any]) -> tuple[str, str, str]:
    if score["missing"]:
        return "missing prediction", "status-mismatch", ""
    confidence = score["confidence"] if score["confidence"] is not None else 0.5
    text = (
        f"{'match' if score['equivalent'] else 'mismatch'} (conf {confidence:.2f})"
    )
    return (
        text,
        "status-match" if score["equivalent"] else "status-mismatch",
        score["explanation"],
    )


def build_experiment_report(
    expected_experiment: Any,
    actual_experiment: Any,
    *,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    expected_desc = ""
    if isinstance(expected_experiment, dict):
        expected_desc = str(expected_experiment.get("experiment_description", ""))

    expected_results = (
        expected_experiment.get("experiment_results", {})
        if isinstance(expected_experiment, dict)
        else {}
    )
    actual_results = (
        actual_experiment.get("experiment_results", {})
        if isinstance(actual_experiment, dict)
        else {}
    )

    if not isinstance(expected_results, dict) or not expected_results:
        return {
            "experiment_description": expected_desc,
            "result_rows": [],
            "metrics": _empty_experiment_metrics(),
        }

    quality_values: list[float] = []
    numeric_quality_values: list[float] = []
    numeric_nll_values: list[float] = []
    coverage_1sigma_hits: list[float] = []
    coverage_2sigma_hits: list[float] = []
    bool_log_loss_values: list[float] = []
    bool_quality_values: list[float] = []
    classification_total = 0
    classification_correct = 0
    bool_count = 0
    categorical_count = 0
    categorical_log_loss_values: list[float] = []
    categorical_quality_values: list[float] = []
    formula_count = 0
    formula_correct = 0
    formula_log_loss_values: list[float] = []
    formula_quality_values: list[float] = []
    missing = 0
    numeric_count = 0
    result_rows: list[dict[str, Any]] = []

    for field_name, expected_meta in expected_results.items():
        expected_meta_dict = expected_meta if isinstance(expected_meta, dict) else {}
        field_description = str(expected_meta_dict.get("description", ""))
        field_type = str(expected_meta_dict.get("type", "")).strip().lower()
        expected_value = expected_meta_dict.get("result")

        actual_present = isinstance(actual_results, dict) and field_name in actual_results
        actual_meta_raw = actual_results.get(field_name) if actual_present else None
        actual_meta_dict = actual_meta_raw if isinstance(actual_meta_raw, dict) else None
        actual_value = actual_meta_dict.get("result") if actual_meta_dict else None

        if not actual_present:
            missing += 1

        row: dict[str, Any] = {
            "result_key": str(field_name),
            "description": field_description,
            "type": field_type,
            "ground_truth": expected_value,
            "predicted": actual_value,
        }

        if is_numeric_result(field_type, expected_value):
            numeric_count += 1
            score = score_numeric(
                expected_value=expected_value,
                actual_meta=actual_meta_dict,
            )
            quality_values.append(score["quality"])
            numeric_quality_values.append(score["quality"])
            numeric_nll_values.append(score["nll"])
            if score["within_1sigma"] is not None:
                coverage_1sigma_hits.append(1.0 if score["within_1sigma"] else 0.0)
            if score["within_2sigma"] is not None:
                coverage_2sigma_hits.append(1.0 if score["within_2sigma"] else 0.0)
            status_text, status_class, status_title = _format_numeric_status(score)
            row.update(
                {
                    "distribution": score["distribution"],
                    "sigma": score["sigma"],
                    "z": score["z"],
                    "nll": score["nll"],
                    "quality": score["quality"],
                }
            )
        elif is_formula_result(field_type):
            formula_count += 1
            score = score_formula(
                expected_value=expected_value,
                expected_meta=expected_meta_dict,
                actual_meta=actual_meta_dict,
                experiment_description=expected_desc,
                result_key=str(field_name),
                formula_judge=formula_judge,
            )
            quality_values.append(score["quality"])
            formula_log_loss_values.append(score["log_loss"])
            formula_quality_values.append(score["quality"])
            if score["equivalent"]:
                formula_correct += 1
            status_text, status_class, status_title = _format_formula_status(score)
            row.update(
                {
                    "confidence": score["confidence"],
                    "equivalent": score["equivalent"],
                    "log_loss": score["log_loss"],
                    "quality": score["quality"],
                }
            )
        elif is_bool_result(field_type, expected_value):
            bool_count += 1
            classification_total += 1
            score = score_bool(
                expected_value=expected_value,
                actual_meta=actual_meta_dict,
            )
            quality_values.append(score["quality"])
            bool_log_loss_values.append(score["log_loss"])
            bool_quality_values.append(score["quality"])
            if score["correct"]:
                classification_correct += 1
            status_text, status_class, status_title = _format_bool_status(score)
            row.update(
                {
                    "prob_true": score["prob_true"],
                    "log_loss": score["log_loss"],
                    "quality": score["quality"],
                }
            )
        elif is_categorical_result(field_type, expected_value):
            categorical_count += 1
            classification_total += 1
            score = score_categorical(
                expected_value=expected_value,
                expected_meta=expected_meta_dict,
                actual_meta=actual_meta_dict,
            )
            quality_values.append(score["quality"])
            categorical_log_loss_values.append(score["log_loss"])
            categorical_quality_values.append(score["quality"])
            if score["correct"]:
                classification_correct += 1
            status_text, status_class, status_title = _format_categorical_status(score)
            row.update(
                {
                    "probabilities": score["probabilities"],
                    "log_loss": score["log_loss"],
                    "quality": score["quality"],
                }
            )
        else:
            # Unknown type: fall back to argmax-only match scoring.
            match = actual_present and values_match(expected_value, actual_value)
            quality = 1.0 if match else 0.0
            quality_values.append(quality)
            status_text = "match" if match else "mismatch"
            status_class = "status-match" if match else "status-mismatch"
            status_title = ""
            row.update({"quality": quality})

        row["status_text"] = status_text
        row["status_class"] = status_class
        row["status_title"] = status_title or None
        result_rows.append(row)

    total_results = len(result_rows)
    bool_categorical_count = bool_count + categorical_count
    metrics = {
        "prediction_quality": (
            sum(quality_values) / len(quality_values) if quality_values else None
        ),
        "numeric_quality": (
            sum(numeric_quality_values) / len(numeric_quality_values)
            if numeric_quality_values
            else None
        ),
        "numeric_nll": (
            sum(numeric_nll_values) / len(numeric_nll_values)
            if numeric_nll_values
            else None
        ),
        "coverage_1sigma": (
            sum(coverage_1sigma_hits) / len(coverage_1sigma_hits)
            if coverage_1sigma_hits
            else None
        ),
        "coverage_2sigma": (
            sum(coverage_2sigma_hits) / len(coverage_2sigma_hits)
            if coverage_2sigma_hits
            else None
        ),
        "bool_log_loss": (
            sum(bool_log_loss_values) / len(bool_log_loss_values)
            if bool_log_loss_values
            else None
        ),
        "bool_quality": (
            sum(bool_quality_values) / len(bool_quality_values)
            if bool_quality_values
            else None
        ),
        "bool_categorical_accuracy": (
            classification_correct / classification_total if classification_total else None
        ),
        "categorical_log_loss": (
            sum(categorical_log_loss_values) / len(categorical_log_loss_values)
            if categorical_log_loss_values
            else None
        ),
        "categorical_quality": (
            sum(categorical_quality_values) / len(categorical_quality_values)
            if categorical_quality_values
            else None
        ),
        "formula_accuracy": (
            formula_correct / formula_count if formula_count else None
        ),
        "formula_log_loss": (
            sum(formula_log_loss_values) / len(formula_log_loss_values)
            if formula_log_loss_values
            else None
        ),
        "formula_quality": (
            sum(formula_quality_values) / len(formula_quality_values)
            if formula_quality_values
            else None
        ),
        "result_count": total_results,
        "numeric_count": numeric_count,
        "bool_count": bool_count,
        "categorical_count": categorical_count,
        "bool_categorical_count": bool_categorical_count,
        "formula_count": formula_count,
        "missing_predictions": missing,
    }
    return {
        "experiment_description": expected_desc,
        "result_rows": result_rows,
        "metrics": metrics,
    }


def _experiments_for_aggregation(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    return [payload]


def compute_experiment_metrics(
    expected_experiment: Any,
    actual_experiment: Any,
    *,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    return build_experiment_report(
        expected_experiment,
        actual_experiment,
        formula_judge=formula_judge,
    )["metrics"]


def build_file_report_data(
    expected_json: Any,
    actual_json: Any,
    *,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    expected_experiments = _experiments_for_aggregation(expected_json)
    actual_experiments = _experiments_for_aggregation(actual_json)

    experiments: list[dict[str, Any]] = []
    per_experiment_metrics: list[dict[str, Any]] = []
    for index, expected_experiment in enumerate(expected_experiments, start=1):
        actual_experiment = (
            actual_experiments[index - 1] if index - 1 < len(actual_experiments) else {}
        )
        experiment_report = build_experiment_report(
            expected_experiment,
            actual_experiment,
            formula_judge=formula_judge,
        )
        per_experiment_metrics.append(experiment_report["metrics"])
        experiments.append(
            {
                "experiment_index": index,
                "experiment_description": experiment_report["experiment_description"],
                "result_rows": experiment_report["result_rows"],
            }
        )

    aggregated: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in per_experiment_metrics if m[key] is not None]
        aggregated[key] = sum(values) / len(values) if values else None
    for key in COUNT_KEYS:
        aggregated[key] = sum(m[key] for m in per_experiment_metrics)

    return {
        **aggregated,
        "report_experiments": experiments,
    }


def compute_file_metrics(
    expected_json: Any,
    actual_json: Any,
    *,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    report_data = build_file_report_data(
        expected_json,
        actual_json,
        formula_judge=formula_judge,
    )
    return {key: report_data[key] for key in (*METRIC_KEYS, *COUNT_KEYS)}
