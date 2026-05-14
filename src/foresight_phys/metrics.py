from __future__ import annotations

import math
from typing import Any

from .formula_judging import FormulaJudge, FormulaJudgment
from .json_payloads import get_at_path, iter_result_paths


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


def compute_smape(expected_numeric: float, actual_numeric: float) -> float:
    denominator = abs(expected_numeric) + abs(actual_numeric)
    if denominator == 0.0:
        return 0.0
    return 2.0 * abs(actual_numeric - expected_numeric) / denominator


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


def judge_formula_result(
    expected_payload: Any,
    actual_payload: Any,
    result_path: tuple[Any, ...],
    *,
    formula_judge: FormulaJudge | None = None,
) -> FormulaJudgment:
    expected_found, expected_value = get_at_path(expected_payload, result_path)
    actual_found, actual_value = get_at_path(actual_payload, result_path)
    if not expected_found:
        return FormulaJudgment(
            equivalent=False,
            explanation="Reference formula path was not found.",
        )
    if not actual_found:
        return FormulaJudgment(
            equivalent=False,
            explanation="Predicted formula path was not found.",
        )

    metadata = get_result_metadata(expected_payload, result_path)
    result_key = str(result_path[-2]) if len(result_path) >= 2 else "formula"
    result_description = str(metadata.get("description", ""))
    experiment_description = get_experiment_description(expected_payload, result_path)
    return judge_formula_values(
        expected_formula=expected_value,
        actual_formula=actual_value,
        experiment_description=experiment_description,
        result_key=result_key,
        result_description=result_description,
        formula_judge=formula_judge,
    )


def has_bool_or_categorical_targets(payload: Any) -> bool:
    for path, expected_value in iter_result_paths(payload):
        result_type = get_result_type(payload, path)
        if is_bool_or_categorical_result(result_type, expected_value):
            return True
    return False


METRIC_KEYS = (
    "prediction_quality",
    "smape",
    "normalized_smape_score",
    "bool_categorical_accuracy",
    "formula_accuracy",
)

COUNT_KEYS = (
    "result_count",
    "numeric_count",
    "nonzero_numeric_count",
    "zero_reference_numeric_count",
    "smape_count",
    "normalized_smape_count",
    "bool_categorical_count",
    "formula_count",
    "missing_predictions",
)


def _experiments_for_aggregation(payload: Any) -> list[Any]:
    """Return the list of experiment objects in a benchmark payload.

    Top-level payloads are either a list of experiments or a single
    experiment dict. Anything else collapses to a single-element list so
    `compute_experiment_metrics` can still walk its result paths.
    """
    if isinstance(payload, list):
        return list(payload)
    return [payload]


def compute_experiment_metrics(
    expected_experiment: Any,
    actual_experiment: Any,
    *,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    """Compute metrics for a single experiment (one element of the payload list)."""
    raw_smape_values: list[float] = []
    normalized_smape_scores: list[float] = []
    classification_total = 0
    classification_correct = 0
    formula_total = 0
    formula_correct = 0
    missing = 0
    total_prediction_quality = 0.0
    total_results = 0
    numeric_count = 0
    nonzero_numeric_count = 0
    zero_reference_numeric_count = 0

    for path, expected_value in iter_result_paths(expected_experiment):
        total_results += 1
        found, actual_value = get_at_path(actual_experiment, path)
        if not found:
            missing += 1

        result_type = get_result_type(expected_experiment, path)
        is_numeric = is_numeric_result(result_type, expected_value)
        is_formula = is_formula_result(result_type)
        is_bool_or_categorical = is_bool_or_categorical_result(result_type, expected_value)

        if is_numeric:
            expected_ok, expected_numeric = coerce_numeric(expected_value)
            if not expected_ok:
                continue

            numeric_count += 1
            actual_ok, actual_numeric = coerce_numeric(actual_value) if found else (False, 0.0)

            if expected_numeric == 0.0:
                zero_reference_numeric_count += 1
                if actual_ok and actual_numeric == 0.0:
                    total_prediction_quality += 1.0
                continue

            nonzero_numeric_count += 1
            if not actual_ok:
                normalized_smape_scores.append(0.0)
                continue

            smape = compute_smape(expected_numeric, actual_numeric)
            normalized_score = 1.0 - min(smape, 1.0)
            raw_smape_values.append(smape)
            normalized_smape_scores.append(normalized_score)
            total_prediction_quality += normalized_score
            continue

        if is_formula:
            formula_total += 1
            formula_judgment = judge_formula_result(
                expected_experiment,
                actual_experiment,
                path,
                formula_judge=formula_judge,
            )
            if formula_judgment.equivalent:
                formula_correct += 1
                total_prediction_quality += 1.0
            continue

        accuracy_score = 1.0 if found and values_match(expected_value, actual_value) else 0.0
        total_prediction_quality += accuracy_score

        if is_bool_or_categorical:
            classification_total += 1
            if accuracy_score == 1.0:
                classification_correct += 1

    prediction_quality = (
        total_prediction_quality / total_results if total_results else None
    )
    smape = sum(raw_smape_values) / len(raw_smape_values) if raw_smape_values else None
    normalized_smape_score = (
        sum(normalized_smape_scores) / len(normalized_smape_scores)
        if normalized_smape_scores
        else None
    )
    accuracy = (
        classification_correct / classification_total if classification_total else None
    )
    formula_accuracy = formula_correct / formula_total if formula_total else None
    return {
        "prediction_quality": prediction_quality,
        "smape": smape,
        "normalized_smape_score": normalized_smape_score,
        "bool_categorical_accuracy": accuracy,
        "formula_accuracy": formula_accuracy,
        "result_count": total_results,
        "numeric_count": numeric_count,
        "nonzero_numeric_count": nonzero_numeric_count,
        "zero_reference_numeric_count": zero_reference_numeric_count,
        "smape_count": len(raw_smape_values),
        "normalized_smape_count": len(normalized_smape_scores),
        "bool_categorical_count": classification_total,
        "formula_count": formula_total,
        "missing_predictions": missing,
    }


def compute_file_metrics(
    expected_json: Any,
    actual_json: Any,
    *,
    formula_judge: FormulaJudge | None = None,
) -> dict[str, Any]:
    """Aggregate per-experiment metrics into per-paper metrics.

    For each metric, average the per-experiment values, skipping experiments
    where the metric is undefined (None). Count fields are summed across
    experiments. This gives the intended
    ``mean(mean(experiments in paper) for paper in all_papers)`` semantics
    once the caller averages over papers.
    """
    expected_experiments = _experiments_for_aggregation(expected_json)
    actual_experiments = _experiments_for_aggregation(actual_json)

    per_experiment = []
    for index, expected_experiment in enumerate(expected_experiments):
        actual_experiment = (
            actual_experiments[index] if index < len(actual_experiments) else {}
        )
        per_experiment.append(
            compute_experiment_metrics(
                expected_experiment,
                actual_experiment,
                formula_judge=formula_judge,
            )
        )

    aggregated: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in per_experiment if m[key] is not None]
        aggregated[key] = sum(values) / len(values) if values else None
    for key in COUNT_KEYS:
        aggregated[key] = sum(m[key] for m in per_experiment)
    return aggregated