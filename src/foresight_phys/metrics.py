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


def compute_log_accuracy(expected_numeric: float, actual_numeric: float) -> float:
    if actual_numeric == 0.0:
        return float("inf")
    # We use abs() on both to handle potential negative physical quantities,
    # though they are typically positive in this benchmark.
    return abs(math.log10(abs(actual_numeric) / abs(expected_numeric)))


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
    "log_accuracy",
    "normalized_log_accuracy_score",
    "bool_categorical_accuracy",
    "formula_accuracy",
)

COUNT_KEYS = (
    "result_count",
    "numeric_count",
    "nonzero_numeric_count",
    "zero_reference_numeric_count",
    "log_accuracy_count",
    "normalized_log_accuracy_count",
    "bool_categorical_count",
    "formula_count",
    "missing_predictions",
)


def _empty_experiment_metrics() -> dict[str, Any]:
    return {
        "prediction_quality": None,
        "log_accuracy": None,
        "normalized_log_accuracy_score": None,
        "bool_categorical_accuracy": None,
        "formula_accuracy": None,
        "result_count": 0,
        "numeric_count": 0,
        "nonzero_numeric_count": 0,
        "zero_reference_numeric_count": 0,
        "log_accuracy_count": 0,
        "normalized_log_accuracy_count": 0,
        "bool_categorical_count": 0,
        "formula_count": 0,
        "missing_predictions": 0,
    }


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

    raw_log_accuracy_values: list[float] = []
    normalized_log_accuracy_scores: list[float] = []
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
    result_rows: list[dict[str, Any]] = []

    for field_name, expected_meta in expected_results.items():
        total_results += 1
        expected_meta_dict = expected_meta if isinstance(expected_meta, dict) else {}
        field_description = str(expected_meta_dict.get("description", ""))
        field_type = str(expected_meta_dict.get("type", "")).strip().lower()
        expected_value = expected_meta_dict.get("result")

        actual_meta = actual_results.get(field_name, {}) if isinstance(actual_results, dict) else {}
        actual_meta_dict = actual_meta if isinstance(actual_meta, dict) else {}
        actual_value = actual_meta_dict.get("result")
        field_found = isinstance(actual_results, dict) and field_name in actual_results
        if not field_found:
            missing += 1

        is_numeric_expected = is_numeric_result(field_type, expected_value)
        status_title = ""

        if is_numeric_expected:
            expected_ok, expected_numeric = coerce_numeric(expected_value)
            if expected_ok:
                numeric_count += 1
            actual_ok, actual_numeric = (
                coerce_numeric(actual_value) if field_found else (False, 0.0)
            )

            if expected_ok and expected_numeric == 0.0:
                zero_reference_numeric_count += 1
                zero_match = actual_ok and actual_numeric == 0.0
                if zero_match:
                    total_prediction_quality += 1.0
                status_text = 'zero match' if zero_match else 'zero mismatch'
                status_class = 'status-match' if zero_match else 'status-mismatch'
            elif expected_ok:
                nonzero_numeric_count += 1
                if actual_ok:
                    log_acc = compute_log_accuracy(expected_numeric, actual_numeric)
                    normalized_score = 1.0 - min(log_acc, 1.0)
                    raw_log_accuracy_values.append(log_acc)
                    normalized_log_accuracy_scores.append(normalized_score)
                    total_prediction_quality += normalized_score
                    status_text = f'Log-Acc {log_acc:.4f}'
                    status_class = 'status-numeric'
                else:
                    normalized_log_accuracy_scores.append(0.0)
                    status_text = 'Log-Acc n/a'
                    status_class = 'status-mismatch'
            else:
                match = values_match(expected_value, actual_value)
                if match:
                    total_prediction_quality += 1.0
                status_text = 'match' if match else 'mismatch'
                status_class = 'status-match' if match else 'status-mismatch'
        elif is_formula_result(field_type):
            formula_total += 1
            formula_judgment = judge_formula_values(
                expected_formula=expected_value,
                actual_formula=actual_value,
                experiment_description=expected_desc,
                result_key=str(field_name),
                result_description=field_description,
                formula_judge=formula_judge,
            )
            if formula_judgment.equivalent:
                formula_correct += 1
                total_prediction_quality += 1.0
            status_text = 'formula match' if formula_judgment.equivalent else 'formula mismatch'
            status_class = 'status-match' if formula_judgment.equivalent else 'status-mismatch'
            status_title = formula_judgment.explanation
        else:
            match = field_found and values_match(expected_value, actual_value)
            if match:
                total_prediction_quality += 1.0

            if is_bool_or_categorical_result(field_type, expected_value):
                classification_total += 1
                if match:
                    classification_correct += 1

            status_text = 'match' if match else 'mismatch'
            status_class = 'status-match' if match else 'status-mismatch'

        result_rows.append(
            {
                "result_key": str(field_name),
                "description": field_description,
                "ground_truth": expected_value,
                "predicted": actual_value,
                "status_text": status_text,
                "status_class": status_class,
                "status_title": status_title or None,
            }
        )

    metrics = {
        "prediction_quality": total_prediction_quality / total_results if total_results else None,
        "log_accuracy": (
            sum(raw_log_accuracy_values) / len(raw_log_accuracy_values)
            if raw_log_accuracy_values
            else None
        ),
        "normalized_log_accuracy_score": (
            sum(normalized_log_accuracy_scores) / len(normalized_log_accuracy_scores)
            if normalized_log_accuracy_scores
            else None
        ),
        "bool_categorical_accuracy": (
            classification_correct / classification_total if classification_total else None
        ),
        "formula_accuracy": formula_correct / formula_total if formula_total else None,
        "result_count": total_results,
        "numeric_count": numeric_count,
        "nonzero_numeric_count": nonzero_numeric_count,
        "zero_reference_numeric_count": zero_reference_numeric_count,
        "log_accuracy_count": len(raw_log_accuracy_values),
        "normalized_log_accuracy_count": len(normalized_log_accuracy_scores),
        "bool_categorical_count": classification_total,
        "formula_count": formula_total,
        "missing_predictions": missing,
    }
    return {
        "experiment_description": expected_desc,
        "result_rows": result_rows,
        "metrics": metrics,
    }


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
    """Aggregate per-experiment metrics into per-paper metrics."""
    report_data = build_file_report_data(
        expected_json,
        actual_json,
        formula_judge=formula_judge,
    )
    return {key: report_data[key] for key in (*METRIC_KEYS, *COUNT_KEYS)}