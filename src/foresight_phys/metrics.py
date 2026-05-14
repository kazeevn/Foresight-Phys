from __future__ import annotations

from typing import Any

from .constants import MAPE_MIN_ABS_TARGET
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


def has_bool_or_categorical_targets(payload: Any) -> bool:
    for path, expected_value in iter_result_paths(payload):
        result_type = get_result_type(payload, path)
        is_bool_or_categorical = (
            result_type in {"bool", "boolean", "categorical"}
            or isinstance(expected_value, bool)
            or isinstance(expected_value, str)
        )
        if is_bool_or_categorical:
            return True
    return False


def compute_file_metrics(expected_json: Any, actual_json: Any) -> dict[str, Any]:
    numeric_apes: list[float] = []
    classification_total = 0
    classification_correct = 0
    missing = 0
    excluded_numeric_for_mape = 0

    for path, expected_value in iter_result_paths(expected_json):
        found, actual_value = get_at_path(actual_json, path)
        if not found:
            missing += 1
            continue

        result_type = get_result_type(expected_json, path)
        is_numeric = (
            result_type in {"float", "int", "integer", "number"}
            or (isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool))
        )
        is_bool_or_categorical = (
            result_type in {"bool", "boolean", "categorical"}
            or isinstance(expected_value, bool)
            or isinstance(expected_value, str)
        )

        if is_numeric:
            try:
                expected_numeric = float(expected_value)
                actual_numeric = float(actual_value)
            except (TypeError, ValueError):
                continue

            if abs(expected_numeric) < MAPE_MIN_ABS_TARGET:
                excluded_numeric_for_mape += 1
                continue

            denominator = abs(expected_numeric)
            ape = abs(actual_numeric - expected_numeric) / denominator * 100.0
            numeric_apes.append(ape)
            continue

        if is_bool_or_categorical:
            classification_total += 1
            if isinstance(expected_value, bool):
                ok, bool_actual = coerce_bool(actual_value)
                if ok and bool_actual == expected_value:
                    classification_correct += 1
            else:
                if str(actual_value).strip().lower() == str(expected_value).strip().lower():
                    classification_correct += 1

    mape = sum(numeric_apes) / len(numeric_apes) if numeric_apes else None
    accuracy = (
        classification_correct / classification_total if classification_total else None
    )
    return {
        "mape": mape,
        "bool_categorical_accuracy": accuracy,
        "numeric_count": len(numeric_apes),
        "excluded_numeric_for_mape": excluded_numeric_for_mape,
        "bool_categorical_count": classification_total,
        "missing_predictions": missing,
    }