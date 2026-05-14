from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from .constants import TO_PREDICT_TOKEN


def iter_result_paths(node: Any, path: tuple[Any, ...] = ()):
    if isinstance(node, dict):
        for key, value in node.items():
            new_path = (*path, key)
            if key == "result":
                yield new_path, value
            else:
                yield from iter_result_paths(value, new_path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_result_paths(value, (*path, index))


def get_at_path(node: Any, path: tuple[Any, ...]) -> tuple[bool, Any]:
    current = node
    for step in path:
        if isinstance(step, int):
            if not isinstance(current, list) or step >= len(current):
                return False, None
            current = current[step]
        else:
            if not isinstance(current, dict) or step not in current:
                return False, None
            current = current[step]
    return True, current


def set_at_path(node: Any, path: tuple[Any, ...], value: Any) -> None:
    current = node
    for step in path[:-1]:
        current = current[step]
    current[path[-1]] = value


def build_masked_payload(ground_truth: Any) -> Any:
    masked = copy.deepcopy(ground_truth)
    for result_path, _ in iter_result_paths(masked):
        set_at_path(masked, result_path, TO_PREDICT_TOKEN)
    return masked


def scalar_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if value is None:
        return {"type": "null"}
    return {}


def build_json_schema_from_example(example: Any) -> dict[str, Any]:
    if isinstance(example, dict):
        properties: dict[str, Any] = {}
        required: list[str] = []
        for key, value in example.items():
            properties[key] = build_json_schema_from_example(value)
            required.append(key)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    if isinstance(example, list):
        if not example:
            return {"type": "array", "items": {}}

        item_schemas = [build_json_schema_from_example(item) for item in example]
        unique_item_schemas = list(
            {json.dumps(schema, sort_keys=True): schema for schema in item_schemas}.values()
        )
        items_schema = (
            unique_item_schemas[0]
            if len(unique_item_schemas) == 1
            else {"anyOf": unique_item_schemas}
        )
        return {
            "type": "array",
            "items": items_schema,
        }

    return scalar_schema(example)


def build_prediction_response_format(ground_truth: Any) -> dict[str, Any]:
    payload_schema = build_json_schema_from_example(ground_truth)
    schema = {
        "type": "object",
        "properties": {"payload": payload_schema},
        "required": ["payload"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "name": "experiment_predictions",
        "strict": True,
        "schema": schema,
    }


def build_prediction_cache_key(
    *,
    model: str,
    system_prompt: str,
    masked_payload: Any,
    response_format: dict[str, Any],
) -> str:
    canonical_payload = {
        "model": model,
        "system_prompt": system_prompt,
        "masked_payload": masked_payload,
        "response_format": response_format,
    }
    canonical_json = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()