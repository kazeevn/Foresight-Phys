from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from .constants import TO_PREDICT_TOKEN
from .models import BenchmarkPredictionEnvelope


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


# Description shown in the name-only ablation: strips all experiment-specific
# context so the model must predict from the typed result key (with its embedded
# units) and allowed values alone. The gap between the full-context score and this
# baseline is the "foresight lift" — the predictive content carried by the
# experiment description over generic observable-type priors.
NAME_ONLY_DESCRIPTION = (
    "No experiment context is provided. Predict each physical quantity from its "
    "result key (which encodes the observable and its units), its type, and any "
    "allowed values alone, using only general physics priors."
)


def build_name_only_payload(ground_truth: Any) -> Any:
    """Masked payload with the experiment description and result descriptions removed.

    Keeps the result keys, types, units (encoded in keys), and allowed categorical
    values so the schema and target set are identical to the full run; only the
    experiment-specific context is withheld.
    """
    masked = build_masked_payload(ground_truth)
    experiments = masked if isinstance(masked, list) else [masked]
    for experiment in experiments:
        if not isinstance(experiment, dict):
            continue
        experiment["experiment_description"] = NAME_ONLY_DESCRIPTION
        results = experiment.get("experiment_results")
        if isinstance(results, dict):
            for meta in results.values():
                if isinstance(meta, dict) and "description" in meta:
                    meta["description"] = ""
    return masked


def build_prediction_text_format() -> type[BenchmarkPredictionEnvelope]:
    return BenchmarkPredictionEnvelope


def build_prediction_format_signature() -> dict[str, Any]:
    return BenchmarkPredictionEnvelope.model_json_schema()


def build_prediction_cache_key(
    *,
    model: str,
    system_prompt: str,
    masked_payload: Any,
    response_format: dict[str, Any],
    include_system_prompt: bool = True,
) -> str:
    canonical_payload = {
        "model": model,
        "masked_payload": masked_payload,
        "response_format": response_format,
    }
    if include_system_prompt:
        canonical_payload["system_prompt"] = system_prompt
    canonical_json = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
