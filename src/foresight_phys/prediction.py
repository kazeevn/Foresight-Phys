from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from .cache import PredictionCache
from .json_payloads import (
    build_masked_payload,
    build_prediction_cache_key,
    build_prediction_response_format,
)
from .models import BenchmarkItem


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
)
def call_openai_with_retry(
    *,
    model: str,
    system_prompt: str,
    masked_payload: Any,
    response_format: dict[str, Any],
) -> Any:
    client = OpenAI()
    user_prompt = (
        "Fill all TO_PREDICT values with your best predictions. "
        "Return the completed JSON in exactly the required schema under the key 'payload'.\n\n"
        f"{json.dumps(masked_payload, ensure_ascii=False, indent=2)}"
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={"format": response_format},
    )

    content = getattr(response, "output_text", "")
    if not content:
        try:
            for item in response.output:
                for part in item.content:
                    if getattr(part, "type", None) == "output_text" and getattr(part, "text", None):
                        content = part.text
                        break
                if content:
                    break
        except (AttributeError, TypeError):
            pass

    if not content:
        raise ValueError("Model returned an empty structured output.")

    parsed = json.loads(content)
    if not isinstance(parsed, dict) or "payload" not in parsed:
        raise ValueError("Structured output did not contain required 'payload' key.")
    return parsed["payload"]


def build_benchmark_items(
    *,
    json_dir: Path,
    system_prompt: str,
    model: str,
    max_files: int | None,
    max_workers: int,
    prediction_cache: PredictionCache,
) -> list[BenchmarkItem]:
    json_paths = sorted(json_dir.glob("*.json"))
    if max_files is not None:
        json_paths = json_paths[:max_files]

    items_by_file: dict[str, BenchmarkItem] = {}
    tasks: list[tuple[str, Any, Any, dict[str, Any], str]] = []
    for json_path in json_paths:
        ground_truth = json.loads(json_path.read_text(encoding="utf-8"))
        masked_payload = build_masked_payload(ground_truth)
        response_format = build_prediction_response_format(ground_truth)
        cache_key = build_prediction_cache_key(
            model=model,
            system_prompt=system_prompt,
            masked_payload=masked_payload,
            response_format=response_format,
        )

        cached_prediction = prediction_cache.get(cache_key)
        if cached_prediction is not None:
            items_by_file[json_path.name] = BenchmarkItem(
                file_name=json_path.name,
                masked_input=masked_payload,
                expected_output=ground_truth,
                actual_output=cached_prediction,
            )
            continue

        tasks.append((json_path.name, masked_payload, ground_truth, response_format, cache_key))

    if not tasks:
        return [items_by_file[path.name] for path in json_paths]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_openai_with_retry,
                model=model,
                system_prompt=system_prompt,
                masked_payload=masked_payload,
                response_format=response_format,
            ): (file_name, masked_payload, ground_truth, cache_key)
            for file_name, masked_payload, ground_truth, response_format, cache_key in tasks
        }

        with tqdm(total=len(futures), desc="Predicting", unit="file") as progress:
            for future in concurrent.futures.as_completed(futures):
                file_name, masked_payload, ground_truth, cache_key = futures[future]
                predicted_json = future.result()
                items_by_file[file_name] = BenchmarkItem(
                    file_name=file_name,
                    masked_input=masked_payload,
                    expected_output=ground_truth,
                    actual_output=predicted_json,
                )
                prediction_cache.set(cache_key, predicted_json)
                progress.update(1)

    return [items_by_file[path.name] for path in json_paths]