from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

TO_PREDICT_TOKEN = "TO_PREDICT"
MAPE_MIN_ABS_TARGET = 1e-3


@dataclass
class BenchmarkItem:
    file_name: str
    masked_input: Any
    expected_output: Any
    actual_output: Any


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
        unique_item_schemas = list({json.dumps(schema, sort_keys=True): schema for schema in item_schemas}.values())
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


class MapeMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.0
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_model = "deterministic"
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_cost = 0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected_json = json.loads(test_case.expected_output)
        actual_json = json.loads(test_case.actual_output)
        metrics = compute_file_metrics(expected_json, actual_json)
        mape_value = metrics["mape"]
        self.score = float(mape_value) if mape_value is not None else 0.0
        self.reason = (
            f"mape={mape_value}, excluded_numeric_for_mape={metrics['excluded_numeric_for_mape']}, "
            f"numeric_count={metrics['numeric_count']}, missing_predictions={metrics['missing_predictions']}"
        )
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        self.success = True
        return True

    @property
    def __name__(self):
        return "mape"


class BoolCategoricalAccuracyMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.0
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_model = "deterministic"
        self.strict_mode = False
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_cost = 0

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected_json = json.loads(test_case.expected_output)
        actual_json = json.loads(test_case.actual_output)
        metrics = compute_file_metrics(expected_json, actual_json)
        accuracy_value = metrics["bool_categorical_accuracy"]
        self.score = float(accuracy_value) if accuracy_value is not None else 0.0
        self.reason = (
            f"bool_categorical_accuracy={accuracy_value}, "
            f"bool_categorical_count={metrics['bool_categorical_count']}, "
            f"missing_predictions={metrics['missing_predictions']}"
        )
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        self.success = True
        return True

    @property
    def __name__(self):
        return "bool_categorical_accuracy"


def run_deepeval_report(benchmark_items: list[BenchmarkItem]) -> None:
    if not benchmark_items:
        return

    test_cases = [
        LLMTestCase(
            input=json.dumps(item.masked_input, ensure_ascii=False),
            actual_output=json.dumps(item.actual_output, ensure_ascii=False),
            expected_output=json.dumps(item.expected_output, ensure_ascii=False),
            context=[item.file_name],
        )
        for item in benchmark_items
    ]

    evaluate(test_cases=test_cases, metrics=[MapeMetric()])

    bool_case_indices = [
        i
        for i, item in enumerate(benchmark_items)
        if has_bool_or_categorical_targets(item.expected_output)
    ]
    if bool_case_indices:
        bool_test_cases = [test_cases[i] for i in bool_case_indices]
        evaluate(test_cases=bool_test_cases, metrics=[BoolCategoricalAccuracyMetric()])


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
) -> list[BenchmarkItem]:
    json_paths = sorted(json_dir.glob("*.json"))
    if max_files is not None:
        json_paths = json_paths[:max_files]

    tasks: list[tuple[str, Any, Any, dict[str, Any]]] = []
    for json_path in json_paths:
        ground_truth = json.loads(json_path.read_text(encoding="utf-8"))
        masked_payload = build_masked_payload(ground_truth)
        response_format = build_prediction_response_format(ground_truth)
        tasks.append((json_path.name, masked_payload, ground_truth, response_format))

    items_by_file: dict[str, BenchmarkItem] = {}

    if not tasks:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_openai_with_retry,
                model=model,
                system_prompt=system_prompt,
                masked_payload=masked_payload,
                response_format=response_format,
            ): (file_name, masked_payload, ground_truth)
            for file_name, masked_payload, ground_truth, response_format in tasks
        }

        with tqdm(total=len(futures), desc="Predicting", unit="file") as progress:
            for future in concurrent.futures.as_completed(futures):
                file_name, masked_payload, ground_truth = futures[future]
                predicted_json = future.result()
                items_by_file[file_name] = BenchmarkItem(
                    file_name=file_name,
                    masked_input=masked_payload,
                    expected_output=ground_truth,
                    actual_output=predicted_json,
                )
                progress.update(1)

    items = [items_by_file[path.name] for path in json_paths]
    return items


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    benchmark_items = build_benchmark_items(
        json_dir=Path(args.json_dir),
        system_prompt=system_prompt,
        model=args.model,
        max_files=args.max_files,
        max_workers=args.max_workers,
    )

    if args.enable_deepeval:
        run_deepeval_report(benchmark_items)

    rows = []
    for item in benchmark_items:
        metrics = compute_file_metrics(item.expected_output, item.actual_output)
        rows.append(
            {
                "file": item.file_name,
                **metrics,
            }
        )

    mape_values = [row["mape"] for row in rows if row["mape"] is not None]
    accuracy_values = [
        row["bool_categorical_accuracy"]
        for row in rows
        if row["bool_categorical_accuracy"] is not None
    ]

    aggregate_mape = sum(mape_values) / len(mape_values) if mape_values else None
    aggregate_bool_categorical_accuracy = (
        sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
    )
    total_excluded_numeric_values = sum(
        row["excluded_numeric_for_mape"] for row in rows
    )

    summary = {
        "model": args.model,
        "max_workers": args.max_workers,
        "deepeval_enabled": args.enable_deepeval,
        "files_evaluated": len(rows),
        "aggregate_mape": aggregate_mape,
        "aggregate_bool_categorical_accuracy": aggregate_bool_categorical_accuracy,
        "total_excluded_numeric_values_for_mape": total_excluded_numeric_values,
        "per_file": rows,
    }

    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM benchmark over experiment JSON files."
    )
    parser.add_argument("--json-dir", default="JSONs", help="Directory with JSON benchmark files.")
    parser.add_argument(
        "--system-prompt",
        default="system_prompt.txt",
        help="Path to system prompt text file.",
    )
    parser.add_argument("--model", default="gpt-5-nano", help="LLM model name.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum parallel OpenAI requests.",
    )
    parser.add_argument(
        "--disable-deepeval",
        action="store_true",
        help="Disable DeepEval reporting output.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of JSON files for quick runs.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path for summary JSON output.",
    )
    args = parser.parse_args()
    args.enable_deepeval = not args.disable_deepeval
    return args


def main() -> None:
    args = parse_args()
    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"Total excluded numeric values for MAPE (abs(expected) < {MAPE_MIN_ABS_TARGET}): "
        f"{summary['total_excluded_numeric_values_for_mape']}"
    )


if __name__ == "__main__":
    main()
