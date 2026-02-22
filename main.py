from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

LANGFUSE_IMPORT_ERROR: str | None = None
try:
    from langfuse import Langfuse
except Exception as exc:
    Langfuse = None
    LANGFUSE_IMPORT_ERROR = str(exc)

TO_PREDICT_TOKEN = "TO_PREDICT"
MAPE_MIN_ABS_TARGET = 1e-3


@dataclass
class BenchmarkItem:
    file_name: str
    masked_input: Any
    expected_output: Any
    actual_output: Any


@dataclass
class LangfuseRunLogger:
    enabled: bool
    model: str
    run_name: str | None = None
    session_id: str | None = None
    host: str | None = None
    client: Any | None = None
    warning: str | None = None
    trace_urls: list[str] | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LangfuseRunLogger":
        if args.disable_langfuse:
            return cls(enabled=False, model=args.model)

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = args.langfuse_host or os.getenv("LANGFUSE_HOST")

        if not public_key or not secret_key:
            return cls(
                enabled=False,
                model=args.model,
                warning="Langfuse disabled: LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY not set.",
            )

        if Langfuse is None:
            import_detail = f" ({LANGFUSE_IMPORT_ERROR})" if LANGFUSE_IMPORT_ERROR else ""
            return cls(
                enabled=False,
                model=args.model,
                warning=f"Langfuse disabled: package import failed{import_detail}.",
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = args.langfuse_run_name or f"foresight-phys-{timestamp}"
        session_id = f"{run_name}-{uuid.uuid4().hex[:8]}"

        client_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            client_kwargs["host"] = host

        try:
            client = Langfuse(**client_kwargs)
        except Exception as exc:
            return cls(
                enabled=False,
                model=args.model,
                warning=f"Langfuse disabled: failed to initialize client ({exc}).",
            )

        return cls(
            enabled=True,
            model=args.model,
            run_name=run_name,
            session_id=session_id,
            host=host,
            client=client,
            trace_urls=[],
        )

    def log_file_result(self, item: BenchmarkItem, metrics: dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return

        try:
            with self.client.start_as_current_observation(
                name="llm-prediction",
                as_type="generation",
                model=self.model,
                input={
                    "file": item.file_name,
                    "masked_input": item.masked_input,
                },
                output={
                    "predicted": item.actual_output,
                    "reference": item.expected_output,
                },
                metadata={
                    "file": item.file_name,
                    "run_name": self.run_name,
                    "metrics": metrics,
                },
            ) as generation:
                generation.update_trace(
                    name="foresight-phys.file-eval",
                    session_id=self.session_id,
                    input={
                        "file": item.file_name,
                        "masked_input": item.masked_input,
                    },
                    output={
                        "predicted": item.actual_output,
                        "reference": item.expected_output,
                    },
                    metadata={
                        "file": item.file_name,
                        "model": self.model,
                        "run_name": self.run_name,
                        **metrics,
                    },
                    tags=["foresight-phys", "benchmark", "file-eval"],
                )

                if metrics.get("mape") is not None:
                    generation.score(
                    name="mape",
                    value=float(metrics["mape"]),
                    data_type="NUMERIC",
                    comment="Lower is better",
                )

                accuracy = metrics.get("bool_categorical_accuracy")
                if accuracy is not None:
                    generation.score(
                        name="bool_categorical_accuracy",
                        value=float(accuracy),
                        data_type="NUMERIC",
                        comment="Higher is better",
                    )

                correction_payload = json.dumps(
                    item.expected_output,
                    ensure_ascii=False,
                    indent=2,
                )
                correction_common = {
                    "name": "output",
                    "value": correction_payload,
                    "dataType": "CORRECTION",
                    "source": "ANNOTATION",
                    "comment": "Ground-truth expected output",
                    "metadata": {
                        "file": item.file_name,
                        "run_name": self.run_name,
                    },
                }

                self.client.api.score.create(
                    request={
                        **correction_common,
                        "traceId": generation.trace_id,
                    }
                )

                self.client.api.score.create(
                    request={
                        **correction_common,
                        "traceId": generation.trace_id,
                        "observationId": generation.id,
                    }
                )

                if self.trace_urls is not None:
                    trace_url = self.client.get_trace_url(trace_id=generation.trace_id)
                    if trace_url:
                        self.trace_urls.append(trace_url)
        except Exception as exc:
            self.warning = f"Langfuse logging warning: {exc}"

    def log_run_summary(self, summary: dict[str, Any], args: argparse.Namespace) -> None:
        if not self.enabled or self.client is None:
            return

        try:
            with self.client.start_as_current_span(
                name="foresight-phys.run-summary",
                input={
                    "json_dir": args.json_dir,
                    "system_prompt": args.system_prompt,
                    "model": args.model,
                    "max_files": args.max_files,
                    "max_workers": args.max_workers,
                },
                output=summary,
                metadata={
                    "run_name": self.run_name,
                },
            ) as span:
                span.update_trace(
                    name="foresight-phys.run-summary",
                    session_id=self.session_id,
                    tags=["foresight-phys", "benchmark", "run-summary"],
                    metadata={"run_name": self.run_name},
                )

                if self.trace_urls is not None:
                    trace_url = self.client.get_trace_url(trace_id=span.trace_id)
                    if trace_url:
                        self.trace_urls.append(trace_url)
        except Exception as exc:
            self.warning = f"Langfuse logging warning: {exc}"

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exc:
            self.warning = f"Langfuse flush warning: {exc}"

    def summary(self) -> dict[str, Any]:
        primary_trace_url = self.trace_urls[0] if self.trace_urls else None
        return {
            "enabled": self.enabled,
            "run_name": self.run_name,
            "session_id": self.session_id,
            "host": self.host,
            "trace_url": primary_trace_url,
            "warning": self.warning,
        }


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
    langfuse_logger = LangfuseRunLogger.from_args(args)

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    benchmark_items = build_benchmark_items(
        json_dir=Path(args.json_dir),
        system_prompt=system_prompt,
        model=args.model,
        max_files=args.max_files,
        max_workers=args.max_workers,
    )

    rows = []
    for item in benchmark_items:
        metrics = compute_file_metrics(item.expected_output, item.actual_output)
        langfuse_logger.log_file_result(item, metrics)
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
        "files_evaluated": len(rows),
        "aggregate_mape": aggregate_mape,
        "aggregate_bool_categorical_accuracy": aggregate_bool_categorical_accuracy,
        "total_excluded_numeric_values_for_mape": total_excluded_numeric_values,
        "langfuse": langfuse_logger.summary(),
        "per_file": rows,
    }

    langfuse_logger.log_run_summary(summary, args)
    langfuse_logger.flush()

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
        "--disable-langfuse",
        action="store_true",
        help="Disable Langfuse tracing output.",
    )
    parser.add_argument(
        "--langfuse-host",
        default=None,
        help="Optional Langfuse host URL override (otherwise LANGFUSE_HOST is used).",
    )
    parser.add_argument(
        "--langfuse-run-name",
        default=None,
        help="Optional run name for grouping traces in Langfuse dashboard.",
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
    return parser.parse_args()


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
