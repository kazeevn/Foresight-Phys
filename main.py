from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import html
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
class PredictionCache:
    enabled: bool
    path: Path
    entries: dict[str, Any]
    warning: str | None = None
    hits: int = 0
    misses: int = 0
    _dirty: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PredictionCache":
        cache_path = Path(args.cache_path)
        if args.disable_cache:
            return cls(enabled=False, path=cache_path, entries={})

        if cache_path.exists():
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return cls(enabled=True, path=cache_path, entries=loaded)
                return cls(
                    enabled=True,
                    path=cache_path,
                    entries={},
                    warning=f"Cache reset: expected JSON object at {cache_path}.",
                )
            except Exception as exc:
                return cls(
                    enabled=True,
                    path=cache_path,
                    entries={},
                    warning=f"Cache reset: failed reading {cache_path} ({exc}).",
                )

        return cls(enabled=True, path=cache_path, entries={})

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        if key in self.entries:
            self.hits += 1
            return copy.deepcopy(self.entries[key])
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.entries[key] = copy.deepcopy(value)
        self._dirty = True

    def flush(self) -> None:
        if not self.enabled or not self._dirty:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temp_path.write_text(
                json.dumps(self.entries, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temp_path.replace(self.path)
            self._dirty = False
        except Exception as exc:
            self.warning = f"Cache write warning: {exc}"

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "warning": self.warning,
        }


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
                    "predicted": item.actual_output
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
                        "predicted": item.actual_output
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

    items = [items_by_file[path.name] for path in json_paths]
    return items


def write_human_readable_report(
    *,
    items: list[BenchmarkItem],
    output_path: Path,
    model: str,
    aggregate_mape: float | None,
    aggregate_bool_categorical_accuracy: float | None,
) -> None:
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    def format_metric(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    def display_file_title(file_name: str) -> str:
        if file_name.lower().endswith(".json"):
            return file_name[:-5]
        return file_name

    def format_value(value: Any) -> str:
        if value is None:
            return "MISSING"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def normalize_for_comparison(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def values_match(expected_value: Any, actual_value: Any) -> bool:
        if expected_value is None and actual_value is None:
            return True
        return normalize_for_comparison(expected_value) == normalize_for_comparison(actual_value)

    def collect_file_section(item: BenchmarkItem) -> str:
        metrics = compute_file_metrics(item.expected_output, item.actual_output)
        expected_experiments = item.expected_output if isinstance(item.expected_output, list) else [item.expected_output]
        actual_experiments = item.actual_output if isinstance(item.actual_output, list) else [item.actual_output]

        section_parts: list[str] = []
        section_parts.append("<div class=\"paper-metrics\">")
        section_parts.append(
            f"<div class=\"metric-chip\"><span class=\"metric-label\">MAPE</span><span class=\"metric-value\">{html.escape(format_metric(metrics.get('mape')))}</span></div>"
        )
        section_parts.append(
            f"<div class=\"metric-chip\"><span class=\"metric-label\">Bool/Categorical Accuracy</span><span class=\"metric-value\">{html.escape(format_metric(metrics.get('bool_categorical_accuracy')))}</span></div>"
        )
        section_parts.append("</div>")

        for experiment_index, expected_exp in enumerate(expected_experiments, start=1):
            actual_exp = actual_experiments[experiment_index - 1] if experiment_index - 1 < len(actual_experiments) else {}
            expected_desc = ""
            if isinstance(expected_exp, dict):
                expected_desc = str(expected_exp.get("experiment_description", ""))

            expected_results = expected_exp.get("experiment_results", {}) if isinstance(expected_exp, dict) else {}
            actual_results = actual_exp.get("experiment_results", {}) if isinstance(actual_exp, dict) else {}

            section_parts.append('<article class="experiment-card">')
            section_parts.append(f"<h3>Experiment {experiment_index}</h3>")
            section_parts.append(
                f"<p class=\"experiment-description\">{html.escape(expected_desc)}</p>"
            )

            if not isinstance(expected_results, dict) or not expected_results:
                section_parts.append('<p class="empty-results">No result fields found.</p>')
                section_parts.append("</article>")
                continue

            section_parts.append("<div class=\"table-wrap\">")
            section_parts.append(
                "<table><thead><tr><th>Result</th><th>Description</th><th>Ground Truth</th><th>Predicted</th><th>Status</th></tr></thead><tbody>"
            )

            for field_name, expected_meta in expected_results.items():
                expected_meta_dict = expected_meta if isinstance(expected_meta, dict) else {}
                field_description = str(expected_meta_dict.get("description", ""))
                field_type = str(expected_meta_dict.get("type", "")).strip().lower()
                expected_value = expected_meta_dict.get("result")

                actual_meta = actual_results.get(field_name, {}) if isinstance(actual_results, dict) else {}
                actual_meta_dict = actual_meta if isinstance(actual_meta, dict) else {}
                actual_value = actual_meta_dict.get("result")

                is_numeric_expected = isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool)
                is_numeric_actual = isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool)

                if (
                    field_type == "float"
                    and is_numeric_expected
                    and abs(float(expected_value)) > MAPE_MIN_ABS_TARGET
                ):
                    status_class = "status-relative-error"
                    if is_numeric_actual:
                        relative_error = (
                            abs(float(actual_value) - float(expected_value))
                            / abs(float(expected_value))
                        ) * 100
                        status_text = f"rel err {relative_error:.2f}%"
                    else:
                        status_text = "rel err n/a"
                else:
                    match = values_match(expected_value, actual_value)
                    status_text = "match" if match else "mismatch"
                    status_class = "status-match" if match else "status-mismatch"

                section_parts.append(
                    "<tr>"
                    f"<td>{html.escape(str(field_name))}</td>"
                    f"<td>{html.escape(field_description)}</td>"
                    f"<td>{html.escape(format_value(expected_value))}</td>"
                    f"<td>{html.escape(format_value(actual_value))}</td>"
                    f"<td><span class=\"{status_class}\">{status_text}</span></td>"
                    "</tr>"
                )

            section_parts.append("</tbody></table></div>")
            section_parts.append("</article>")

        return "\n".join(section_parts)

    sections = [collect_file_section(item) for item in items]

    sidebar_buttons: list[str] = []
    paper_panels: list[str] = []
    for index, item in enumerate(items):
        active_class = " is-active" if index == 0 else ""
        button_escaped = html.escape(display_file_title(item.file_name))
        sidebar_buttons.append(
            f'<button class="paper-tab{active_class}" data-paper-id="paper-{index}" type="button">{button_escaped}</button>'
        )
        paper_panels.append(
            "\n".join(
                [
                    f'<section class="paper-panel{active_class}" id="paper-{index}">',
                    f'<h2>{button_escaped}</h2>',
                    sections[index],
                    "</section>",
                ]
            )
        )

    if not items:
        sidebar_html = '<div class="empty-sidebar">No files evaluated.</div>'
        panels_html = '<section class="paper-panel is-active" id="paper-empty"><p>No files evaluated.</p></section>'
    else:
        sidebar_html = "\n".join(sidebar_buttons)
        panels_html = "\n".join(paper_panels)

    report_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Foresight-Phys Human-Readable Report</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #ffffff;
            --surface: #ffffff;
            --text: #000000;
            --muted: #333333;
            --border: #cccccc;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 20px;
            font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }}
        .header {{ margin-bottom: 14px; }}
        .title {{ font-size: 24px; font-weight: 700; margin: 0 0 8px; }}
        .meta {{ color: var(--muted); font-size: 14px; line-height: 1.5; }}
        .layout {{
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 16px;
            margin-top: 16px;
            min-height: calc(100vh - 190px);
        }}
        .sidebar {{
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            background: var(--surface);
        }}
        .paper-tab {{
            width: 100%;
            text-align: left;
            border: 1px solid var(--border);
            background: #f7f7f7;
            color: var(--text);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            cursor: pointer;
            font-size: 13px;
            line-height: 1.35;
        }}
        .paper-tab:hover {{ background: #efefef; }}
        .paper-tab.is-active {{
            background: #e9f2ff;
            border-color: #8bb8ff;
            font-weight: 600;
        }}
        .content {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--surface);
            padding: 14px;
            overflow: auto;
        }}
        .paper-panel {{ display: none; }}
        .paper-panel.is-active {{ display: block; }}
        .paper-panel h2 {{ margin: 0 0 10px; font-size: 22px; }}
        .paper-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .metric-chip {{
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 13px;
            background: #fafafa;
        }}
        .metric-label {{ color: var(--muted); margin-right: 8px; }}
        .metric-value {{ font-weight: 700; }}
        .experiment-card {{
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            background: #fcfcfc;
        }}
        .experiment-card h3 {{ margin: 0 0 8px; font-size: 18px; }}
        .experiment-description {{ margin: 0 0 10px; color: #111; }}
        .empty-results {{ color: var(--muted); margin: 0; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 8px;
            vertical-align: top;
            text-align: left;
        }}
        th {{ background: #f3f3f3; }}
        .status-match {{ color: #0b7d2b; font-weight: 700; }}
        .status-mismatch {{ color: #b3261e; font-weight: 700; }}
        .status-relative-error {{ color: #0b57d0; font-weight: 700; }}
        @media (max-width: 980px) {{
            .layout {{ grid-template-columns: 1fr; }}
            .sidebar {{ order: 1; }}
            .content {{ order: 2; }}
        }}
    </style>
</head>
<body>
    <div class=\"header\">
        <h1 class=\"title\">Offline Prediction vs Reference Report</h1>
        <div class=\"meta\">Model: {html.escape(model)}</div>
        <div class=\"meta\">Generated: {html.escape(generated_at_utc)}</div>
        <div class=\"meta\">Files: {len(items)}</div>
        <div class=\"meta\">Aggregate MAPE: {format_metric(aggregate_mape)}</div>
        <div class=\"meta\">Aggregate bool/categorical accuracy: {format_metric(aggregate_bool_categorical_accuracy)}</div>
    </div>
    <div class=\"layout\">
        <aside class=\"sidebar\" aria-label=\"Papers\">
            {sidebar_html}
        </aside>
        <main class=\"content\">
            {panels_html}
        </main>
    </div>
    <script>
        const tabs = Array.from(document.querySelectorAll('.paper-tab'));
        const panels = Array.from(document.querySelectorAll('.paper-panel'));
        function activatePanel(panelId) {{
            tabs.forEach((tab) => tab.classList.toggle('is-active', tab.dataset.paperId === panelId));
            panels.forEach((panel) => panel.classList.toggle('is-active', panel.id === panelId));
        }}
        tabs.forEach((tab) => {{
            tab.addEventListener('click', () => activatePanel(tab.dataset.paperId));
        }});
    </script>
</body>
</html>
"""

    output_path.write_text(report_html, encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    langfuse_logger = LangfuseRunLogger.from_args(args)
    prediction_cache = PredictionCache.from_args(args)

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    benchmark_items = build_benchmark_items(
        json_dir=Path(args.json_dir),
        system_prompt=system_prompt,
        model=args.model,
        max_files=args.max_files,
        max_workers=args.max_workers,
        prediction_cache=prediction_cache,
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
        "human_readable_report": args.html_output,
        "cache": prediction_cache.summary(),
        "langfuse": langfuse_logger.summary(),
        "per_file": rows,
    }

    if args.html_output:
        write_human_readable_report(
            items=benchmark_items,
            output_path=Path(args.html_output),
            model=args.model,
            aggregate_mape=aggregate_mape,
            aggregate_bool_categorical_accuracy=aggregate_bool_categorical_accuracy,
        )

    langfuse_logger.log_run_summary(summary, args)
    langfuse_logger.flush()
    prediction_cache.flush()

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
        default="docs/benchmark_results.json",
        help="Path for summary JSON output.",
    )
    parser.add_argument(
        "--html-output",
        default="docs/benchmark_human_readable_report.html",
        help="Path for static offline HTML human-readable report (set empty string to disable).",
    )
    parser.add_argument(
        "--cache-path",
        default=".cache/llm_predictions.json",
        help="Path to persistent cache for raw LLM predictions.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable local prediction cache and always call the LLM.",
    )
    args = parser.parse_args()
    if args.html_output == "":
        args.html_output = None
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
