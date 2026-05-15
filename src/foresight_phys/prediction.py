from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from .cache import PredictionCache
from .extraction import load_response_id_manifest
from .json_payloads import (
    build_prediction_format_signature,
    build_masked_payload,
    build_prediction_cache_key,
    build_prediction_text_format,
)
from .models import BenchmarkItem, BenchmarkPredictionEnvelope
from .paper_titles import extract_arxiv_id_from_file_name, resolve_arxiv_titles


def extract_refusal_text(response: Any) -> str | None:
    try:
        for item in response.output:
            for part in item.content:
                refusal = getattr(part, 'refusal', None)
                if getattr(part, 'type', None) == 'refusal' and isinstance(refusal, str):
                    return refusal
    except (AttributeError, TypeError):
        return None
    return None


def normalize_prediction_payload(parsed: BenchmarkPredictionEnvelope) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for experiment in parsed.payload:
        experiment_results: dict[str, Any] = {}
        for result_field in experiment.experiment_results:
            if result_field.key in experiment_results:
                raise ValueError(
                    f"Structured prediction output repeated result key: {result_field.key}"
                )
            dumped = result_field.model_dump(
                mode='json',
                exclude_none=True,
                exclude={'key'},
            )
            # Flatten the categorical probability list into a dict for storage
            # and downstream scoring; OpenAI structured output requires a list,
            # but {value: prob} is friendlier for analysis.
            probabilities_list = dumped.get('probabilities')
            if isinstance(probabilities_list, list):
                dumped['probabilities'] = {
                    entry['value']: entry['probability']
                    for entry in probabilities_list
                    if isinstance(entry, dict) and 'value' in entry
                }
            # Synthesize a point-estimate `result` from p50 for numeric fields
            # so downstream display / parquet code stays type-agnostic.
            if dumped.get('type') in {'float', 'integer'} and 'p50' in dumped:
                dumped['result'] = dumped['p50']
            experiment_results[result_field.key] = dumped

        payload.append(
            {
                'experiment_description': experiment.experiment_description,
                'experiment_results': experiment_results,
            }
        )

    return payload


def ensure_experiment_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    return [payload]


def restore_payload_shape(*, experiments: list[Any], original_payload: Any) -> Any:
    if isinstance(original_payload, list):
        return experiments
    if not experiments:
        return {}
    return experiments[0]


def should_skip_benchmark_payload(payload: Any) -> bool:
    return isinstance(payload, list) and len(payload) == 0


def load_paper_titles_for_sources(
    source_paths: list[Path],
    *,
    ids_path: Path | None = None,
) -> dict[str, str]:
    manifest_path = ids_path or Path('.cache/extraction_response_ids.json')
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = load_response_id_manifest(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            manifest = {}

    resolved_sources: dict[Path, str] = {}
    for source_path in source_paths:
        try:
            resolved_sources[source_path.resolve()] = source_path.name
        except OSError:
            continue

    titles_by_file: dict[str, str] = {}
    runs = manifest.get('runs', []) if isinstance(manifest, dict) else []
    for run in runs:
        if not isinstance(run, dict):
            continue

        paper_title = run.get('paper_title')
        if not isinstance(paper_title, str) or not paper_title.strip():
            continue

        output_path = run.get('filtered_output_path') or run.get('output_path')
        if not isinstance(output_path, str) or not output_path.strip():
            continue

        try:
            resolved_output_path = Path(output_path).resolve()
        except OSError:
            continue

        file_name = resolved_sources.get(resolved_output_path)
        if file_name is None:
            continue

        titles_by_file[file_name] = paper_title.strip()

    missing_arxiv_ids_by_file: dict[str, str] = {}
    for source_path in source_paths:
        if source_path.name in titles_by_file:
            continue

        arxiv_id = extract_arxiv_id_from_file_name(source_path.name)
        if arxiv_id is None:
            continue
        missing_arxiv_ids_by_file[source_path.name] = arxiv_id

    if missing_arxiv_ids_by_file:
        resolved_arxiv_titles = resolve_arxiv_titles(list(missing_arxiv_ids_by_file.values()))
        for file_name, arxiv_id in missing_arxiv_ids_by_file.items():
            paper_title = resolved_arxiv_titles.get(arxiv_id)
            if paper_title is not None:
                titles_by_file[file_name] = paper_title

    return titles_by_file


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
    text_format: type[BaseModel],
    service_tier: str = 'flex',
) -> Any:
    client = OpenAI()
    user_prompt = (
        "Fill all TO_PREDICT values with your best predictions and calibrated uncertainty. "
        "Return the completed JSON in exactly the required schema under the key 'payload'. "
        "Return exactly one experiment in payload. "
        "For that experiment, return experiment_results as a list of result objects, each with "
        "key, type, description, and the per-type fields below.\n"
        "- type 'float' or 'integer': set 'distribution' ('normal' or 'log_normal') and your "
        "10th/50th/90th percentiles as 'p10', 'p50', 'p90' (with p10 < p50 < p90). For "
        "'log_normal' all three must be > 0 and are taken in linear units (we apply log10 "
        "internally). Aim for true 10/50/90 percentiles: the truth should fall outside "
        "[p10, p90] about 20% of the time. If unsure, err wide.\n"
        "- type 'bool': set 'result' (true/false) and 'prob_true' (probability of true, in [0,1]).\n"
        "- type 'categorical': set 'result' to your best-guess value, 'allowed_categorial_values' "
        "to the list given, and 'probabilities' as a list of {value, probability} covering every "
        "allowed value (summing to ~1).\n"
        "- type 'formula': set 'result' (the symbolic expression) and 'confidence' in [0,1].\n"
        "Predictions are scored with proper scoring rules, so calibrate your uncertainty.\n\n"
        f"{json.dumps(masked_payload, ensure_ascii=False, indent=2)}"
    )

    response = client.responses.parse(
        model=model,
        service_tier=service_tier,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text_format=text_format,
    )

    parsed = getattr(response, 'output_parsed', None)
    if not isinstance(parsed, text_format):
        refusal = extract_refusal_text(response)
        if refusal:
            raise ValueError(f'OpenAI refused prediction request: {refusal}')
        raise ValueError('OpenAI structured output parsing did not return the prediction payload.')

    if not isinstance(parsed, BenchmarkPredictionEnvelope):
        raise ValueError('Prediction parser returned an unexpected payload model.')

    return normalize_prediction_payload(parsed)


def build_benchmark_items(
    *,
    json_dir: Path,
    system_prompt: str,
    model: str,
    max_files: int | None,
    max_workers: int,
    prediction_cache: PredictionCache,
    service_tier: str = 'flex',
) -> list[BenchmarkItem]:
    benchmark_sources: list[tuple[Path, Any, Any]] = []
    for json_path in sorted(json_dir.glob("*.json")):
        ground_truth = json.loads(json_path.read_text(encoding='utf-8'))
        if should_skip_benchmark_payload(ground_truth):
            continue
        benchmark_sources.append((json_path, ground_truth, build_masked_payload(ground_truth)))

    if max_files is not None:
        benchmark_sources = benchmark_sources[:max_files]

    paper_titles_by_file = load_paper_titles_for_sources(
        [json_path for json_path, _, _ in benchmark_sources]
    )

    text_format = build_prediction_text_format()
    format_signature = build_prediction_format_signature()
    predicted_experiments_by_file: dict[str, list[Any]] = {}
    tasks: list[tuple[str, int, Any, str]] = []
    for json_path, ground_truth, masked_payload in benchmark_sources:
        expected_experiments = ensure_experiment_list(ground_truth)
        masked_experiments = ensure_experiment_list(masked_payload)
        if len(masked_experiments) != len(expected_experiments):
            raise ValueError(
                f'Benchmark payload shape mismatch for {json_path.name}: '
                f'{len(masked_experiments)} masked experiments vs '
                f'{len(expected_experiments)} expected experiments.'
            )

        predicted_experiments = [None] * len(expected_experiments)
        predicted_experiments_by_file[json_path.name] = predicted_experiments

        for experiment_index, masked_experiment in enumerate(masked_experiments):
            cache_key = build_prediction_cache_key(
                model=model,
                system_prompt=system_prompt,
                masked_payload=[masked_experiment],
                response_format=format_signature,
            )

            cached_prediction = prediction_cache.get(cache_key)
            if cached_prediction is not None:
                predicted_experiments[experiment_index] = cached_prediction
                continue

            tasks.append((json_path.name, experiment_index, masked_experiment, cache_key))

    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    call_openai_with_retry,
                    model=model,
                    service_tier=service_tier,
                    system_prompt=system_prompt,
                    masked_payload=[masked_experiment],
                    text_format=text_format,
                ): (file_name, experiment_index, cache_key)
                for file_name, experiment_index, masked_experiment, cache_key in tasks
            }

            with tqdm(total=len(futures), desc="Predicting", unit="experiment") as progress:
                for future in concurrent.futures.as_completed(futures):
                    file_name, experiment_index, cache_key = futures[future]
                    predicted_json = future.result()
                    if len(predicted_json) != 1:
                        raise ValueError(
                            f'Expected exactly one experiment prediction for {file_name} '
                            f'experiment {experiment_index}, got {len(predicted_json)}.'
                        )

                    predicted_experiment = predicted_json[0]
                    predicted_experiments_by_file[file_name][experiment_index] = predicted_experiment
                    prediction_cache.set(cache_key, predicted_experiment)
                    progress.update(1)

    items: list[BenchmarkItem] = []
    for json_path, ground_truth, masked_payload in benchmark_sources:
        predicted_experiments = predicted_experiments_by_file[json_path.name]
        missing_experiments = [
            experiment_index
            for experiment_index, prediction in enumerate(predicted_experiments)
            if prediction is None
        ]
        if missing_experiments:
            raise ValueError(
                f'Missing predictions for {json_path.name} experiments {missing_experiments}.'
            )

        items.append(
            BenchmarkItem(
                file_name=json_path.name,
                paper_title=paper_titles_by_file.get(json_path.name),
                masked_input=masked_payload,
                expected_output=ground_truth,
                actual_output=restore_payload_shape(
                    experiments=predicted_experiments,
                    original_payload=ground_truth,
                ),
            )
        )

    return items