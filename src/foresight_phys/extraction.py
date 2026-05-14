from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


BENCHMARK_FILTER_SYSTEM_PROMPT = (
    "Are those experiment descriptions suitable for benchmarking the ability of AIs to predict the results of physical experiments?"
)
DEFAULT_BENCHMARK_FILTER_MODEL = "gpt-5.5"


class ExperimentResultField(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str = Field(description="The name of the measured intrinsic physical or material property. Must represent a predictable scientific outcome, not a contingent artifact. Encode units in the key (e.g., 'optical_bandgap_eV').")
    type: Literal['float', 'integer', 'bool', 'categorical', 'string'] = Field(description="The data type of the result.")
    description: str = Field(description="A precise description of this specific intrinsic measurement. MUST NOT be a subjective assessment, random artifact (like flake thickness), or highly contingent value. DO NOT use this for experimental settings or independent variables.")
    result: StrictFloat | StrictInt | StrictBool | str = Field(description="The actual measured intrinsic value or empirical result obtained from the experiment.")
    allowed_categorial_values: list[str] | None = Field(default=None, description="If type is categorical, list the possible valid string categories.")


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')

    experiment_description: str = Field(description="A rich, self-contained procedure detailing the setup, material, equipment, environmental conditions, sweep ranges, and independent variables. Must provide enough context to predict the intrinsic properties. Must NOT contain the final measured results/outcomes, as these are meant to be predicted.")
    experiment_results: list[ExperimentResultField] = Field(description="The intrinsic empirical findings or dependent variables obtained from the experiment. NEVER include extrinsic/contingent properties (e.g., random thicknesses), subjective qualitative agreements, experimental settings, equipment parameters, or independent variables here.")


class PaperExtractionPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    paper_title: str
    experiments: list[ExperimentRecord]


class ExperimentSuitabilityPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    experiment_validity: list[StrictBool] = Field(
        description=(
            "One boolean per experiment in the same order as the input JSON. "
            "True means the experiment is suitable for benchmarking the ability of "
            "AIs to predict the results of physical experiments."
        )
    )


@dataclass
class ExtractionResult:
    paper_title: str
    experiments: list[dict[str, Any]]
    response_id: str


@dataclass
class FilteringResult:
    filtered_experiments: list[dict[str, Any]]
    validity_by_experiment: list[bool]
    response_id: str


def to_benchmark_experiment(experiment: ExperimentRecord) -> dict[str, Any]:
    experiment_results: dict[str, Any] = {}
    for result_field in experiment.experiment_results:
        if result_field.key in experiment_results:
            raise ValueError(
                f"Duplicate experiment result key in structured output: {result_field.key}"
            )
        experiment_results[result_field.key] = result_field.model_dump(
            mode='json',
            exclude_none=True,
            exclude={'key'},
        )

    return {
        'experiment_description': experiment.experiment_description,
        'experiment_results': experiment_results,
    }


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


def extract_response_id(response: Any) -> str:
    response_id = getattr(response, 'id', None)
    if not isinstance(response_id, str) or not response_id:
        raise ValueError('OpenAI response did not include a response ID.')
    return response_id


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
)
def extract_experiments_from_url(
    *,
    model: str,
    paper_url: str,
    extraction_system_prompt: str,
) -> ExtractionResult:
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": extraction_system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_url": paper_url}
                ],
            },
        ],
        text_format=PaperExtractionPayload,
        service_tier="flex",
    )

    response_id = extract_response_id(response)

    parsed = getattr(response, 'output_parsed', None)
    if not isinstance(parsed, PaperExtractionPayload):
        refusal = extract_refusal_text(response)
        if refusal:
            raise ValueError(f"Model refused extraction request: {refusal}")
        raise ValueError("OpenAI structured output parsing did not return a PaperExtractionPayload.")

    paper_title = parsed.paper_title.strip()
    if not paper_title:
        raise ValueError("Structured output did not contain a paper title.")

    experiments = [to_benchmark_experiment(experiment) for experiment in parsed.experiments]

    return ExtractionResult(
        paper_title=paper_title,
        experiments=experiments,
        response_id=response_id,
    )


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
)
def filter_experiments_for_benchmark(
    *,
    paper_title: str,
    experiments: list[dict[str, Any]],
    model: str = DEFAULT_BENCHMARK_FILTER_MODEL,
) -> FilteringResult:
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'input_text',
                        'text': BENCHMARK_FILTER_SYSTEM_PROMPT,
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': (
                            f'Paper title: {paper_title}\n\n'
                            'Per-experiment JSON:\n'
                            f'{json.dumps(experiments, ensure_ascii=False, indent=2)}'
                        ),
                    }
                ],
            },
        ],
        text_format=ExperimentSuitabilityPayload,
        service_tier='flex',
    )

    response_id = extract_response_id(response)

    parsed = getattr(response, 'output_parsed', None)
    if not isinstance(parsed, ExperimentSuitabilityPayload):
        refusal = extract_refusal_text(response)
        if refusal:
            raise ValueError(f'Model refused benchmark suitability request: {refusal}')
        raise ValueError(
            'OpenAI structured output parsing did not return an ExperimentSuitabilityPayload.'
        )

    validity_by_experiment = list(parsed.experiment_validity)
    if len(validity_by_experiment) != len(experiments):
        raise ValueError(
            'Structured suitability output length did not match the number of experiments.'
        )

    filtered_experiments = [
        experiment
        for experiment, is_valid in zip(experiments, validity_by_experiment)
        if is_valid
    ]

    return FilteringResult(
        filtered_experiments=filtered_experiments,
        validity_by_experiment=validity_by_experiment,
        response_id=response_id,
    )


def sanitize_title_for_filename(title: str) -> str:
    sanitized = re.sub(r"[\x00-\x1f]", "", title).strip()
    sanitized = sanitized.replace("/", "-")
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized or "extracted-paper"


def resolve_raw_output_path(output: str | None, *, title: str) -> Path:
    if output:
        return Path(output)
    return Path('JSONs/raw') / f'{sanitize_title_for_filename(title)}.json'


def resolve_filtered_output_path(output: str | None, *, title: str) -> Path:
    if output:
        return Path(output)
    return Path('JSONs/filtered') / f'{sanitize_title_for_filename(title)}.json'


def validate_output_path(output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f'Refusing to overwrite existing file: {output_path}. Pass --overwrite to replace it.'
        )


def write_extraction_output(output_path: Path, experiments: list[dict[str, Any]], *, overwrite: bool) -> None:
    validate_output_path(output_path, overwrite=overwrite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_response_id(
    ids_path: Path,
    *,
    source_url: str,
    raw_output_path: Path,
    filtered_output_path: Path,
    paper_title: str,
    model: str,
    response_id: str,
    filter_model: str,
    filter_response_id: str,
    validity_by_experiment: list[bool],
    system_prompt_path: Path,
) -> None:
    payload: dict[str, Any] = {"runs": []}
    if ids_path.exists():
        try:
            loaded = json.loads(ids_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
            payload = loaded

    payload["runs"].append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_url": source_url,
            "paper_title": paper_title,
            "output_path": str(filtered_output_path.resolve()),
            "raw_output_path": str(raw_output_path.resolve()),
            "filtered_output_path": str(filtered_output_path.resolve()),
            "model": model,
            "response_id": response_id,
            'filter_model': filter_model,
            'filter_response_id': filter_response_id,
            'validity_by_experiment': validity_by_experiment,
            "system_prompt_path": str(system_prompt_path.resolve()),
        }
    )

    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )