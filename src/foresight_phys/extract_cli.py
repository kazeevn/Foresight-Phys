from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .extraction import (
    DEFAULT_BENCHMARK_FILTER_MODEL,
    extract_experiments_from_url,
    filter_experiments_for_benchmark,
    record_response_id,
    resolve_filtered_output_path,
    resolve_raw_output_path,
    validate_output_path,
    write_extraction_output,
)
from .resources import resolve_extraction_system_prompt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract experiment/result JSON from a public paper PDF URL.'
    )
    parser.add_argument(
        '--paper-url',
        required=True,
        help='Public PDF URL supplied to OpenAI as an input_file, for example an arXiv PDF URL.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Filtered output JSON path. Defaults to JSONs/filtered/<extracted paper title>.json.',
    )
    parser.add_argument(
        '--raw-output',
        default=None,
        help='Raw parsed JSON path. Defaults to JSONs/raw/<extracted paper title>.json.',
    )
    parser.add_argument(
        '--model',
        default='gpt-5.5',
        help='OpenAI model name.',
    )
    parser.add_argument(
        '--system-prompt',
        default=None,
        help='Optional path to extraction system prompt text file.',
    )
    parser.add_argument(
        '--ids-path',
        default='.cache/extraction_response_ids.json',
        help='Local JSON manifest path for storing OpenAI response IDs.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the output JSON file if it already exists.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    extraction_system_prompt_path = resolve_extraction_system_prompt_path(args.system_prompt)
    extraction_system_prompt = extraction_system_prompt_path.read_text(encoding='utf-8').strip()

    extraction = extract_experiments_from_url(
        model=args.model,
        paper_url=args.paper_url,
        extraction_system_prompt=extraction_system_prompt,
    )
    filtering = filter_experiments_for_benchmark(
        paper_title=extraction.paper_title,
        experiments=extraction.experiments,
        model=DEFAULT_BENCHMARK_FILTER_MODEL,
    )
    raw_output_path = resolve_raw_output_path(args.raw_output, title=extraction.paper_title)
    filtered_output_path = resolve_filtered_output_path(args.output, title=extraction.paper_title)

    if raw_output_path.resolve() == filtered_output_path.resolve():
        raise ValueError('Raw and filtered outputs must be different files.')

    validate_output_path(raw_output_path, overwrite=args.overwrite)
    validate_output_path(filtered_output_path, overwrite=args.overwrite)

    write_extraction_output(
        raw_output_path,
        extraction.experiments,
        overwrite=args.overwrite,
    )
    write_extraction_output(
        filtered_output_path,
        filtering.filtered_experiments,
        overwrite=args.overwrite,
    )
    record_response_id(
        Path(args.ids_path),
        source_url=args.paper_url,
        raw_output_path=raw_output_path,
        filtered_output_path=filtered_output_path,
        paper_title=extraction.paper_title,
        model=args.model,
        response_id=extraction.response_id,
        filter_model=DEFAULT_BENCHMARK_FILTER_MODEL,
        filter_response_id=filtering.response_id,
        validity_by_experiment=filtering.validity_by_experiment,
        system_prompt_path=extraction_system_prompt_path,
    )

    print(
        json.dumps(
            {
                'paper_title': extraction.paper_title,
                'source_url': args.paper_url,
                'raw_output_path': str(raw_output_path),
                'filtered_output_path': str(filtered_output_path),
                'extraction_response_id': extraction.response_id,
                'filter_model': DEFAULT_BENCHMARK_FILTER_MODEL,
                'filter_response_id': filtering.response_id,
                'experiments_extracted': len(extraction.experiments),
                'experiments_retained': len(filtering.filtered_experiments),
                'validity_by_experiment': filtering.validity_by_experiment,
                'ids_path': args.ids_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )