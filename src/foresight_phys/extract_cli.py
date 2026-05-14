from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--paper-url',
        help='Public PDF URL supplied to OpenAI as an input_file, for example an arXiv PDF URL.',
    )
    source_group.add_argument(
        '--paper-urls-file',
        default=None,
        help='Text file containing one public PDF URL per line. Blank lines and lines starting with # are ignored.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Filtered output JSON path. Defaults to JSONs/filtered/<extracted paper title>.json. Only valid for single-paper runs.',
    )
    parser.add_argument(
        '--raw-output',
        default=None,
        help='Raw parsed JSON path. Defaults to JSONs/raw/<extracted paper title>.json. Only valid for single-paper runs.',
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


def read_paper_urls(args: argparse.Namespace) -> list[str]:
    if args.paper_url:
        return [args.paper_url]

    paper_urls_path = Path(args.paper_urls_file)
    paper_urls = [
        line
        for raw_line in paper_urls_path.read_text(encoding='utf-8').splitlines()
        if (line := raw_line.strip()) and not line.startswith('#')
    ]
    if not paper_urls:
        raise ValueError(f'No paper URLs found in {paper_urls_path}.')

    return paper_urls


def extract_single_url(
    *,
    paper_url: str,
    model: str,
    extraction_system_prompt: str,
    extraction_system_prompt_path: Path,
    raw_output_override: str | None,
    filtered_output_override: str | None,
    ids_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    extraction = extract_experiments_from_url(
        model=model,
        paper_url=paper_url,
        extraction_system_prompt=extraction_system_prompt,
    )
    filtering = filter_experiments_for_benchmark(
        paper_title=extraction.paper_title,
        experiments=extraction.experiments,
        model=DEFAULT_BENCHMARK_FILTER_MODEL,
    )
    raw_output_path = resolve_raw_output_path(raw_output_override, title=extraction.paper_title)
    filtered_output_path = resolve_filtered_output_path(
        filtered_output_override,
        title=extraction.paper_title,
    )

    if raw_output_path.resolve() == filtered_output_path.resolve():
        raise ValueError('Raw and filtered outputs must be different files.')

    validate_output_path(raw_output_path, overwrite=overwrite)
    validate_output_path(filtered_output_path, overwrite=overwrite)

    write_extraction_output(
        raw_output_path,
        extraction.experiments,
        overwrite=overwrite,
    )
    write_extraction_output(
        filtered_output_path,
        filtering.filtered_experiments,
        overwrite=overwrite,
    )
    record_response_id(
        ids_path,
        source_url=paper_url,
        raw_output_path=raw_output_path,
        filtered_output_path=filtered_output_path,
        paper_title=extraction.paper_title,
        model=model,
        response_id=extraction.response_id,
        filter_model=DEFAULT_BENCHMARK_FILTER_MODEL,
        filter_response_id=filtering.response_id,
        validity_by_experiment=filtering.validity_by_experiment,
        system_prompt_path=extraction_system_prompt_path,
    )

    return {
        'paper_title': extraction.paper_title,
        'source_url': paper_url,
        'raw_output_path': str(raw_output_path),
        'filtered_output_path': str(filtered_output_path),
        'extraction_response_id': extraction.response_id,
        'filter_model': DEFAULT_BENCHMARK_FILTER_MODEL,
        'filter_response_id': filtering.response_id,
        'experiments_extracted': len(extraction.experiments),
        'experiments_retained': len(filtering.filtered_experiments),
        'validity_by_experiment': filtering.validity_by_experiment,
        'ids_path': str(ids_path),
    }


def main() -> None:
    args = parse_args()
    load_dotenv()

    extraction_system_prompt_path = resolve_extraction_system_prompt_path(args.system_prompt)
    extraction_system_prompt = extraction_system_prompt_path.read_text(encoding='utf-8').strip()
    paper_urls = read_paper_urls(args)

    if len(paper_urls) > 1 and (args.output or args.raw_output):
        raise ValueError(
            'The --output and --raw-output options are only supported for single-paper runs. '
            'Use default title-derived filenames when extracting multiple URLs from a file.'
        )

    ids_path = Path(args.ids_path)
    results = [
        extract_single_url(
            paper_url=paper_url,
            model=args.model,
            extraction_system_prompt=extraction_system_prompt,
            extraction_system_prompt_path=extraction_system_prompt_path,
            raw_output_override=args.raw_output,
            filtered_output_override=args.output,
            ids_path=ids_path,
            overwrite=args.overwrite,
        )
        for paper_url in paper_urls
    ]

    payload: dict[str, Any]
    if len(results) == 1:
        payload = results[0]
    else:
        payload = {
            'papers_processed': len(results),
            'papers_requested': len(paper_urls),
            'ids_path': str(ids_path),
            'runs': results,
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))