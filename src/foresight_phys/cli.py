from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark
from .constants import MAPE_MIN_ABS_TARGET
from .resources import DEFAULT_SYSTEM_PROMPT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run LLM benchmark over experiment JSON files.'
    )
    parser.add_argument('--json-dir', default='JSONs', help='Directory with JSON benchmark files.')
    parser.add_argument(
        '--system-prompt',
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help='Path to system prompt text file.',
    )
    parser.add_argument('--model', default='gpt-5-nano', help='LLM model name.')
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum parallel OpenAI requests.',
    )
    parser.add_argument(
        '--disable-langfuse',
        action='store_true',
        help='Disable Langfuse tracing output.',
    )
    parser.add_argument(
        '--langfuse-host',
        default=None,
        help='Optional Langfuse host URL override (otherwise LANGFUSE_HOST is used).',
    )
    parser.add_argument(
        '--langfuse-run-name',
        default=None,
        help='Optional run name for grouping traces in Langfuse dashboard.',
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Optional cap on number of JSON files for quick runs.',
    )
    parser.add_argument(
        '--output',
        default='docs/benchmark_results.json',
        help='Path for summary JSON output.',
    )
    parser.add_argument(
        '--html-output',
        default='docs/benchmark_human_readable_report.html',
        help='Path for static offline HTML human-readable report (set empty string to disable).',
    )
    parser.add_argument(
        '--cache-path',
        default='.cache/llm_predictions.json',
        help='Path to persistent cache for raw LLM predictions.',
    )
    parser.add_argument(
        '--disable-cache',
        action='store_true',
        help='Disable local prediction cache and always call the LLM.',
    )
    args = parser.parse_args()
    if args.html_output == '':
        args.html_output = None
    return args


def main() -> None:
    args = parse_args()
    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f'Total excluded numeric values for MAPE (abs(expected) < {MAPE_MIN_ABS_TARGET}): '
        f"{summary['total_excluded_numeric_values_for_mape']}"
    )


if __name__ == '__main__':
    main()