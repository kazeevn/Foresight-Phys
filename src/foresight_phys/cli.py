from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import secrets

from .benchmark import run_benchmark
from .resources import DEFAULT_SYSTEM_PROMPT_PATH


RUN_NAME_ADJECTIVES = (
    'amber',
    'arc',
    'brisk',
    'clear',
    'cosmic',
    'crisp',
    'delta',
    'eager',
    'ember',
    'gentle',
    'keen',
    'lattice',
    'lucid',
    'nova',
    'prism',
    'quiet',
    'radial',
    'rapid',
    'silver',
    'solar',
    'steady',
    'tidal',
    'vivid',
    'zenith',
)
RUN_NAME_NOUNS = (
    'beam',
    'boson',
    'circuit',
    'field',
    'flux',
    'frame',
    'lens',
    'mode',
    'orbit',
    'phase',
    'photon',
    'pulse',
    'quanta',
    'signal',
    'spark',
    'spectrum',
    'spin',
    'stream',
    'tensor',
    'vector',
    'vertex',
    'wave',
)


def build_run_name(model: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    nice_suffix = '-'.join(
        (
            secrets.choice(RUN_NAME_ADJECTIVES),
            secrets.choice(RUN_NAME_NOUNS),
            timestamp,
        )
    )
    return f'{model}-{nice_suffix}'


def apply_run_defaults(args: argparse.Namespace) -> argparse.Namespace:
    run_name = args.langfuse_run_name or build_run_name(args.model)
    run_dir = Path('docs') / run_name

    args.langfuse_run_name = run_name
    args.run_name = run_name

    if args.output is None:
        args.output = str(run_dir / 'benchmark_results.json')

    if args.html_output is None:
        args.html_output = str(run_dir / 'benchmark_human_readable_report.html')
    elif args.html_output == '':
        args.html_output = None

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run LLM benchmark over experiment JSON files.'
    )
    parser.add_argument('--json-dir', type=Path,
        default='JSONs/filtered',
        help='Directory with JSON benchmark files.')
    parser.add_argument(
        '--system-prompt',
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help='Path to system prompt text file.',
    )
    parser.add_argument('--model', default='gpt-5.4-nano', help='LLM model name.')
    parser.add_argument(
        '--service-tier',
        default='flex',
        help='OpenAI Responses API service tier passed to benchmark prediction calls.',
    )
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
        '--run-name', '--langfuse-run-name',
        dest='langfuse_run_name',
        default=None,
        help='Optional run name for Langfuse and default docs/<run-name>/ outputs.',
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Optional cap on number of JSON files for quick runs.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Path for summary JSON output. Defaults to docs/<run-name>/benchmark_results.json.',
    )
    parser.add_argument(
        '--html-output',
        default=None,
        help='Path for static offline HTML report. Defaults to docs/<run-name>/benchmark_human_readable_report.html; set empty string to disable.',
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
    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='Never call OpenAI; require every prediction to already exist in the local cache.',
    )
    parser.add_argument(
        '--cache-ignore-system-prompt',
        action='store_true',
        help='Ignore the system prompt when computing prediction cache keys.',
    )
    args = parser.parse_args()
    if args.disable_cache and args.cache_only:
        parser.error('--cache-only cannot be used with --disable-cache.')
    return apply_run_defaults(args)


def main() -> None:
    args = parse_args()
    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    def _print(label: str, key: str) -> None:
        value = summary.get(key)
        if value is None:
            return
        print(f"{label}: {value:.4f}")

    _print("Aggregate prediction quality", "aggregate_prediction_quality")
    _print("Aggregate numeric quality", "aggregate_numeric_quality")
    _print("Aggregate numeric CRPS", "aggregate_numeric_crps")
    _print("Aggregate numeric rel. CRPS", "aggregate_numeric_crps_scaled")
    _print("Aggregate coverage @1σ", "aggregate_coverage_1sigma")
    _print("Aggregate coverage @2σ", "aggregate_coverage_2sigma")
    _print("Aggregate bool Brier", "aggregate_bool_brier")
    _print("Aggregate categorical Brier", "aggregate_categorical_brier")
    _print("Aggregate bool/categorical accuracy", "aggregate_bool_categorical_accuracy")
    if summary.get('aggregate_formula_accuracy') is not None:
        print(
            f"Aggregate formula accuracy ({summary['formula_judge_model']} judge): "
            f"{summary['aggregate_formula_accuracy']:.4f}"
        )
    _print("Aggregate formula Brier", "aggregate_formula_brier")


if __name__ == '__main__':
    main()
