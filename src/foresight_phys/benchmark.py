from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .cache import PredictionCache
from .formula_judging import FORMULA_JUDGE_MODEL, FormulaJudge
from .langfuse_logging import LangfuseRunLogger
from .metrics import compute_file_metrics
from .prediction import build_benchmark_items
from .reporting import write_human_readable_report
from .resources import resolve_system_prompt_path


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    langfuse_logger = LangfuseRunLogger.from_args(args)
    prediction_cache = PredictionCache.from_args(args)

    system_prompt_path = resolve_system_prompt_path(args.system_prompt)
    system_prompt = system_prompt_path.read_text(encoding='utf-8').strip()
    formula_judge = FormulaJudge(model=FORMULA_JUDGE_MODEL)
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
        metrics = compute_file_metrics(
            item.expected_output,
            item.actual_output,
            formula_judge=formula_judge,
        )
        langfuse_logger.log_file_result(item, metrics)
        rows.append(
            {
                'file': item.file_name,
                **metrics,
            }
        )

    mape_values = [row['mape'] for row in rows if row['mape'] is not None]
    accuracy_values = [
        row['bool_categorical_accuracy']
        for row in rows
        if row['bool_categorical_accuracy'] is not None
    ]
    formula_accuracy_values = [
        row['formula_accuracy']
        for row in rows
        if row['formula_accuracy'] is not None
    ]

    aggregate_mape = sum(mape_values) / len(mape_values) if mape_values else None
    aggregate_bool_categorical_accuracy = (
        sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
    )
    aggregate_formula_accuracy = (
        sum(formula_accuracy_values) / len(formula_accuracy_values)
        if formula_accuracy_values
        else None
    )
    total_excluded_numeric_values = sum(
        row['excluded_numeric_for_mape'] for row in rows
    )

    summary = {
        'run_name': args.run_name,
        'model': args.model,
        'max_workers': args.max_workers,
        'files_evaluated': len(rows),
        'aggregate_mape': aggregate_mape,
        'aggregate_bool_categorical_accuracy': aggregate_bool_categorical_accuracy,
        'aggregate_formula_accuracy': aggregate_formula_accuracy,
        'total_excluded_numeric_values_for_mape': total_excluded_numeric_values,
        'formula_judge_model': FORMULA_JUDGE_MODEL,
        'human_readable_report': args.html_output,
        'cache': prediction_cache.summary(),
        'langfuse': langfuse_logger.summary(),
        'per_file': rows,
    }

    if args.html_output:
        Path(args.html_output).parent.mkdir(parents=True, exist_ok=True)
        report_kwargs: dict[str, Any] = {
            'items': benchmark_items,
            'output_path': Path(args.html_output),
            'model': args.model,
            'aggregate_mape': aggregate_mape,
            'aggregate_bool_categorical_accuracy': aggregate_bool_categorical_accuracy,
            'aggregate_formula_accuracy': aggregate_formula_accuracy,
            'formula_judge': formula_judge,
        }
        write_human_readable_report(**report_kwargs)

    langfuse_logger.log_run_summary(summary, args)
    langfuse_logger.flush()
    prediction_cache.flush()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return summary