"""Command-line entry point for the analysis pipeline.

Usage::

    foresight-phys-analysis all          # parse, build, analyze, plots
    foresight-phys-analysis parse        # benchmark_results.json → predictions.parquet
    foresight-phys-analysis build        # predictions + GT → scored.parquet
    foresight-phys-analysis analyze      # scored → per_field.parquet + summary.json
    foresight-phys-analysis plots        # scored + per_field → PDFs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import write_summary
from .build_dataset import build_scored_dataset
from .parse_runs import parse_all_runs
from .paths import AnalysisPaths
from .plots import write_all_plots

STEPS = ("parse", "build", "analyze", "plots")


def _resolve_paths(args: argparse.Namespace) -> AnalysisPaths:
    return AnalysisPaths(project_root=Path(args.project_root).resolve())


def _run_step(step: str, paths: AnalysisPaths) -> None:
    if step == "parse":
        df = parse_all_runs(paths)
        print(f"parsed {len(df)} rows → {paths.predictions_parquet}")
    elif step == "build":
        df = build_scored_dataset(paths)
        print(f"scored {len(df)} rows → {paths.scored_parquet}")
    elif step == "analyze":
        summary = write_summary(paths)
        print(f"summary ({summary['n_fields']} fields × {summary['n_models']} models) → {paths.summary_json}")
        print(f"per-field table → {paths.per_field_parquet}")
    elif step == "plots":
        written = write_all_plots(paths)
        print(f"wrote {len(written)} PDFs to {paths.plots_dir}")
        for p in written:
            print(f"  - {p.name}")
    else:
        raise ValueError(f"unknown step: {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("all", *STEPS),
                        help="which pipeline step to run; 'all' runs everything in order")
    parser.add_argument("--project-root", default=str(Path.cwd()),
                        help="repository root (default: current working directory)")
    args = parser.parse_args()

    paths = _resolve_paths(args)
    steps = STEPS if args.step == "all" else (args.step,)
    for step in steps:
        _run_step(step, paths)


if __name__ == "__main__":
    main()
