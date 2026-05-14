"""Post-hoc analysis of Foresight-Phys benchmark runs.

The benchmark CLI writes one run directory under ``docs/`` per invocation.
This subpackage parses those run directories, joins them against ground truth,
and produces summary statistics and plots.

Public entry points:
- :func:`parse_runs.parse_all_runs` — benchmark_results.json → predictions parquet
- :func:`build_dataset.build_scored_dataset` — predictions + GT → scored parquet
- :func:`analyze.write_summary` — scored parquet → per-field parquet + summary JSON
- :func:`plots.write_all_plots` — scored / per-field → one PDF per figure

Run them in that order, or use the ``foresight-phys-analysis`` CLI.
"""
from .analyze import write_summary
from .build_dataset import build_scored_dataset
from .parse_runs import parse_all_runs
from .plots import write_all_plots

__all__ = [
    "parse_all_runs",
    "build_scored_dataset",
    "write_summary",
    "write_all_plots",
]
