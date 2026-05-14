"""Parse benchmark HTML reports into a tidy predictions parquet.

Each run directory under ``docs/<run-name>/`` is expected to contain a
``benchmark_human_readable_report.html`` and ``benchmark_results.json`` from
which the model name is read.

We use the HTML report (not the JSON summary) because per-field predictions
are not included in the summary JSON — only per-paper metrics.
"""
from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

import pandas as pd

from .paths import AnalysisPaths, default_paths

PANEL_RE = re.compile(
    r'<section class="paper-panel[^"]*" id="paper-\d+">(.*?)</section>', re.S
)
TITLE_RE = re.compile(r"<h2>(.*?)</h2>", re.S)
FILE_RE = re.compile(r'<div class="paper-file-name">(.*?)</div>', re.S)
EXP_RE = re.compile(
    r"<h3>Experiment\s*(\d+)</h3>(.*?)<table>.*?<tbody>(.*?)</tbody>\s*</table>",
    re.S,
)
EXP_DESC_RE = re.compile(r'<p class="experiment-description">(.*?)</p>', re.S)
ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL_RE = re.compile(r"<td.*?>(.*?)</td>", re.S)
STATUS_SPAN_RE = re.compile(r'<span class="(status-[^"]+)"[^>]*>(.*?)</span>', re.S)


def _strip_tags(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _parse_status(html_cell: str) -> tuple[str, str]:
    m = STATUS_SPAN_RE.search(html_cell)
    if m:
        return m.group(1), unescape(_strip_tags(m.group(2)))
    return "status-unknown", _strip_tags(html_cell)


def parse_report(model: str, run_name: str, html_path: Path) -> list[dict]:
    text = html_path.read_text()
    rows: list[dict] = []
    for panel in PANEL_RE.findall(text):
        title_m = TITLE_RE.search(panel)
        file_m = FILE_RE.search(panel)
        if not title_m or not file_m:
            continue
        paper_title = _strip_tags(title_m.group(1))
        file_id = _strip_tags(file_m.group(1))
        for exp_m in EXP_RE.finditer(panel):
            exp_idx = int(exp_m.group(1)) - 1
            desc_m = EXP_DESC_RE.search(exp_m.group(2))
            exp_desc = _strip_tags(desc_m.group(1)) if desc_m else ""
            for tr in ROW_RE.findall(exp_m.group(3)):
                cells = CELL_RE.findall(tr)
                if len(cells) != 5:
                    continue
                key = _strip_tags(cells[0])
                desc = _strip_tags(cells[1])
                gt = _strip_tags(cells[2])
                pred = _strip_tags(cells[3])
                status_cls, status_text = _parse_status(cells[4])
                rows.append({
                    "run_name": run_name,
                    "model": model,
                    "file": file_id,
                    "paper_title": paper_title,
                    "experiment": exp_idx,
                    "experiment_description": exp_desc,
                    "key": key,
                    "description": desc,
                    "gt": gt,
                    "pred": pred,
                    "status_class": status_cls,
                    "status_text": status_text,
                })
    return rows


def discover_run_dirs(docs_dir: Path) -> list[Path]:
    """Find run directories: docs/<name>/ that contain both a results JSON and HTML."""
    run_dirs: list[Path] = []
    for child in sorted(docs_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "benchmark_results.json").exists() and (
            child / "benchmark_human_readable_report.html"
        ).exists():
            run_dirs.append(child)
    return run_dirs


def parse_all_runs(paths: AnalysisPaths | None = None) -> pd.DataFrame:
    """Parse every run directory found under ``docs/`` into a single dataframe.

    The dataframe is also written to :pyattr:`AnalysisPaths.predictions_parquet`.
    """
    paths = paths or default_paths()
    paths.ensure_dirs()

    run_dirs = discover_run_dirs(paths.docs_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No benchmark run directories found under {paths.docs_dir}. "
            "Expected docs/<run-name>/benchmark_human_readable_report.html."
        )

    all_rows: list[dict] = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "benchmark_results.json").read_text())
        model = summary.get("model") or run_dir.name
        all_rows.extend(parse_report(
            model=model,
            run_name=run_dir.name,
            html_path=run_dir / "benchmark_human_readable_report.html",
        ))

    df = pd.DataFrame(all_rows)
    df.to_parquet(paths.predictions_parquet, index=False)
    return df
