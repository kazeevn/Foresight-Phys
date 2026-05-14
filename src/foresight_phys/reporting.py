from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .formula_judging import FormulaJudge
from .metrics import (
    coerce_numeric,
    compute_file_metrics,
    compute_smape,
    is_formula_result,
    is_numeric_result,
    judge_formula_values,
    values_match,
)
from .models import BenchmarkItem


def write_human_readable_report(
    *,
    items: list[BenchmarkItem],
    output_path: Path,
    model: str,
    aggregate_prediction_quality: float | None,
    aggregate_smape: float | None,
    aggregate_normalized_smape_score: float | None,
    aggregate_bool_categorical_accuracy: float | None,
    aggregate_formula_accuracy: float | None,
    formula_judge: FormulaJudge | None = None,
) -> None:
    generated_at_utc = datetime.now(timezone.utc).isoformat()

    def format_metric(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    def display_file_title(file_name: str) -> str:
        if file_name.lower().endswith(".json"):
            return file_name[:-5]
        return file_name

    def format_value(value: Any) -> str:
        if value is None:
            return "MISSING"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def collect_file_section(item: BenchmarkItem) -> str:
        metrics = compute_file_metrics(
            item.expected_output,
            item.actual_output,
            formula_judge=formula_judge,
        )
        expected_experiments = item.expected_output if isinstance(item.expected_output, list) else [item.expected_output]
        actual_experiments = item.actual_output if isinstance(item.actual_output, list) else [item.actual_output]

        section_parts: list[str] = []
        section_parts.append('<div class="paper-metrics">')
        section_parts.append(
            f'<div class="metric-chip"><span class="metric-label">Prediction Quality</span><span class="metric-value">{html.escape(format_metric(metrics.get("prediction_quality")))}</span></div>'
        )
        section_parts.append(
            f'<div class="metric-chip"><span class="metric-label">Raw sMAPE</span><span class="metric-value">{html.escape(format_metric(metrics.get("smape")))}</span></div>'
        )
        section_parts.append(
            f'<div class="metric-chip"><span class="metric-label">Normalized sMAPE Score</span><span class="metric-value">{html.escape(format_metric(metrics.get("normalized_smape_score")))}</span></div>'
        )
        section_parts.append(
            f'<div class="metric-chip"><span class="metric-label">Bool/Categorical Accuracy</span><span class="metric-value">{html.escape(format_metric(metrics.get("bool_categorical_accuracy")))}</span></div>'
        )
        section_parts.append(
            f'<div class="metric-chip"><span class="metric-label">Formula Accuracy</span><span class="metric-value">{html.escape(format_metric(metrics.get("formula_accuracy")))}</span></div>'
        )
        section_parts.append('</div>')

        for experiment_index, expected_exp in enumerate(expected_experiments, start=1):
            actual_exp = actual_experiments[experiment_index - 1] if experiment_index - 1 < len(actual_experiments) else {}
            expected_desc = ""
            if isinstance(expected_exp, dict):
                expected_desc = str(expected_exp.get("experiment_description", ""))

            expected_results = expected_exp.get("experiment_results", {}) if isinstance(expected_exp, dict) else {}
            actual_results = actual_exp.get("experiment_results", {}) if isinstance(actual_exp, dict) else {}

            section_parts.append('<article class="experiment-card">')
            section_parts.append(f'<h3>Experiment {experiment_index}</h3>')
            section_parts.append(
                f'<p class="experiment-description">{html.escape(expected_desc)}</p>'
            )

            if not isinstance(expected_results, dict) or not expected_results:
                section_parts.append('<p class="empty-results">No result fields found.</p>')
                section_parts.append('</article>')
                continue

            section_parts.append('<div class="table-wrap">')
            section_parts.append(
                '<table><thead><tr><th>Result</th><th>Description</th><th>Ground Truth</th><th>Predicted</th><th>Status</th></tr></thead><tbody>'
            )

            for field_name, expected_meta in expected_results.items():
                expected_meta_dict = expected_meta if isinstance(expected_meta, dict) else {}
                field_description = str(expected_meta_dict.get("description", ""))
                field_type = str(expected_meta_dict.get("type", "")).strip().lower()
                expected_value = expected_meta_dict.get("result")

                actual_meta = actual_results.get(field_name, {}) if isinstance(actual_results, dict) else {}
                actual_meta_dict = actual_meta if isinstance(actual_meta, dict) else {}
                actual_value = actual_meta_dict.get("result")

                is_numeric_expected = is_numeric_result(field_type, expected_value)
                status_title = ""

                if is_numeric_expected:
                    expected_ok, expected_numeric = coerce_numeric(expected_value)
                    actual_ok, actual_numeric = coerce_numeric(actual_value)

                    if expected_ok and expected_numeric == 0.0:
                        zero_match = actual_ok and actual_numeric == 0.0
                        status_text = 'zero match' if zero_match else 'zero mismatch'
                        status_class = 'status-match' if zero_match else 'status-mismatch'
                    elif expected_ok:
                        if actual_ok:
                            smape = compute_smape(expected_numeric, actual_numeric)
                            status_text = f'sMAPE {smape:.4f}'
                            status_class = 'status-numeric'
                        else:
                            status_text = 'sMAPE n/a'
                            status_class = 'status-mismatch'
                    else:
                        match = values_match(expected_value, actual_value)
                        status_text = 'match' if match else 'mismatch'
                        status_class = 'status-match' if match else 'status-mismatch'
                elif is_formula_result(field_type):
                    formula_judgment = judge_formula_values(
                        expected_formula=expected_value,
                        actual_formula=actual_value,
                        experiment_description=expected_desc,
                        result_key=str(field_name),
                        result_description=field_description,
                        formula_judge=formula_judge,
                    )
                    status_text = 'formula match' if formula_judgment.equivalent else 'formula mismatch'
                    status_class = 'status-match' if formula_judgment.equivalent else 'status-mismatch'
                    status_title = formula_judgment.explanation
                else:
                    match = values_match(expected_value, actual_value)
                    status_text = 'match' if match else 'mismatch'
                    status_class = 'status-match' if match else 'status-mismatch'

                status_title_attr = ''
                if status_title:
                    status_title_attr = f' title="{html.escape(status_title, quote=True)}"'

                section_parts.append(
                    '<tr>'
                    f'<td>{html.escape(str(field_name))}</td>'
                    f'<td>{html.escape(field_description)}</td>'
                    f'<td>{html.escape(format_value(expected_value))}</td>'
                    f'<td>{html.escape(format_value(actual_value))}</td>'
                    f'<td><span class="{status_class}"{status_title_attr}>{status_text}</span></td>'
                    '</tr>'
                )

            section_parts.append('</tbody></table></div>')
            section_parts.append('</article>')

        return '\n'.join(section_parts)

    sections = [collect_file_section(item) for item in items]

    sidebar_buttons: list[str] = []
    paper_panels: list[str] = []
    for index, item in enumerate(items):
        active_class = ' is-active' if index == 0 else ''
        button_escaped = html.escape(display_file_title(item.file_name))
        sidebar_buttons.append(
            f'<button class="paper-tab{active_class}" data-paper-id="paper-{index}" type="button">{button_escaped}</button>'
        )
        paper_panels.append(
            '\n'.join(
                [
                    f'<section class="paper-panel{active_class}" id="paper-{index}">',
                    f'<h2>{button_escaped}</h2>',
                    sections[index],
                    '</section>',
                ]
            )
        )

    if not items:
        sidebar_html = '<div class="empty-sidebar">No files evaluated.</div>'
        panels_html = '<section class="paper-panel is-active" id="paper-empty"><p>No files evaluated.</p></section>'
    else:
        sidebar_html = '\n'.join(sidebar_buttons)
        panels_html = '\n'.join(paper_panels)

    report_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Foresight-Phys Human-Readable Report</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #ffffff;
            --surface: #ffffff;
            --text: #000000;
            --muted: #333333;
            --border: #cccccc;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 20px;
            font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
        }}
        .header {{ margin-bottom: 14px; }}
        .title {{ font-size: 24px; font-weight: 700; margin: 0 0 8px; }}
        .meta {{ color: var(--muted); font-size: 14px; line-height: 1.5; }}
        .layout {{
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 16px;
            margin-top: 16px;
            min-height: calc(100vh - 190px);
        }}
        .sidebar {{
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            background: var(--surface);
        }}
        .paper-tab {{
            width: 100%;
            text-align: left;
            border: 1px solid var(--border);
            background: #f7f7f7;
            color: var(--text);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            cursor: pointer;
            font-size: 13px;
            line-height: 1.35;
        }}
        .paper-tab:hover {{ background: #efefef; }}
        .paper-tab.is-active {{
            background: #e9f2ff;
            border-color: #8bb8ff;
            font-weight: 600;
        }}
        .content {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--surface);
            padding: 14px;
            overflow: auto;
        }}
        .paper-panel {{ display: none; }}
        .paper-panel.is-active {{ display: block; }}
        .paper-panel h2 {{ margin: 0 0 10px; font-size: 22px; }}
        .paper-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .metric-chip {{
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 13px;
            background: #fafafa;
        }}
        .metric-label {{ color: var(--muted); margin-right: 8px; }}
        .metric-value {{ font-weight: 700; }}
        .experiment-card {{
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            background: #fcfcfc;
        }}
        .experiment-card h3 {{ margin: 0 0 8px; font-size: 18px; }}
        .experiment-description {{ margin: 0 0 10px; color: #111; }}
        .empty-results {{ color: var(--muted); margin: 0; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 8px;
            vertical-align: top;
            text-align: left;
        }}
        th {{ background: #f3f3f3; }}
        .status-match {{ color: #0b7d2b; font-weight: 700; }}
        .status-mismatch {{ color: #b3261e; font-weight: 700; }}
        .status-numeric {{ color: #0b57d0; font-weight: 700; }}
        @media (max-width: 980px) {{
            .layout {{ grid-template-columns: 1fr; }}
            .sidebar {{ order: 1; }}
            .content {{ order: 2; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">Offline Prediction vs Reference Report</h1>
        <div class="meta">Model: {html.escape(model)}</div>
        <div class="meta">Generated: {html.escape(generated_at_utc)}</div>
        <div class="meta">Files: {len(items)}</div>
        <div class="meta">Aggregate prediction quality: {format_metric(aggregate_prediction_quality)}</div>
        <div class="meta">Aggregate raw sMAPE: {format_metric(aggregate_smape)}</div>
        <div class="meta">Aggregate normalized sMAPE score: {format_metric(aggregate_normalized_smape_score)}</div>
        <div class="meta">Aggregate bool/categorical accuracy: {format_metric(aggregate_bool_categorical_accuracy)}</div>
        <div class="meta">Aggregate formula accuracy: {format_metric(aggregate_formula_accuracy)}</div>
    </div>
    <div class="layout">
        <aside class="sidebar" aria-label="Papers">
            {sidebar_html}
        </aside>
        <main class="content">
            {panels_html}
        </main>
    </div>
    <script>
        const tabs = Array.from(document.querySelectorAll('.paper-tab'));
        const panels = Array.from(document.querySelectorAll('.paper-panel'));
        function activatePanel(panelId) {{
            tabs.forEach((tab) => tab.classList.toggle('is-active', tab.dataset.paperId === panelId));
            panels.forEach((panel) => panel.classList.toggle('is-active', panel.id === panelId));
        }}
        tabs.forEach((tab) => {{
            tab.addEventListener('click', () => activatePanel(tab.dataset.paperId));
        }});
    </script>
</body>
</html>
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding='utf-8')