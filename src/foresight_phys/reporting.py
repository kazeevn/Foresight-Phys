from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def format_metric_value(value: float | None) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.4f}'


def format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return 'n/a'
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _numeric_metric_label_and_key(
    values: dict[str, Any],
    *,
    aggregate: bool,
) -> tuple[str, str]:
    prefix = 'aggregate_' if aggregate else ''
    crps_key = f'{prefix}numeric_crps'
    if crps_key in values:
        return 'Numeric CRPS', crps_key

    nll_key = f'{prefix}numeric_nll'
    if nll_key in values:
        return 'Numeric NLL', nll_key

    return 'Numeric CRPS', crps_key


def write_runs_index(*, docs_dir: Path, latest_run_name: str | None = None) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for run_dir in docs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / 'benchmark_results.json'
        report_path = run_dir / 'benchmark_human_readable_report.html'
        if not summary_path.exists() and not report_path.exists():
            continue

        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                loaded_summary = json.loads(summary_path.read_text(encoding='utf-8'))
                if isinstance(loaded_summary, dict):
                    summary = loaded_summary
            except json.JSONDecodeError:
                summary = {}

        timestamps = []
        if summary_path.exists():
            timestamps.append(summary_path.stat().st_mtime)
        if report_path.exists():
            timestamps.append(report_path.stat().st_mtime)
        if not timestamps:
            timestamps.append(run_dir.stat().st_mtime)

        numeric_metric_label, numeric_metric_key = _numeric_metric_label_and_key(
            summary,
            aggregate=True,
        )

        runs.append(
            {
                'run_name': run_dir.name,
                'model': summary.get('model'),
                'files_evaluated': summary.get('files_evaluated'),
                'aggregate_prediction_quality': summary.get('aggregate_prediction_quality'),
                'aggregate_numeric_metric_label': numeric_metric_label,
                'aggregate_numeric_metric': summary.get(numeric_metric_key),
                'aggregate_coverage_1sigma': summary.get('aggregate_coverage_1sigma'),
                'aggregate_coverage_2sigma': summary.get('aggregate_coverage_2sigma'),
                'summary_href': f'{run_dir.name}/benchmark_results.json' if summary_path.exists() else None,
                'report_href': f'{run_dir.name}/benchmark_human_readable_report.html' if report_path.exists() else None,
                'updated_at': max(timestamps),
            }
        )

    runs.sort(
        key=lambda run: (run['updated_at'], run['run_name']),
        reverse=True,
    )

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    cards: list[str] = []
    for run in runs:
        badge_html = ''
        if latest_run_name and run['run_name'] == latest_run_name:
            badge_html = '<span class="badge">Current run</span>'

        report_link = '<span class="link-disabled">Report unavailable</span>'
        if run['report_href']:
            report_link = (
                f'<a href="{html.escape(str(run["report_href"]))}">Open report</a>'
            )

        summary_link = '<span class="link-disabled">Summary unavailable</span>'
        if run['summary_href']:
            summary_link = (
                f'<a href="{html.escape(str(run["summary_href"]))}">Open JSON</a>'
            )

        files_evaluated = run['files_evaluated']
        files_text = 'n/a' if files_evaluated is None else str(files_evaluated)

        cards.append(
            '\n'.join(
                [
                    '<article class="run-card">',
                    '<div class="run-card-header">',
                    f'<h2>{html.escape(str(run["run_name"]))}</h2>',
                    badge_html,
                    '</div>',
                    '<dl class="metrics-grid">',
                    '<div><dt>Model</dt>'
                    f'<dd>{html.escape(str(run["model"] or "n/a"))}</dd></div>',
                    '<div><dt>Files</dt>'
                    f'<dd>{html.escape(files_text)}</dd></div>',
                    '<div><dt>Prediction Quality</dt>'
                    f'<dd>{html.escape(format_metric_value(run["aggregate_prediction_quality"]))}</dd></div>',
                    f'<div><dt>{html.escape(run["aggregate_numeric_metric_label"])}</dt>'
                    f'<dd>{html.escape(format_metric_value(run["aggregate_numeric_metric"]))}</dd></div>',
                    '<div><dt>Coverage @1σ</dt>'
                    f'<dd>{html.escape(format_metric_value(run["aggregate_coverage_1sigma"]))}</dd></div>',
                    '<div><dt>Coverage @2σ</dt>'
                    f'<dd>{html.escape(format_metric_value(run["aggregate_coverage_2sigma"]))}</dd></div>',
                    '<div><dt>Updated</dt>'
                    f'<dd>{html.escape(format_timestamp(run["updated_at"]))}</dd></div>',
                    '</dl>',
                    '<div class="run-links">',
                    report_link,
                    summary_link,
                    '</div>',
                    '</article>',
                ]
            )
        )

    if not cards:
        cards_html = '<section class="empty-state"><p>No benchmark runs available yet.</p></section>'
    else:
        cards_html = '\n'.join(cards)

    index_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Foresight-Phys Runs</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f1e8;
            --surface: rgba(255, 252, 245, 0.96);
            --surface-strong: #fffdf8;
            --text: #1f1d1a;
            --muted: #5f5647;
            --border: #d8cfbf;
            --accent: #0f766e;
            --accent-soft: #d8f0ec;
            --shadow: rgba(77, 63, 36, 0.12);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
                linear-gradient(180deg, #f7f4ec 0%, var(--bg) 100%);
        }}
        main {{
            width: min(1100px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 40px 0 64px;
        }}
        .hero {{
            padding: 28px;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: var(--surface);
            box-shadow: 0 18px 48px var(--shadow);
            backdrop-filter: blur(12px);
        }}
        .eyebrow {{
            margin: 0 0 10px;
            color: var(--accent);
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        h1 {{
            margin: 0;
            font-size: clamp(32px, 4vw, 56px);
            line-height: 0.98;
        }}
        .hero p {{
            margin: 14px 0 0;
            max-width: 720px;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.6;
        }}
        .hero-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 18px;
        }}
        .hero-chip {{
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--surface-strong);
            font-size: 13px;
            color: var(--muted);
        }}
        .runs-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 18px;
            margin-top: 24px;
        }}
        .run-card, .empty-state {{
            border: 1px solid var(--border);
            border-radius: 22px;
            background: var(--surface);
            padding: 22px;
            box-shadow: 0 18px 36px var(--shadow);
        }}
        .run-card-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
        }}
        .run-card h2 {{
            margin: 0;
            font-size: 21px;
            line-height: 1.2;
            word-break: break-word;
        }}
        .badge {{
            flex-shrink: 0;
            padding: 7px 10px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px 12px;
            margin: 18px 0 0;
        }}
        dt {{
            margin: 0;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--muted);
        }}
        dd {{
            margin: 6px 0 0;
            font-size: 16px;
            font-weight: 600;
        }}
        .run-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 18px;
        }}
        a {{
            color: var(--accent);
            font-weight: 700;
            text-decoration: none;
        }}
        a:hover {{ text-decoration: underline; }}
        .link-disabled {{
            color: var(--muted);
            font-size: 14px;
        }}
        .empty-state p {{
            margin: 0;
            color: var(--muted);
            font-size: 15px;
        }}
        @media (max-width: 640px) {{
            main {{ width: min(100vw - 20px, 1100px); padding: 20px 0 40px; }}
            .hero {{ padding: 20px; border-radius: 18px; }}
            .run-card {{ padding: 18px; border-radius: 18px; }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main>
        <section class="hero">
            <p class="eyebrow">Foresight-Phys</p>
            <h1>Benchmark Run Index</h1>
            <p>Available benchmark runs discovered under the local docs directory. Each card links to the machine-readable summary and the offline human-readable report when present.</p>
            <div class="hero-meta">
                <div class="hero-chip">Generated: {html.escape(generated_at_utc)}</div>
                <div class="hero-chip">Runs indexed: {len(runs)}</div>
            </div>
        </section>
        <section class="runs-grid">
            {cards_html}
        </section>
    </main>
</body>
</html>
'''

    (docs_dir / 'index.html').write_text(index_html, encoding='utf-8')


def write_human_readable_report(
    *,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    generated_at_utc = str(
        summary.get('generated_at_utc') or datetime.now(timezone.utc).isoformat()
    )
    model = str(summary.get('model', 'n/a'))
    items = summary.get('per_file', [])
    if not isinstance(items, list):
        items = []

    def display_file_title(item_summary: dict[str, Any]) -> str:
        paper_title = item_summary.get('paper_title')
        if isinstance(paper_title, str) and paper_title.strip():
            return paper_title.strip()
        file_name = str(item_summary.get('file', ''))
        if file_name.lower().endswith('.json'):
            return file_name[:-5]
        return file_name

    def display_file_name(file_name: str) -> str:
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

    def format_uncertainty(row: dict[str, Any]) -> str:
        field_type = str(row.get('type', '')).strip().lower()
        if field_type in {'float', 'integer', 'int', 'number'}:
            distribution = row.get('distribution')
            p10 = row.get('p10')
            p50 = row.get('p50')
            p90 = row.get('p90')
            sigma = row.get('sigma')
            if distribution is None or p10 is None or p50 is None or p90 is None:
                return 'n/a'
            unit = 'dex' if distribution == 'log_normal' else 'linear'
            sigma_part = f', σ ≈ {sigma:.3g} {unit}' if sigma is not None else ''
            return (
                f'p10/p50/p90 = {p10:.3g} / {p50:.3g} / {p90:.3g} '
                f'({distribution}{sigma_part})'
            )
        if field_type in {'bool', 'boolean'}:
            prob_true = row.get('prob_true')
            if prob_true is None:
                return 'n/a'
            return f'P(true) = {prob_true * 100.0:.1f}%'
        if field_type == 'categorical':
            probabilities = row.get('probabilities') or {}
            if not probabilities:
                return 'n/a'
            top_items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
            return ', '.join(f'{k}={v * 100.0:.0f}%' for k, v in top_items)
        if field_type == 'formula':
            confidence = row.get('confidence')
            if confidence is None:
                return 'n/a'
            return f'confidence = {confidence:.2f}'
        return 'n/a'

    def collect_file_section(item_summary: dict[str, Any]) -> str:
        section_parts: list[str] = []
        section_parts.append('<div class="paper-metrics">')
        numeric_metric_label, numeric_metric_key = _numeric_metric_label_and_key(
            item_summary,
            aggregate=False,
        )
        chips = (
            ("Prediction Quality", "prediction_quality"),
            (numeric_metric_label, numeric_metric_key),
            ("Coverage @1σ", "coverage_1sigma"),
            ("Coverage @2σ", "coverage_2sigma"),
            ("Bool Brier", "bool_brier"),
            ("Categorical Brier", "categorical_brier"),
            ("Bool/Categorical Accuracy", "bool_categorical_accuracy"),
            ("Formula Accuracy", "formula_accuracy"),
        )
        for label, key in chips:
            section_parts.append(
                '<div class="metric-chip">'
                f'<span class="metric-label">{html.escape(label)}</span>'
                f'<span class="metric-value">{html.escape(format_metric_value(item_summary.get(key)))}</span>'
                '</div>'
            )
        section_parts.append('</div>')

        report_experiments = item_summary.get('report_experiments', [])
        if not isinstance(report_experiments, list):
            report_experiments = []

        for experiment_index, experiment_summary in enumerate(report_experiments, start=1):
            if not isinstance(experiment_summary, dict):
                continue
            expected_desc = str(experiment_summary.get('experiment_description', ''))
            result_rows = experiment_summary.get('result_rows', [])
            if not isinstance(result_rows, list):
                result_rows = []
            section_parts.append('<article class="experiment-card">')
            section_parts.append(f'<h3>Experiment {experiment_index}</h3>')
            section_parts.append(
                f'<p class="experiment-description">{html.escape(expected_desc)}</p>'
            )

            if not result_rows:
                section_parts.append('<p class="empty-results">No result fields found.</p>')
                section_parts.append('</article>')
                continue

            section_parts.append('<div class="table-wrap">')
            section_parts.append(
                '<table><thead><tr><th>Result</th><th>Description</th><th>Ground Truth</th><th>Predicted</th><th>Uncertainty</th><th>Status</th></tr></thead><tbody>'
            )

            for row in result_rows:
                if not isinstance(row, dict):
                    continue
                field_name = str(row.get('result_key', ''))
                field_description = str(row.get('description', ''))
                expected_value = row.get('ground_truth')
                actual_value = row.get('predicted')
                status_text = str(row.get('status_text', ''))
                status_class = str(row.get('status_class', ''))
                status_title = row.get('status_title')
                status_title_attr = ''
                if isinstance(status_title, str) and status_title:
                    status_title_attr = f' title="{html.escape(status_title, quote=True)}"'
                uncertainty_cell = format_uncertainty(row)

                section_parts.append(
                    '<tr>'
                    f'<td>{html.escape(str(field_name))}</td>'
                    f'<td>{html.escape(field_description)}</td>'
                    f'<td>{html.escape(format_value(expected_value))}</td>'
                    f'<td>{html.escape(format_value(actual_value))}</td>'
                    f'<td>{html.escape(uncertainty_cell)}</td>'
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
        if not isinstance(item, dict):
            continue
        active_class = ' is-active' if index == 0 else ''
        button_escaped = html.escape(display_file_title(item))
        file_name_escaped = html.escape(display_file_name(str(item.get('file', ''))))
        sidebar_buttons.append(
            f'<button class="paper-tab{active_class}" data-paper-id="paper-{index}" type="button">{button_escaped}</button>'
        )
        paper_panels.append(
            '\n'.join(
                [
                    f'<section class="paper-panel{active_class}" id="paper-{index}">',
                    f'<h2>{button_escaped}</h2>',
                    f'<div class="paper-file-name">{file_name_escaped}</div>',
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

    aggregate_numeric_label, aggregate_numeric_key = _numeric_metric_label_and_key(
        summary,
        aggregate=True,
    )

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
        .paper-file-name {{ margin: -4px 0 12px; color: var(--muted); font-size: 13px; }}
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
        <div class="meta">Aggregate prediction quality: {format_metric_value(summary.get('aggregate_prediction_quality'))}</div>
        <div class="meta">Aggregate {html.escape(aggregate_numeric_label)}: {format_metric_value(summary.get(aggregate_numeric_key))}</div>
        <div class="meta">Aggregate coverage @1σ: {format_metric_value(summary.get('aggregate_coverage_1sigma'))}</div>
        <div class="meta">Aggregate coverage @2σ: {format_metric_value(summary.get('aggregate_coverage_2sigma'))}</div>
        <div class="meta">Aggregate bool Brier: {format_metric_value(summary.get('aggregate_bool_brier'))}</div>
        <div class="meta">Aggregate categorical Brier: {format_metric_value(summary.get('aggregate_categorical_brier'))}</div>
        <div class="meta">Aggregate bool/categorical accuracy: {format_metric_value(summary.get('aggregate_bool_categorical_accuracy'))}</div>
        <div class="meta">Aggregate formula accuracy: {format_metric_value(summary.get('aggregate_formula_accuracy'))}</div>
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