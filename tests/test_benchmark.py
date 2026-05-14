from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foresight_phys.benchmark import run_benchmark
from foresight_phys.models import BenchmarkItem
from foresight_phys.reporting import write_human_readable_report, write_runs_index


class DummyLogger:
    def log_file_result(self, _item: BenchmarkItem, _metrics: dict[str, object]) -> None:
        return None

    def log_run_summary(self, _summary: dict[str, object], _args: argparse.Namespace) -> None:
        return None

    def flush(self) -> None:
        return None

    def summary(self) -> dict[str, object]:
        return {"enabled": False}


class DummyCache:
    def flush(self) -> None:
        return None

    def summary(self) -> dict[str, object]:
        return {"enabled": False}


class DummyFormulaJudge:
    def __init__(self, *, model: str) -> None:
        self.model = model


class RunBenchmarkTests(unittest.TestCase):
    def test_aggregates_mean_of_paper_scores(self) -> None:
        paper_one = BenchmarkItem(
            file_name="paper-one.json",
            masked_input={},
            expected_output={
                "experiment_description": "Paper one",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                    "phase": {
                        "type": "categorical",
                        "description": "Observed phase",
                        "result": "solid",
                    },
                    "dispersion": {
                        "type": "formula",
                        "description": "Dispersion relation",
                        "result": "E = mc^2",
                    },
                },
            },
            actual_output={
                "experiment_description": "Paper one",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                    "phase": {
                        "type": "categorical",
                        "description": "Observed phase",
                        "result": "solid",
                    },
                    "dispersion": {
                        "type": "formula",
                        "description": "Dispersion relation",
                        "result": "E=mc^2",
                    },
                },
            },
        )
        paper_two = BenchmarkItem(
            file_name="paper-two.json",
            masked_input={},
            expected_output={
                "experiment_description": "Paper two",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                    "stable": {
                        "type": "bool",
                        "description": "System remains stable",
                        "result": False,
                    },
                },
            },
            actual_output={
                "experiment_description": "Paper two",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 0.0,
                    },
                    "stable": {
                        "type": "bool",
                        "description": "System remains stable",
                        "result": True,
                    },
                },
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "benchmark_results.json"
            args = argparse.Namespace(
                run_name="unit-test-run",
                model="gpt-5.4-nano",
                max_workers=2,
                json_dir=Path("JSONs/filtered"),
                system_prompt=Path(__file__).resolve().parents[1] / "src/foresight_phys/system_prompt.txt",
                max_files=None,
                output=str(output_path),
                html_output=None,
            )

            with patch("foresight_phys.benchmark.load_dotenv"), patch(
                "foresight_phys.benchmark.LangfuseRunLogger.from_args",
                return_value=DummyLogger(),
            ), patch(
                "foresight_phys.benchmark.PredictionCache.from_args",
                return_value=DummyCache(),
            ), patch(
                "foresight_phys.benchmark.FormulaJudge",
                DummyFormulaJudge,
            ), patch(
                "foresight_phys.benchmark.build_benchmark_items",
                return_value=[paper_one, paper_two],
            ):
                summary = run_benchmark(args)
                written_summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["files_evaluated"], 2)
        self.assertAlmostEqual(summary["aggregate_prediction_quality"], 0.5)
        self.assertEqual(summary["aggregate_log_accuracy"], float("inf"))
        self.assertAlmostEqual(summary["aggregate_normalized_log_accuracy_score"], 0.5)
        self.assertAlmostEqual(summary["aggregate_bool_categorical_accuracy"], 0.5)
        self.assertAlmostEqual(summary["aggregate_formula_accuracy"], 1.0)

        self.assertEqual(
            written_summary["aggregate_prediction_quality"],
            summary["aggregate_prediction_quality"],
        )
        self.assertEqual(
            written_summary["aggregate_log_accuracy"], summary["aggregate_log_accuracy"]
        )

    def test_run_benchmark_updates_docs_index(self) -> None:
        paper = BenchmarkItem(
            file_name="paper-one.json",
            masked_input={},
            expected_output={
                "experiment_description": "Paper one",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                },
            },
            actual_output={
                "experiment_description": "Paper one",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                },
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "benchmark_results.json"
            args = argparse.Namespace(
                run_name="unit-test-run",
                model="gpt-5.4-nano",
                max_workers=2,
                json_dir=Path("JSONs/filtered"),
                system_prompt=Path(__file__).resolve().parents[1] / "src/foresight_phys/system_prompt.txt",
                max_files=None,
                output=str(output_path),
                html_output=None,
            )

            with patch("foresight_phys.benchmark.load_dotenv"), patch(
                "foresight_phys.benchmark.LangfuseRunLogger.from_args",
                return_value=DummyLogger(),
            ), patch(
                "foresight_phys.benchmark.PredictionCache.from_args",
                return_value=DummyCache(),
            ), patch(
                "foresight_phys.benchmark.FormulaJudge",
                DummyFormulaJudge,
            ), patch(
                "foresight_phys.benchmark.build_benchmark_items",
                return_value=[paper],
            ), patch("foresight_phys.benchmark.write_runs_index") as write_runs_index_mock:
                run_benchmark(args)

        write_runs_index_mock.assert_called_once_with(
            docs_dir=Path("docs"),
            latest_run_name="unit-test-run",
        )


class ReportingTests(unittest.TestCase):
    def test_write_human_readable_report_displays_paper_title(self) -> None:
        item = BenchmarkItem(
            file_name="paper-one.json",
            masked_input={},
            expected_output={
                "experiment_description": "Paper one",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                },
            },
            actual_output={
                "experiment_description": "Paper one",
                "experiment_results": {
                    "temperature": {
                        "type": "float",
                        "description": "Measured temperature",
                        "result": 10.0,
                    },
                },
            },
            paper_title="Visible Paper Title",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "benchmark_human_readable_report.html"
            write_human_readable_report(
                items=[item],
                output_path=report_path,
                model="gpt-5.4-nano",
                aggregate_prediction_quality=1.0,
                aggregate_log_accuracy=0.0,
                aggregate_normalized_log_accuracy_score=1.0,
                aggregate_bool_categorical_accuracy=None,
                aggregate_formula_accuracy=None,
            )

            report_html = report_path.read_text(encoding="utf-8")

        self.assertIn("Visible Paper Title", report_html)
        self.assertIn(">paper-one<", report_html)

    def test_write_runs_index_lists_available_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            docs_dir = Path(tmp_dir) / "docs"
            older_run_dir = docs_dir / "older-run"
            older_run_dir.mkdir(parents=True)
            (older_run_dir / "benchmark_results.json").write_text(
                json.dumps(
                    {
                        "run_name": "older-run",
                        "model": "gpt-5.4-nano",
                        "files_evaluated": 3,
                        "aggregate_prediction_quality": 0.625,
                        "aggregate_log_accuracy": 0.75,
                        "aggregate_normalized_log_accuracy_score": 0.25,
                    }
                ),
                encoding="utf-8",
            )
            (older_run_dir / "benchmark_human_readable_report.html").write_text(
                "<html></html>",
                encoding="utf-8",
            )

            current_run_dir = docs_dir / "unit-test-run"
            current_run_dir.mkdir(parents=True)
            (current_run_dir / "benchmark_results.json").write_text(
                json.dumps(
                    {
                        "run_name": "unit-test-run",
                        "model": "gpt-5.4-mini",
                        "files_evaluated": 5,
                        "aggregate_prediction_quality": 0.875,
                        "aggregate_log_accuracy": 0.125,
                        "aggregate_normalized_log_accuracy_score": 0.875,
                    }
                ),
                encoding="utf-8",
            )
            (current_run_dir / "benchmark_human_readable_report.html").write_text(
                "<html></html>",
                encoding="utf-8",
            )

            write_runs_index(docs_dir=docs_dir, latest_run_name="unit-test-run")

            index_path = docs_dir / "index.html"
            index_html = index_path.read_text(encoding="utf-8")
            self.assertTrue(index_path.exists())
            self.assertIn("Benchmark Run Index", index_html)
            self.assertIn("older-run", index_html)
            self.assertIn("unit-test-run", index_html)
            self.assertIn("Current run", index_html)
            self.assertIn("older-run/benchmark_results.json", index_html)
            self.assertIn("older-run/benchmark_human_readable_report.html", index_html)
            self.assertIn("unit-test-run/benchmark_results.json", index_html)
            self.assertIn("unit-test-run/benchmark_human_readable_report.html", index_html)


if __name__ == "__main__":
    unittest.main()