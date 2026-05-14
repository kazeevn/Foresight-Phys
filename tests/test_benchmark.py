from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foresight_phys.benchmark import run_benchmark
from foresight_phys.models import BenchmarkItem


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
        self.assertAlmostEqual(summary["aggregate_smape"], 1.0)
        self.assertAlmostEqual(summary["aggregate_normalized_smape_score"], 0.5)
        self.assertAlmostEqual(summary["aggregate_bool_categorical_accuracy"], 0.5)
        self.assertAlmostEqual(summary["aggregate_formula_accuracy"], 1.0)

        self.assertEqual(written_summary["aggregate_prediction_quality"], summary["aggregate_prediction_quality"])
        self.assertEqual(written_summary["aggregate_smape"], summary["aggregate_smape"])


if __name__ == "__main__":
    unittest.main()