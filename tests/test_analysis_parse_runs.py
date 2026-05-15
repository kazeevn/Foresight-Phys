from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from foresight_phys.analysis.parse_runs import discover_run_dirs, parse_all_runs
from foresight_phys.analysis.paths import AnalysisPaths


class ParseRunsTests(unittest.TestCase):
    def test_discover_and_parse_run_dirs_from_summary_json_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_dir = project_root / "docs" / "run-one"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "model": "gpt-5.4-nano",
                "per_file": [
                    {
                        "file": "2601.12345.json",
                        "paper_title": "Test Paper",
                        "report_experiments": [
                            {
                                "experiment_index": 1,
                                "experiment_description": "Cooling curve",
                                "result_rows": [
                                    {
                                        "result_key": "temperature",
                                        "description": "Measured temperature",
                                        "type": "float",
                                        "ground_truth": 10.0,
                                        "predicted": 9.5,
                                        "distribution": "log_normal",
                                        "sigma": 0.3,
                                        "z": 0.1,
                                        "nll": 0.5,
                                        "quality": 0.95,
                                        "status_class": "status-numeric",
                                        "status_text": "z = +0.10",
                                    },
                                    {
                                        "result_key": "phase",
                                        "description": "Observed phase",
                                        "type": "categorical",
                                        "ground_truth": "solid",
                                        "predicted": "solid",
                                        "probabilities": {"solid": 0.7, "liquid": 0.3},
                                        "log_loss": 0.357,
                                        "quality": 0.6,
                                        "status_class": "status-match",
                                        "status_text": "P(solid) = 70%",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
            (run_dir / "benchmark_results.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            paths = AnalysisPaths(project_root=project_root)
            self.assertEqual(discover_run_dirs(paths.docs_dir), [run_dir])

            df = parse_all_runs(paths)

            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["run_name"], "run-one")
            self.assertEqual(df.iloc[0]["model"], "gpt-5.4-nano")
            self.assertEqual(df.iloc[0]["file"], "2601.12345.json")
            self.assertEqual(df.iloc[0]["paper_title"], "Test Paper")
            self.assertEqual(df.iloc[0]["experiment"], 0)
            self.assertEqual(df.iloc[0]["key"], "temperature")
            self.assertEqual(df.iloc[0]["type"], "float")
            self.assertEqual(df.iloc[0]["distribution"], "log_normal")
            self.assertAlmostEqual(df.iloc[0]["sigma"], 0.3)
            self.assertAlmostEqual(df.iloc[0]["z"], 0.1)
            self.assertAlmostEqual(df.iloc[0]["nll"], 0.5)
            self.assertAlmostEqual(df.iloc[0]["quality"], 0.95)
            self.assertEqual(df.iloc[1]["status_class"], "status-match")
            self.assertEqual(
                json.loads(df.iloc[1]["probabilities_json"]),
                {"solid": 0.7, "liquid": 0.3},
            )
            self.assertTrue(paths.predictions_parquet.exists())

    def test_parse_all_runs_formats_values_like_human_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_dir = project_root / "docs" / "run-two"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "model": "gpt-5.4-mini",
                "per_file": [
                    {
                        "file": "2601.54321.json",
                        "paper_title": "Formatting Test",
                        "report_experiments": [
                            {
                                "experiment_index": 1,
                                "experiment_description": "Formatting curve",
                                "result_rows": [
                                    {
                                        "result_key": "metadata",
                                        "description": "Structured metadata",
                                        "type": "categorical",
                                        "ground_truth": {"unit": "K"},
                                        "predicted": None,
                                        "status_class": "status-mismatch",
                                        "status_text": "missing prediction",
                                    },
                                    {
                                        "result_key": "stable",
                                        "description": "Stability flag",
                                        "type": "bool",
                                        "ground_truth": True,
                                        "predicted": False,
                                        "prob_true": 0.3,
                                        "status_class": "status-mismatch",
                                        "status_text": "P(true) = 30%",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
            (run_dir / "benchmark_results.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            df = parse_all_runs(AnalysisPaths(project_root=project_root))

            self.assertEqual(df.iloc[0]["gt"], '{"unit": "K"}')
            self.assertEqual(df.iloc[0]["pred"], "MISSING")
            self.assertEqual(df.iloc[1]["gt"], "true")
            self.assertEqual(df.iloc[1]["pred"], "false")
            self.assertAlmostEqual(df.iloc[1]["prob_true"], 0.3)


if __name__ == "__main__":
    unittest.main()
