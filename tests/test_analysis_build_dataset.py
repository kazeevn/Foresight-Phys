from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from foresight_phys.analysis.build_dataset import build_scored_dataset
from foresight_phys.analysis.paths import AnalysisPaths


class BuildDatasetTests(unittest.TestCase):
    def test_quality_is_passed_through_and_drives_correctness(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            paths = AnalysisPaths(project_root=project_root)
            paths.ensure_dirs()

            (project_root / "JSONs" / "filtered").mkdir(parents=True, exist_ok=True)
            (project_root / "JSONs" / "filtered" / "paper.json").write_text(
                json.dumps(
                    [
                        {
                            "experiment_description": "Numeric test",
                            "experiment_results": {
                                "bandgap_eV": {
                                    "type": "float",
                                    "description": "Bandgap",
                                    "result": 1.5,
                                },
                                "stable": {
                                    "type": "bool",
                                    "description": "Stable",
                                    "result": True,
                                },
                            },
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            pd.DataFrame(
                [
                    {
                        "run_name": "run-one",
                        "model": "gpt-5.4-nano",
                        "file": "paper.json",
                        "paper_title": "Numeric Test",
                        "experiment": 0,
                        "experiment_description": "Numeric test",
                        "key": "bandgap_eV",
                        "description": "Bandgap",
                        "type": "float",
                        "gt": "1.5",
                        "pred": "1.6",
                        "status_class": "status-numeric",
                        "status_text": "z = +0.10",
                        "distribution": "log_normal",
                        "sigma": 0.3,
                        "z": 0.1,
                        "crps": 0.5,
                        "quality": 0.95,
                        "prob_true": None,
                        "probabilities_json": None,
                        "confidence": None,
                        "equivalent": None,
                        "brier": None,
                    },
                    {
                        "run_name": "run-one",
                        "model": "gpt-5.4-nano",
                        "file": "paper.json",
                        "paper_title": "Numeric Test",
                        "experiment": 0,
                        "experiment_description": "Numeric test",
                        "key": "stable",
                        "description": "Stable",
                        "type": "bool",
                        "gt": "true",
                        "pred": "true",
                        "status_class": "status-match",
                        "status_text": "P(true) = 30%",
                        "distribution": None,
                        "sigma": None,
                        "z": None,
                        "crps": None,
                        "quality": 0.3,
                        "prob_true": 0.3,
                        "probabilities_json": None,
                        "confidence": None,
                        "equivalent": None,
                        "brier": 0.49,
                    },
                ]
            ).to_parquet(paths.predictions_parquet, index=False)

            df = build_scored_dataset(paths)

            numeric = df[df["key"] == "bandgap_eV"].iloc[0]
            self.assertAlmostEqual(numeric["quality"], 0.95)
            self.assertTrue(numeric["correct"])
            self.assertAlmostEqual(numeric["numeric_gt"], 1.5)
            self.assertAlmostEqual(numeric["numeric_pred"], 1.6)
            self.assertEqual(numeric["type"], "float")

            bool_row = df[df["key"] == "stable"].iloc[0]
            self.assertAlmostEqual(bool_row["quality"], 0.3)
            self.assertFalse(bool_row["correct"])
            self.assertEqual(bool_row["type"], "bool")


if __name__ == "__main__":
    unittest.main()
