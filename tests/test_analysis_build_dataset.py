from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from foresight_phys.analysis.build_dataset import build_scored_dataset
from foresight_phys.analysis.paths import AnalysisPaths


class BuildDatasetTests(unittest.TestCase):
    def test_zero_reference_numeric_rows_skip_log_accuracy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            paths = AnalysisPaths(project_root=project_root)
            paths.ensure_dirs()

            (project_root / "JSONs" / "filtered").mkdir(parents=True, exist_ok=True)
            (project_root / "JSONs" / "filtered" / "paper.json").write_text(
                json.dumps(
                    [
                        {
                            "experiment_description": "Zero test",
                            "experiment_results": {
                                "zero_crossing": {
                                    "type": "float",
                                    "description": "Zero-valued observable",
                                    "result": 0.0,
                                }
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
                        "paper_title": "Zero Test",
                        "experiment": 0,
                        "experiment_description": "Zero test",
                        "key": "zero_crossing",
                        "description": "Zero-valued observable",
                        "gt": "0.0",
                        "pred": "1.0",
                        "status_class": "status-mismatch",
                        "status_text": "zero mismatch",
                    }
                ]
            ).to_parquet(paths.predictions_parquet, index=False)

            df = build_scored_dataset(paths)

            self.assertIsNone(df.iloc[0]["log_accuracy"])
            self.assertIsNone(df.iloc[0]["normalized_log_accuracy_score"])
            self.assertEqual(df.iloc[0]["score"], 0.0)
            self.assertFalse(df.iloc[0]["correct"])


if __name__ == "__main__":
    unittest.main()