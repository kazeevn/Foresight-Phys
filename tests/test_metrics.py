from __future__ import annotations

import unittest

from foresight_phys.metrics import compute_file_metrics


class ComputeFileMetricsTests(unittest.TestCase):
    def test_computes_prediction_quality_and_supporting_metrics(self) -> None:
        expected_json = {
            "experiment_description": "Reference experiment",
            "experiment_results": {
                "temperature": {
                    "type": "float",
                    "description": "Measured temperature",
                    "result": 10.0,
                },
                "zero_crossing": {
                    "type": "float",
                    "description": "Zero-valued observable",
                    "result": 0.0,
                },
                "phase": {
                    "type": "categorical",
                    "description": "Observed phase",
                    "result": "solid",
                },
                "stable": {
                    "type": "bool",
                    "description": "System remains stable",
                    "result": True,
                },
                "dispersion": {
                    "type": "formula",
                    "description": "Dispersion relation",
                    "result": "E = mc^2",
                },
            },
        }
        actual_json = {
            "experiment_description": "Reference experiment",
            "experiment_results": {
                "temperature": {
                    "type": "float",
                    "description": "Measured temperature",
                    "result": 5.0,
                },
                "zero_crossing": {
                    "type": "float",
                    "description": "Zero-valued observable",
                    "result": "0",
                },
                "phase": {
                    "type": "categorical",
                    "description": "Observed phase",
                    "result": "liquid",
                },
                "stable": {
                    "type": "bool",
                    "description": "System remains stable",
                    "result": "yes",
                },
                "dispersion": {
                    "type": "formula",
                    "description": "Dispersion relation",
                    "result": "E=mc^2",
                },
            },
        }

        metrics = compute_file_metrics(expected_json, actual_json)

        self.assertAlmostEqual(metrics["prediction_quality"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["smape"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["normalized_smape_score"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["bool_categorical_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["formula_accuracy"], 1.0)
        self.assertEqual(metrics["result_count"], 5)
        self.assertEqual(metrics["numeric_count"], 2)
        self.assertEqual(metrics["nonzero_numeric_count"], 1)
        self.assertEqual(metrics["zero_reference_numeric_count"], 1)
        self.assertEqual(metrics["smape_count"], 1)
        self.assertEqual(metrics["normalized_smape_count"], 1)
        self.assertEqual(metrics["missing_predictions"], 0)

    def test_missing_numeric_prediction_counts_as_zero_quality(self) -> None:
        expected_json = {
            "experiment_description": "Missing prediction case",
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
        }
        actual_json = {"experiment_results": {}}

        metrics = compute_file_metrics(expected_json, actual_json)

        self.assertEqual(metrics["prediction_quality"], 0.0)
        self.assertIsNone(metrics["smape"])
        self.assertEqual(metrics["normalized_smape_score"], 0.0)
        self.assertEqual(metrics["bool_categorical_accuracy"], 0.0)
        self.assertEqual(metrics["missing_predictions"], 2)


if __name__ == "__main__":
    unittest.main()