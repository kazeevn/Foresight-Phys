from __future__ import annotations

import unittest

from foresight_phys.metrics import compute_experiment_metrics, compute_file_metrics


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

    def test_aggregates_across_experiments_with_equal_weight(self) -> None:
        # Two experiments in one paper. Experiment 1 has 4 fields (3 right, 1 wrong);
        # experiment 2 has 1 field (wrong). The legacy per-field-mean would give
        # 3/5 = 0.6. The intended per-experiment-then-paper mean is
        # mean(3/4, 0/1) = 0.375.
        expected_json = [
            {
                "experiment_description": "Experiment A",
                "experiment_results": {
                    "phase_a1": {"type": "categorical", "description": "", "result": "solid"},
                    "phase_a2": {"type": "categorical", "description": "", "result": "solid"},
                    "phase_a3": {"type": "categorical", "description": "", "result": "solid"},
                    "phase_a4": {"type": "categorical", "description": "", "result": "solid"},
                },
            },
            {
                "experiment_description": "Experiment B",
                "experiment_results": {
                    "phase_b1": {"type": "categorical", "description": "", "result": "solid"},
                },
            },
        ]
        actual_json = [
            {
                "experiment_description": "Experiment A",
                "experiment_results": {
                    "phase_a1": {"type": "categorical", "description": "", "result": "solid"},
                    "phase_a2": {"type": "categorical", "description": "", "result": "solid"},
                    "phase_a3": {"type": "categorical", "description": "", "result": "solid"},
                    "phase_a4": {"type": "categorical", "description": "", "result": "liquid"},
                },
            },
            {
                "experiment_description": "Experiment B",
                "experiment_results": {
                    "phase_b1": {"type": "categorical", "description": "", "result": "liquid"},
                },
            },
        ]

        metrics = compute_file_metrics(expected_json, actual_json)

        self.assertAlmostEqual(metrics["prediction_quality"], 0.375)
        self.assertAlmostEqual(metrics["bool_categorical_accuracy"], 0.375)
        # Count fields are sums.
        self.assertEqual(metrics["result_count"], 5)
        self.assertEqual(metrics["bool_categorical_count"], 5)

    def test_experiments_without_a_metric_type_are_skipped_for_that_metric(self) -> None:
        # Experiment A has only numeric fields, experiment B has only a bool
        # field. The paper's bool/cat accuracy should be 1.0 (only experiment B
        # contributes), not the field-level 1/3 it would have been previously.
        expected_json = [
            {
                "experiment_description": "Experiment A",
                "experiment_results": {
                    "temperature": {"type": "float", "description": "", "result": 10.0},
                    "pressure": {"type": "float", "description": "", "result": 1.0},
                },
            },
            {
                "experiment_description": "Experiment B",
                "experiment_results": {
                    "stable": {"type": "bool", "description": "", "result": True},
                },
            },
        ]
        actual_json = [
            {
                "experiment_description": "Experiment A",
                "experiment_results": {
                    "temperature": {"type": "float", "description": "", "result": 10.0},
                    "pressure": {"type": "float", "description": "", "result": 1.0},
                },
            },
            {
                "experiment_description": "Experiment B",
                "experiment_results": {
                    "stable": {"type": "bool", "description": "", "result": True},
                },
            },
        ]

        metrics = compute_file_metrics(expected_json, actual_json)

        self.assertAlmostEqual(metrics["bool_categorical_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["smape"], 0.0)
        self.assertAlmostEqual(metrics["prediction_quality"], 1.0)
        self.assertEqual(metrics["bool_categorical_count"], 1)
        self.assertEqual(metrics["numeric_count"], 2)

    def test_compute_experiment_metrics_matches_single_experiment_payload(self) -> None:
        # The single-experiment file_metrics call must agree with the new
        # per-experiment helper exactly — i.e. wrapping the same dict in a
        # one-element list must give the same answer.
        expected_experiment = {
            "experiment_description": "One experiment",
            "experiment_results": {
                "temperature": {"type": "float", "description": "", "result": 10.0},
                "stable": {"type": "bool", "description": "", "result": True},
            },
        }
        actual_experiment = {
            "experiment_description": "One experiment",
            "experiment_results": {
                "temperature": {"type": "float", "description": "", "result": 9.0},
                "stable": {"type": "bool", "description": "", "result": True},
            },
        }

        experiment_metrics = compute_experiment_metrics(expected_experiment, actual_experiment)
        file_metrics_dict = compute_file_metrics(expected_experiment, actual_experiment)
        file_metrics_list = compute_file_metrics([expected_experiment], [actual_experiment])
        self.assertEqual(experiment_metrics, file_metrics_dict)
        self.assertEqual(experiment_metrics, file_metrics_list)

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