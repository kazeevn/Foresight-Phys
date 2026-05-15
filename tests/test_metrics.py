from __future__ import annotations

import math
import unittest

from foresight_phys.metrics import (
    compute_experiment_metrics,
    compute_file_metrics,
    score_bool,
    score_categorical,
    score_formula,
    score_numeric,
)


class ScoreNumericTests(unittest.TestCase):
    def test_log_normal_perfect_prediction_has_quality_one(self) -> None:
        score = score_numeric(
            expected_value=1.0,
            actual_meta={"distribution": "log_normal", "result": 1.0, "sigma": 0.3},
        )
        self.assertAlmostEqual(score["z"], 0.0)
        self.assertAlmostEqual(score["quality"], 1.0)
        # NLL of N(0, sigma) at z=0 is log(sigma) + 0.5 log(2π).
        self.assertAlmostEqual(
            score["nll"],
            math.log(0.3) + 0.5 * math.log(2.0 * math.pi),
        )
        self.assertTrue(score["within_1sigma"])
        self.assertTrue(score["within_2sigma"])

    def test_log_normal_one_sigma_off_quality_matches_gaussian_density(self) -> None:
        # gt is 10^sigma away from result on log10 scale.
        score = score_numeric(
            expected_value=2.0,
            actual_meta={"distribution": "log_normal", "result": 1.0, "sigma": math.log10(2.0)},
        )
        self.assertAlmostEqual(score["z"], 1.0)
        self.assertAlmostEqual(score["quality"], math.exp(-0.5))

    def test_normal_distribution_uses_linear_residual(self) -> None:
        score = score_numeric(
            expected_value=10.0,
            actual_meta={"distribution": "normal", "result": 9.0, "sigma": 0.5},
        )
        self.assertAlmostEqual(score["z"], 2.0)
        self.assertAlmostEqual(score["quality"], math.exp(-2.0))

    def test_missing_prediction_gives_zero_quality(self) -> None:
        score = score_numeric(expected_value=1.0, actual_meta=None)
        self.assertEqual(score["quality"], 0.0)
        self.assertTrue(score["missing"])

    def test_log_normal_with_zero_gt_treated_as_max_penalty(self) -> None:
        score = score_numeric(
            expected_value=0.0,
            actual_meta={"distribution": "log_normal", "result": 1.0, "sigma": 0.3},
        )
        self.assertEqual(score["quality"], 0.0)


class ScoreBoolTests(unittest.TestCase):
    def test_uniform_prediction_has_brier_one_quarter(self) -> None:
        score = score_bool(
            expected_value=True,
            actual_meta={"result": True, "prob_true": 0.5},
        )
        self.assertAlmostEqual(score["brier"], 0.25)
        self.assertAlmostEqual(score["quality"], 0.75)

    def test_confident_correct_gets_high_quality(self) -> None:
        score = score_bool(
            expected_value=True,
            actual_meta={"result": True, "prob_true": 0.9},
        )
        self.assertTrue(score["correct"])
        self.assertAlmostEqual(score["brier"], 0.01)
        self.assertGreater(score["quality"], 0.9)

    def test_confident_wrong_is_catastrophic(self) -> None:
        score = score_bool(
            expected_value=False,
            actual_meta={"result": True, "prob_true": 0.9},
        )
        self.assertFalse(score["correct"])
        self.assertAlmostEqual(score["brier"], 0.81)


class ScoreCategoricalTests(unittest.TestCase):
    def test_peaked_at_truth_has_low_brier(self) -> None:
        score = score_categorical(
            expected_value="phase_A",
            expected_meta={"allowed_categorial_values": ["phase_A", "phase_B", "phase_C"]},
            actual_meta={
                "result": "phase_A",
                "probabilities": {"phase_A": 0.9, "phase_B": 0.05, "phase_C": 0.05},
            },
        )
        self.assertTrue(score["correct"])
        self.assertAlmostEqual(score["brier"], (0.9 - 1.0) ** 2 + 2 * 0.05 ** 2, places=6)

    def test_uniform_distribution_brier_is_k_minus_one_over_k(self) -> None:
        score = score_categorical(
            expected_value="phase_A",
            expected_meta={"allowed_categorial_values": ["phase_A", "phase_B", "phase_C"]},
            actual_meta={
                "result": "phase_A",
                "probabilities": {"phase_A": 1.0, "phase_B": 1.0, "phase_C": 1.0},
            },
        )
        # Uniform 1/3 each, truth one-hot at phase_A: (1/3-1)² + 2*(1/3)² = 6/9 = 2/3.
        self.assertAlmostEqual(score["brier"], 2.0 / 3.0, places=6)

    def test_list_shape_probabilities_are_accepted(self) -> None:
        score = score_categorical(
            expected_value="phase_A",
            expected_meta={"allowed_categorial_values": ["phase_A", "phase_B"]},
            actual_meta={
                "result": "phase_A",
                "probabilities": [
                    {"value": "phase_A", "probability": 0.8},
                    {"value": "phase_B", "probability": 0.2},
                ],
            },
        )
        self.assertAlmostEqual(score["brier"], (0.8 - 1.0) ** 2 + 0.2 ** 2, places=6)


class ScoreFormulaTests(unittest.TestCase):
    def test_equivalent_with_full_confidence_scores_perfect(self) -> None:
        score = score_formula(
            expected_value="E = h * nu",
            expected_meta={"description": ""},
            actual_meta={"result": "E = h*nu", "confidence": 1.0},
            experiment_description="",
            result_key="energy",
            formula_judge=None,
        )
        self.assertTrue(score["equivalent"])
        self.assertAlmostEqual(score["quality"], 1.0)

    def test_equivalent_with_zero_confidence_scores_zero(self) -> None:
        score = score_formula(
            expected_value="E = h * nu",
            expected_meta={"description": ""},
            actual_meta={"result": "E = h*nu", "confidence": 0.0},
            experiment_description="",
            result_key="energy",
            formula_judge=None,
        )
        self.assertTrue(score["equivalent"])
        self.assertAlmostEqual(score["quality"], 0.0)


class ComputeFileMetricsTests(unittest.TestCase):
    def test_aggregates_proper_scoring_per_type(self) -> None:
        expected_json = {
            "experiment_description": "Reference experiment",
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
                    "allowed_categorial_values": ["solid", "liquid"],
                },
                "stable": {
                    "type": "bool",
                    "description": "Stable",
                    "result": True,
                },
                "dispersion": {
                    "type": "formula",
                    "description": "Dispersion",
                    "result": "E = m c^2",
                },
            },
        }
        actual_json = {
            "experiment_description": "Reference experiment",
            "experiment_results": {
                "temperature": {
                    "type": "float",
                    "result": 10.0,
                    "distribution": "log_normal",
                    "sigma": 0.3,
                },
                "phase": {
                    "type": "categorical",
                    "result": "solid",
                    "allowed_categorial_values": ["solid", "liquid"],
                    "probabilities": {"solid": 0.8, "liquid": 0.2},
                },
                "stable": {
                    "type": "bool",
                    "result": True,
                    "prob_true": 0.9,
                },
                "dispersion": {
                    "type": "formula",
                    "result": "E=mc^2",
                    "confidence": 0.7,
                },
            },
        }

        metrics = compute_file_metrics(expected_json, actual_json)

        self.assertAlmostEqual(metrics["numeric_quality"], 1.0)
        self.assertAlmostEqual(metrics["coverage_1sigma"], 1.0)
        self.assertAlmostEqual(metrics["coverage_2sigma"], 1.0)
        self.assertAlmostEqual(metrics["bool_brier"], (0.9 - 1.0) ** 2)
        self.assertAlmostEqual(
            metrics["categorical_brier"],
            (0.8 - 1.0) ** 2 + 0.2 ** 2,
        )
        self.assertAlmostEqual(metrics["formula_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["formula_quality"], 1.0 - (1.0 - 0.7) ** 2)
        self.assertEqual(metrics["result_count"], 4)
        self.assertEqual(metrics["numeric_count"], 1)
        self.assertEqual(metrics["bool_count"], 1)
        self.assertEqual(metrics["categorical_count"], 1)
        self.assertEqual(metrics["formula_count"], 1)
        self.assertEqual(metrics["missing_predictions"], 0)

    def test_aggregates_across_experiments_with_equal_weight(self) -> None:
        # Two experiments. Experiment A has four near-perfect bools (p=0.9),
        # experiment B has one bool where p=0.5. The paper-level mean Brier is
        # the experiment-level mean.
        def _bool(value: bool, prob: float) -> dict:
            return {"type": "bool", "result": value, "prob_true": prob}

        expected_json = [
            {
                "experiment_description": "A",
                "experiment_results": {
                    "a1": {"type": "bool", "description": "", "result": True},
                    "a2": {"type": "bool", "description": "", "result": True},
                    "a3": {"type": "bool", "description": "", "result": True},
                    "a4": {"type": "bool", "description": "", "result": True},
                },
            },
            {
                "experiment_description": "B",
                "experiment_results": {
                    "b1": {"type": "bool", "description": "", "result": True},
                },
            },
        ]
        actual_json = [
            {
                "experiment_description": "A",
                "experiment_results": {
                    "a1": _bool(True, 0.9),
                    "a2": _bool(True, 0.9),
                    "a3": _bool(True, 0.9),
                    "a4": _bool(True, 0.9),
                },
            },
            {
                "experiment_description": "B",
                "experiment_results": {
                    "b1": _bool(True, 0.5),
                },
            },
        ]

        metrics = compute_file_metrics(expected_json, actual_json)
        # Exp A: four bools at p=0.9, all true → Brier = 0.01 each, mean 0.01.
        # Exp B: one bool at p=0.5 → Brier = 0.25.
        # Paper-level mean of experiment-level means: (0.01 + 0.25) / 2 = 0.13.
        self.assertAlmostEqual(metrics["bool_brier"], 0.5 * (0.01 + 0.25))

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
                    "description": "Stable",
                    "result": False,
                },
            },
        }
        actual_json = {"experiment_results": {}}

        metrics = compute_file_metrics(expected_json, actual_json)

        self.assertEqual(metrics["prediction_quality"], 0.0)
        self.assertEqual(metrics["numeric_quality"], 0.0)
        self.assertEqual(metrics["bool_quality"], 0.0)
        self.assertEqual(metrics["missing_predictions"], 2)

    def test_compute_experiment_metrics_matches_single_experiment_payload(self) -> None:
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
                "temperature": {
                    "type": "float",
                    "result": 10.0,
                    "distribution": "log_normal",
                    "sigma": 0.3,
                },
                "stable": {"type": "bool", "result": True, "prob_true": 0.7},
            },
        }

        experiment_metrics = compute_experiment_metrics(expected_experiment, actual_experiment)
        file_metrics_dict = compute_file_metrics(expected_experiment, actual_experiment)
        file_metrics_list = compute_file_metrics([expected_experiment], [actual_experiment])
        self.assertEqual(experiment_metrics, file_metrics_dict)
        self.assertEqual(experiment_metrics, file_metrics_list)


if __name__ == "__main__":
    unittest.main()
