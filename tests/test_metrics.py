from __future__ import annotations

import math
import unittest

from foresight_phys.metrics import (
    CRPS_SCALED_CAP,
    compute_experiment_metrics,
    compute_file_metrics,
    score_bool,
    score_categorical,
    score_formula,
    score_numeric,
)


def expected_crps(*, z: float, sigma: float) -> float:
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def expected_quality(crps_scaled: float) -> float:
    return 1.0 - min(crps_scaled, CRPS_SCALED_CAP) / CRPS_SCALED_CAP


Z_90 = 1.2815515655446004


def _quantile_meta_normal(*, p50: float, sigma: float) -> dict:
    half_span = Z_90 * sigma
    return {
        "distribution": "normal",
        "p10": p50 - half_span,
        "p50": p50,
        "p90": p50 + half_span,
    }


def _quantile_meta_log_normal(*, p50: float, sigma_dex: float) -> dict:
    half_span = Z_90 * sigma_dex
    return {
        "distribution": "log_normal",
        "p10": p50 * 10.0 ** (-half_span),
        "p50": p50,
        "p90": p50 * 10.0 ** half_span,
    }


class ScoreNumericTests(unittest.TestCase):
    def test_log_normal_centered_prediction_has_quality_from_crps(self) -> None:
        score = score_numeric(
            expected_value=1.0,
            actual_meta=_quantile_meta_log_normal(p50=1.0, sigma_dex=0.3),
        )
        self.assertAlmostEqual(score["z"], 0.0)
        # Quality is 1 - normalized CRPS / cap; even at z=0, a Gaussian with
        # nonzero sigma has nonzero CRPS, so quality < 1 (Dirac is the limit).
        self.assertAlmostEqual(
            score["crps"],
            expected_crps(z=0.0, sigma=0.3),
        )
        self.assertAlmostEqual(
            score["quality"], expected_quality(expected_crps(z=0.0, sigma=0.3))
        )
        self.assertTrue(score["within_1sigma"])
        self.assertTrue(score["within_2sigma"])

    def test_log_normal_one_sigma_off_quality_tracks_crps(self) -> None:
        # gt is 10^sigma away from p50 on log10 scale.
        sigma_dex = math.log10(2.0)
        score = score_numeric(
            expected_value=2.0,
            actual_meta=_quantile_meta_log_normal(p50=1.0, sigma_dex=sigma_dex),
        )
        self.assertAlmostEqual(score["z"], 1.0)
        crps = expected_crps(z=1.0, sigma=sigma_dex)
        self.assertAlmostEqual(score["crps"], crps)
        self.assertAlmostEqual(score["quality"], expected_quality(crps))

    def test_normal_distribution_uses_linear_residual(self) -> None:
        score = score_numeric(
            expected_value=10.0,
            actual_meta=_quantile_meta_normal(p50=9.0, sigma=0.5),
        )
        self.assertAlmostEqual(score["z"], 2.0)
        raw = expected_crps(z=2.0, sigma=0.5)
        self.assertAlmostEqual(score["crps"], raw)
        # Normal target: crps_scaled = raw / |y|; quality derives from it.
        self.assertAlmostEqual(score["quality"], expected_quality(raw / 10.0))

    def test_sigma_derived_from_p10_p90_span(self) -> None:
        # p10/p50/p90 = 8/9/10 ⇒ span 2 ⇒ sigma = 2 / (2 * Z_0.9) = 1/Z_0.9.
        score = score_numeric(
            expected_value=9.0,
            actual_meta={
                "distribution": "normal",
                "p10": 8.0,
                "p50": 9.0,
                "p90": 10.0,
            },
        )
        self.assertAlmostEqual(score["sigma"], 1.0 / Z_90)
        self.assertAlmostEqual(score["z"], 0.0)

    def test_missing_prediction_gives_zero_quality(self) -> None:
        score = score_numeric(expected_value=1.0, actual_meta=None)
        self.assertEqual(score["quality"], 0.0)
        self.assertTrue(score["missing"])

    def test_log_normal_with_zero_gt_treated_as_max_penalty(self) -> None:
        score = score_numeric(
            expected_value=0.0,
            actual_meta=_quantile_meta_log_normal(p50=1.0, sigma_dex=0.3),
        )
        self.assertEqual(score["crps"], 30.0)
        # Normalized CRPS is undefined when y == 0, so quality is too.
        self.assertIsNone(score["crps_scaled"])
        self.assertIsNone(score["quality"])

    def test_inverted_quantiles_treated_as_max_penalty(self) -> None:
        score = score_numeric(
            expected_value=1.0,
            actual_meta={
                "distribution": "normal",
                "p10": 2.0,
                "p50": 1.0,
                "p90": 0.0,
            },
        )
        self.assertEqual(score["crps"], 30.0)
        self.assertEqual(score["quality"], 0.0)

    def test_normal_scaled_crps_divides_by_abs_ground_truth(self) -> None:
        score = score_numeric(
            expected_value=10.0,
            actual_meta=_quantile_meta_normal(p50=9.0, sigma=0.5),
        )
        raw = expected_crps(z=2.0, sigma=0.5)
        self.assertAlmostEqual(score["crps_scaled"], raw / 10.0)

    def test_normal_scaled_crps_uses_absolute_value_for_negative_truth(self) -> None:
        score = score_numeric(
            expected_value=-10.0,
            actual_meta=_quantile_meta_normal(p50=-9.0, sigma=0.5),
        )
        raw = expected_crps(z=-2.0, sigma=0.5)
        self.assertAlmostEqual(score["crps_scaled"], raw / 10.0)

    def test_normal_scaled_crps_capped(self) -> None:
        # Tiny |y| with modest raw CRPS produces an enormous ratio: must cap.
        score = score_numeric(
            expected_value=1e-6,
            actual_meta=_quantile_meta_normal(p50=0.0, sigma=1.0),
        )
        self.assertEqual(score["crps_scaled"], CRPS_SCALED_CAP)

    def test_normal_scaled_crps_is_none_when_ground_truth_is_zero(self) -> None:
        score = score_numeric(
            expected_value=0.0,
            actual_meta=_quantile_meta_normal(p50=0.0, sigma=0.5),
        )
        self.assertIsNone(score["crps_scaled"])

    def test_log_normal_scaled_crps_equals_raw_dex_crps(self) -> None:
        score = score_numeric(
            expected_value=1.0,
            actual_meta=_quantile_meta_log_normal(p50=1.0, sigma_dex=0.3),
        )
        # Already in dex (unitless); scaled equals raw, capped at CRPS_SCALED_CAP.
        self.assertAlmostEqual(score["crps_scaled"], score["crps"])

    def test_log_normal_scaled_crps_capped(self) -> None:
        # 6 decades off with a moderately tight log-normal: raw dex-CRPS
        # exceeds the scaled-CRPS cap.
        score = score_numeric(
            expected_value=1e6,
            actual_meta=_quantile_meta_log_normal(p50=1.0, sigma_dex=0.5),
        )
        self.assertGreater(score["crps"], CRPS_SCALED_CAP)
        self.assertEqual(score["crps_scaled"], CRPS_SCALED_CAP)

    def test_missing_prediction_uses_max_scaled_penalty_when_y_nonzero(self) -> None:
        score = score_numeric(expected_value=5.0, actual_meta=None)
        self.assertEqual(score["crps_scaled"], CRPS_SCALED_CAP)

    def test_missing_prediction_scaled_is_none_when_y_zero(self) -> None:
        score = score_numeric(expected_value=0.0, actual_meta=None)
        self.assertIsNone(score["crps_scaled"])


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
                    **_quantile_meta_log_normal(p50=10.0, sigma_dex=0.3),
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

        crps_log_normal = expected_crps(z=0.0, sigma=0.3)
        self.assertAlmostEqual(metrics["numeric_quality"], expected_quality(crps_log_normal))
        self.assertAlmostEqual(metrics["numeric_crps"], crps_log_normal)
        self.assertAlmostEqual(metrics["numeric_crps_scaled"], crps_log_normal)
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
                    **_quantile_meta_log_normal(p50=10.0, sigma_dex=0.3),
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
