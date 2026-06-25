from __future__ import annotations

import math
import unittest

import pandas as pd

from foresight_phys.analysis.decisions import (
    build_decision_rows,
    summarize_decisions,
)


def _row(key, *, exp=0, typ="float", dist="normal", p50=1.0, sigma=0.5, gt=1.0, f3=True, decade=True, model="m1", file_id="P1"):
    return {
        "model": model,
        "file_id": file_id,
        "experiment": exp,
        "key": key,
        "type": typ,
        "distribution": dist,
        "p50": p50,
        "sigma": sigma,
        "numeric_gt": gt,
        "within_factor_3": f3,
        "within_decade": decade,
    }


SCORED = pd.DataFrame(
    [
        # exp0 argmax: model picks a (highest p50) but b is truly best -> wrong, regret 1
        _row("a", exp=0, p50=3.0, gt=1.0),
        _row("b", exp=0, p50=2.0, gt=3.0),
        _row("c", exp=0, p50=1.0, gt=2.0),
        # exp1 monotonic increasing, predicted increasing, tight sigma
        _row("d", exp=1, p50=1.0, sigma=0.1, gt=1.0),
        _row("e", exp=1, p50=2.0, sigma=0.1, gt=2.0),
        _row("f", exp=1, p50=3.0, sigma=0.1, gt=3.0),
        # exp2 sign_vs_baseline positive effect, predicted positive
        _row("g", exp=2, p50=1.0, sigma=0.5, gt=1.0),
        _row("h", exp=2, p50=2.0, sigma=0.5, gt=5.0),
        # exp3 order_of_magnitude: one member within factor 3, one not
        _row("i", exp=3, f3=True, decade=True),
        _row("j", exp=3, f3=False, decade=True),
        # exp4 argmax but one member is categorical -> unscorable
        _row("k", exp=4, p50=2.0, gt=2.0),
        _row("cat", exp=4, typ="categorical", p50=None, gt=None),
    ]
)

SETS = {
    "P1": [
        {"experiment_index": 0, "member_keys": ["a", "b", "c"], "question_type": "argmax_select", "ordering_variable": "x"},
        {"experiment_index": 1, "member_keys": ["d", "e", "f"], "question_type": "monotonic_direction", "ordering_variable": "x"},
        {"experiment_index": 2, "member_keys": ["g", "h"], "question_type": "sign_vs_baseline", "ordering_variable": "base/treat"},
        {"experiment_index": 3, "member_keys": ["i", "j"], "question_type": "order_of_magnitude", "ordering_variable": "x"},
        {"experiment_index": 4, "member_keys": ["k", "cat"], "question_type": "argmax_select", "ordering_variable": "x"},
        {"experiment_index": 0, "member_keys": ["a", "missing"], "question_type": "argmax_select", "ordering_variable": "x"},
    ]
}


class BuildDecisionRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = build_decision_rows(SCORED, SETS)

    def _set(self, question_type, experiment):
        m = self.rows[(self.rows["question_type"] == question_type) & (self.rows["experiment"] == experiment)]
        return m.iloc[0]

    def test_argmax_regret_and_value(self) -> None:
        r = self._set("argmax_select", 0)
        self.assertTrue(r["scorable"])
        self.assertEqual(r["decision_correct"], 0.0)  # picked a, best is b
        self.assertAlmostEqual(r["regret"], 1.0)
        self.assertAlmostEqual(r["baseline_regret"], 0.5)
        self.assertAlmostEqual(r["decision_value"], -0.5)

    def test_monotonic_direction_correct_with_low_sign_brier(self) -> None:
        r = self._set("monotonic_direction", 1)
        self.assertEqual(r["direction_correct"], 1.0)
        self.assertLess(r["sign_brier"], 1e-3)

    def test_sign_vs_baseline_positive(self) -> None:
        r = self._set("sign_vs_baseline", 2)
        self.assertEqual(r["direction_correct"], 1.0)
        expected_p = 0.5 * (1 + math.erf((2.0 - 1.0) / math.sqrt(0.5) / math.sqrt(2)))
        self.assertAlmostEqual(r["sign_brier"], (expected_p - 1.0) ** 2, places=6)

    def test_order_of_magnitude_majority(self) -> None:
        r = self._set("order_of_magnitude", 3)
        self.assertAlmostEqual(r["oom_factor3"], 0.5)
        self.assertAlmostEqual(r["oom_decade"], 1.0)
        self.assertEqual(r["decision_correct"], 1.0)

    def test_non_numeric_member_makes_set_unscorable(self) -> None:
        r = self._set("argmax_select", 4)
        self.assertFalse(r["scorable"])

    def test_missing_member_makes_set_unscorable(self) -> None:
        missing = self.rows[(self.rows["set_index"] == 5)].iloc[0]
        self.assertFalse(missing["scorable"])


class SummarizeDecisionsTests(unittest.TestCase):
    def test_per_model_aggregates(self) -> None:
        rows = build_decision_rows(SCORED, SETS)
        summary = summarize_decisions(rows)
        self.assertEqual(summary["n_sets_total"], 6)
        self.assertEqual(summary["n_sets_unscorable"], 2)
        m1 = next(m for m in summary["per_model"] if m["model"] == "m1")
        # 4 scorable sets: argmax(0), monotonic(1), sign(1), oom(1)
        self.assertEqual(m1["n_scorable_sets"], 4)
        self.assertEqual(m1["selection_accuracy"], 0.0)
        self.assertEqual(m1["direction_accuracy"], 1.0)
        self.assertAlmostEqual(m1["mean_regret"], 1.0)
        self.assertAlmostEqual(m1["decision_value"], -0.5)
        # decision_correct over {0.0(argmax), 1.0(mono), 1.0(sign), 1.0(oom)} = 0.75
        self.assertAlmostEqual(m1["decision_accuracy_micro"], 0.75)


if __name__ == "__main__":
    unittest.main()
