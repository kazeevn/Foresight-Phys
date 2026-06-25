from __future__ import annotations

import unittest

import pandas as pd

from foresight_phys.analysis.analyze import (
    GROUNDED_CENTRALITY_ORDER,
    _abstract_grounding,
    _dataset_composition,
    _dedup_aggregate,
    _foresight_lift,
    _headline_summary,
    _leakage_summary,
    _quality_by_label,
)


def _row(model, base, abl, key, q, *, typ="float", headline=False, centrality="secondary", surprise="uncertain", leak=False, appears=False):
    return {
        "model": model,
        "base_model": base,
        "ablation": abl,
        "file_id": "P",
        "experiment": 0,
        "key": key,
        "type": typ,
        "quality": q,
        "is_headline": headline,
        "centrality": centrality,
        "ex_ante_surprise": surprise,
        "leakage_sufficient": leak,
        "appears_in_abstract": appears,
    }


# Base model A has a name-only ablation twin; B is standard only.
FULL = pd.DataFrame(
    [
        _row("A", "A", "none", "f1", 0.9, headline=True, centrality="headline", surprise="surprising"),
        _row("A", "A", "none", "f2", 0.7, centrality="key_supporting", leak=True),
        _row("A", "A", "none", "f3", 0.8),
        _row("A", "A", "none", "f4", 0.6),
        _row("A [name-only]", "A", "name-only", "f1", 0.5, headline=True, centrality="headline", surprise="surprising"),
        _row("A [name-only]", "A", "name-only", "f2", 0.6, centrality="key_supporting", leak=True),
        _row("A [name-only]", "A", "name-only", "f3", 0.5),
        _row("A [name-only]", "A", "name-only", "f4", 0.5),
        _row("B", "B", "none", "f1", 0.8, headline=True, centrality="headline", surprise="surprising"),
        _row("B", "B", "none", "f2", 0.6, centrality="key_supporting", leak=True),
        _row("B", "B", "none", "f3", 0.5),
        _row("B", "B", "none", "f4", 0.5),
    ]
)
MAIN = FULL[FULL["ablation"] == "none"].copy()
SETS = {"P": [{"experiment_index": 0, "member_keys": ["f3", "f4"], "question_type": "monotonic_direction", "ordering_variable": "x"}]}


class AnalyzeSummaryTests(unittest.TestCase):
    def test_headline_summary(self) -> None:
        hs = _headline_summary(MAIN)
        self.assertEqual(hs["n_headline_fields"], 1)
        by_model = {r["model"]: r for r in hs["per_model"]}
        self.assertAlmostEqual(by_model["A"]["headline_quality_paper_macro"], 0.9)
        self.assertAlmostEqual(by_model["B"]["headline_quality_paper_macro"], 0.8)

    def test_quality_by_label(self) -> None:
        rows = {r["label"]: r for r in _quality_by_label(MAIN, "centrality")}
        self.assertEqual(rows["secondary"]["n_fields"], 2)
        self.assertAlmostEqual(rows["headline"]["quality__A"], 0.9)
        # secondary = mean(f3, f4) for A = 0.7
        self.assertAlmostEqual(rows["secondary"]["quality__A"], 0.7)

    def test_foresight_lift(self) -> None:
        rows = {r["base_model"]: r for r in _foresight_lift(FULL)}
        # Only A has a name-only twin.
        self.assertEqual(set(rows), {"A"})
        self.assertAlmostEqual(rows["A"]["quality_full"], 0.75)
        self.assertAlmostEqual(rows["A"]["quality_name_only"], 0.525)
        self.assertAlmostEqual(rows["A"]["foresight_lift"], 0.225)
        self.assertAlmostEqual(rows["A"]["foresight_lift_headline"], 0.4)

    def test_leakage_summary(self) -> None:
        ls = _leakage_summary(MAIN)
        self.assertEqual(ls["n_leaked_fields"], 1)
        by_model = {r["model"]: r for r in ls["per_model"]}
        # A clean = mean(f1,f3,f4) = mean(0.9,0.8,0.6); leaked = f2 = 0.7
        self.assertAlmostEqual(by_model["A"]["clean_quality_paper_macro"], (0.9 + 0.8 + 0.6) / 3)
        self.assertAlmostEqual(by_model["A"]["leaked_quality_paper_macro"], 0.7)

    def test_dedup_aggregate(self) -> None:
        rows = {r["model"]: r["dedup_quality_paper_macro"] for r in _dedup_aggregate(MAIN, SETS)}
        # A: singles f1,f2 + collapsed mean(f3,f4)=0.7 -> mean(0.9,0.7,0.7)
        self.assertAlmostEqual(rows["A"], (0.9 + 0.7 + 0.7) / 3)

    def test_abstract_grounding(self) -> None:
        df = pd.DataFrame([
            _row("A", "A", "none", "f1", 0.9, headline=True, appears=True),
            _row("A", "A", "none", "f2", 0.7, appears=False),
            _row("A", "A", "none", "f3", 0.8, appears=True),
            _row("A", "A", "none", "f4", 0.6, appears=False),
        ])
        ag = _abstract_grounding(df)
        self.assertEqual(ag["n_in_abstract_fields"], 2)
        self.assertAlmostEqual(ag["appears_rate_headline"], 1.0)
        self.assertAlmostEqual(ag["appears_rate_non_headline"], 1 / 3)
        q = {r["model"]: r["in_abstract_quality_paper_macro"] for r in ag["per_model"]}
        self.assertAlmostEqual(q["A"], (0.9 + 0.8) / 2)

    def test_grounded_centrality_order(self) -> None:
        df = pd.DataFrame([
            _row("A", "A", "none", "f1", 0.9, appears=True),  # -> in_abstract
            _row("A", "A", "none", "f2", 0.7, centrality="setup_or_control"),
            _row("A", "A", "none", "f3", 0.8, centrality="headline"),
        ])
        df["centrality_grounded"] = df["centrality"].where(~(df["appears_in_abstract"] == True), "in_abstract")  # noqa: E712
        rows = _quality_by_label(df, "centrality_grounded", order=GROUNDED_CENTRALITY_ORDER)
        labels = [r["label"] for r in rows]
        self.assertEqual(labels[0], "in_abstract")
        self.assertTrue(labels.index("headline") < labels.index("setup_or_control"))

    def test_dataset_composition(self) -> None:
        comp = _dataset_composition(MAIN)
        self.assertEqual(comp["n_fields"], 4)
        self.assertEqual(comp["by_centrality"]["secondary"], 2)
        self.assertEqual(comp["by_surprise"]["surprising"], 1)
        self.assertEqual(comp["most_field_rich_paper"], {"file_id": "P", "n_fields": 4})


if __name__ == "__main__":
    unittest.main()
