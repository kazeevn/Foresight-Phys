from __future__ import annotations

import unittest

from foresight_phys.annotation import (
    ComparisonSet,
    FieldAnnotation,
    PaperAnnotationPayload,
    abstract_overlap,
    build_annotation_record,
    extract_title_and_abstract,
)


MARKDOWN = (
    "## Bandgap Paper Title\n\n"
    "Author One, Author Two\n\n"
    "We report an optical bandgap measurement of a novel semiconductor.\n\n"
    "## Introduction\n\n"
    "Body text that is not the abstract and should be ignored entirely here.\n"
)

EXPERIMENTS = [
    {
        "experiment_description": "Grow and measure.",
        "experiment_results": {
            "bandgap_eV": {"type": "float", "description": "optical bandgap", "result": 1.4},
            "growth_success": {"type": "bool", "description": "did growth work", "result": True},
        },
    },
    {
        "experiment_description": "Sweep series.",
        "experiment_results": {
            "ratio_x1": {"type": "float", "description": "ratio at x1", "result": 2.0},
            "ratio_x2": {"type": "float", "description": "ratio at x2", "result": 1.0},
        },
    },
]


class AbstractProxyTests(unittest.TestCase):
    def test_extract_title_and_abstract(self) -> None:
        title, abstract = extract_title_and_abstract(MARKDOWN)
        self.assertEqual(title, "Bandgap Paper Title")
        self.assertIn("optical bandgap measurement", abstract)
        self.assertNotIn("Body text", abstract)

    def test_overlap_true_when_key_words_in_abstract(self) -> None:
        _, abstract = extract_title_and_abstract(MARKDOWN)
        from foresight_phys.annotation import _content_words

        corpus = _content_words(abstract)
        overlap, appears = abstract_overlap(key="bandgap_eV", description="optical bandgap", corpus_words=corpus)
        self.assertGreaterEqual(overlap, 0.5)
        self.assertTrue(appears)

    def test_overlap_false_for_unrelated_field(self) -> None:
        _, abstract = extract_title_and_abstract(MARKDOWN)
        from foresight_phys.annotation import _content_words

        corpus = _content_words(abstract)
        _, appears = abstract_overlap(key="growth_success", description="did growth work", corpus_words=corpus)
        self.assertFalse(appears)


class BuildRecordTests(unittest.TestCase):
    def _payload(self) -> PaperAnnotationPayload:
        return PaperAnnotationPayload(
            field_annotations=[
                FieldAnnotation(experiment_index=0, key="bandgap_eV", centrality="headline", is_headline=True, ex_ante_surprise="uncertain", leakage_sufficient=False),
                FieldAnnotation(experiment_index=0, key="growth_success", centrality="setup_or_control", is_headline=False, ex_ante_surprise="implied_or_derivable", leakage_sufficient=False),
                FieldAnnotation(experiment_index=1, key="ratio_x1", centrality="key_supporting", is_headline=False, ex_ante_surprise="uncertain", leakage_sufficient=False),
                # unknown key -> must be dropped
                FieldAnnotation(experiment_index=1, key="ghost", centrality="secondary", is_headline=False, ex_ante_surprise="uncertain", leakage_sufficient=False),
            ],
            comparison_sets=[
                ComparisonSet(experiment_index=1, member_keys=["ratio_x1", "ratio_x2"], ordering_variable="x", question_type="monotonic_direction", description="trend"),
                ComparisonSet(experiment_index=1, member_keys=["ratio_x1"], ordering_variable="x", question_type="argmax_select", description="too small"),
                ComparisonSet(experiment_index=1, member_keys=["ratio_x1", "ghost"], ordering_variable="x", question_type="argmax_select", description="bad key"),
            ],
        )

    def test_coverage_is_normalised(self) -> None:
        record = build_annotation_record(
            arxiv_id="0000.00000",
            paper_title="",
            experiments=EXPERIMENTS,
            payload=self._payload(),
            model="test-model",
            response_id="resp_1",
            paper_markdown=MARKDOWN,
        )
        rows = {(r["experiment_index"], r["key"]): r for r in record["field_annotations"]}
        # Exactly the four real fields, no ghost.
        self.assertEqual(set(rows), {(0, "bandgap_eV"), (0, "growth_success"), (1, "ratio_x1"), (1, "ratio_x2")})
        # ratio_x2 was missing from the LLM output -> filled default.
        self.assertTrue(rows[(1, "ratio_x2")]["annotation_filled"])
        self.assertFalse(rows[(0, "bandgap_eV")]["annotation_filled"])
        # Title falls back to the markdown title when none provided.
        self.assertEqual(record["paper_title"], "Bandgap Paper Title")

    def test_abstract_proxy_attached(self) -> None:
        record = build_annotation_record(
            arxiv_id="0000.00000",
            paper_title="Given Title",
            experiments=EXPERIMENTS,
            payload=self._payload(),
            model="test-model",
            response_id="resp_1",
            paper_markdown=MARKDOWN,
        )
        rows = {(r["experiment_index"], r["key"]): r for r in record["field_annotations"]}
        self.assertTrue(rows[(0, "bandgap_eV")]["appears_in_abstract"])
        self.assertFalse(rows[(0, "growth_success")]["appears_in_abstract"])

    def test_only_valid_comparison_sets_kept(self) -> None:
        record = build_annotation_record(
            arxiv_id="0000.00000",
            paper_title="",
            experiments=EXPERIMENTS,
            payload=self._payload(),
            model="test-model",
            response_id="resp_1",
            paper_markdown=MARKDOWN,
        )
        self.assertEqual(len(record["comparison_sets"]), 1)
        kept = record["comparison_sets"][0]
        self.assertEqual(kept["member_keys"], ["ratio_x1", "ratio_x2"])


if __name__ == "__main__":
    unittest.main()
