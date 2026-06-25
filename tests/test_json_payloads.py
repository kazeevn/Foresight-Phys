from __future__ import annotations

import unittest

from foresight_phys.constants import TO_PREDICT_TOKEN
from foresight_phys.json_payloads import (
    NAME_ONLY_DESCRIPTION,
    build_masked_payload,
    build_name_only_payload,
)

GROUND_TRUTH = [
    {
        "experiment_description": "A very specific ARPES setup with lots of context.",
        "experiment_results": {
            "bandgap_eV": {
                "type": "float",
                "description": "Measured optical bandgap.",
                "result": 1.42,
            },
            "phase": {
                "type": "categorical",
                "description": "Observed phase.",
                "result": "metallic",
                "allowed_categorial_values": ["metallic", "insulating"],
            },
        },
    }
]


class NameOnlyPayloadTests(unittest.TestCase):
    def test_masks_results_like_standard(self) -> None:
        name_only = build_name_only_payload(GROUND_TRUTH)
        results = name_only[0]["experiment_results"]
        self.assertEqual(results["bandgap_eV"]["result"], TO_PREDICT_TOKEN)
        self.assertEqual(results["phase"]["result"], TO_PREDICT_TOKEN)

    def test_strips_experiment_and_result_descriptions(self) -> None:
        name_only = build_name_only_payload(GROUND_TRUTH)
        self.assertEqual(name_only[0]["experiment_description"], NAME_ONLY_DESCRIPTION)
        for meta in name_only[0]["experiment_results"].values():
            self.assertEqual(meta["description"], "")

    def test_preserves_keys_types_and_allowed_values(self) -> None:
        name_only = build_name_only_payload(GROUND_TRUTH)
        results = name_only[0]["experiment_results"]
        self.assertEqual(set(results), {"bandgap_eV", "phase"})
        self.assertEqual(results["bandgap_eV"]["type"], "float")
        self.assertEqual(results["phase"]["allowed_categorial_values"], ["metallic", "insulating"])

    def test_does_not_mutate_input_or_full_masking(self) -> None:
        build_name_only_payload(GROUND_TRUTH)
        # Original untouched; standard masking keeps the real description.
        self.assertEqual(GROUND_TRUTH[0]["experiment_results"]["bandgap_eV"]["result"], 1.42)
        masked = build_masked_payload(GROUND_TRUTH)
        self.assertEqual(
            masked[0]["experiment_description"],
            "A very specific ARPES setup with lots of context.",
        )


if __name__ == "__main__":
    unittest.main()
