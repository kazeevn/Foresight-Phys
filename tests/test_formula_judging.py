from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from foresight_phys.formula_judging import FormulaJudge


class FormulaJudgeCacheTests(unittest.TestCase):
    def test_cache_only_can_prime_from_benchmark_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            docs_dir = Path(tmp_dir) / 'docs'
            run_dir = docs_dir / 'prior-run'
            run_dir.mkdir(parents=True)
            run_dir.joinpath('benchmark_results.json').write_text(
                json.dumps(
                    {
                        'formula_judge_model': 'gpt-5.4-nano',
                        'per_file': [
                            {
                                'report_experiments': [
                                    {
                                        'experiment_description': 'Example experiment',
                                        'result_rows': [
                                            {
                                                'result_key': 'dispersion',
                                                'description': 'Dispersion relation',
                                                'type': 'formula',
                                                'ground_truth': 'E = mc^2',
                                                'predicted': 'E=mc^2+0',
                                                'equivalent': True,
                                                'status_text': 'match (conf 0.80)',
                                                'status_title': 'Equivalent after simplification.',
                                            }
                                        ],
                                    }
                                ]
                            }
                        ],
                    }
                ),
                encoding='utf-8',
            )

            judge = FormulaJudge(cache_only=True)
            judge.prime_from_benchmark_summaries(docs_dir=docs_dir)
            judgment = judge.judge(
                experiment_description='Example experiment',
                result_key='dispersion',
                result_description='Dispersion relation',
                reference_formula='E = mc^2',
                predicted_formula='E=mc^2+0',
            )

        self.assertTrue(judgment.equivalent)
        self.assertEqual(judgment.explanation, 'Equivalent after simplification.')

    def test_cache_only_raises_without_primed_formula_judgment(self) -> None:
        judge = FormulaJudge(cache_only=True)

        with self.assertRaisesRegex(
            ValueError,
            r'Cache-only mode is enabled, but cached formula judgment is missing for dispersion\.',
        ):
            judge.judge(
                experiment_description='Example experiment',
                result_key='dispersion',
                result_description='Dispersion relation',
                reference_formula='E = mc^2',
                predicted_formula='m c^2',
            )


if __name__ == '__main__':
    unittest.main()
