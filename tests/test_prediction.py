from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from foresight_phys.cache import PredictionCache
from foresight_phys.json_payloads import (
    build_masked_payload,
    build_prediction_cache_key,
    build_prediction_format_signature,
    build_prediction_text_format,
)
from foresight_phys.prediction import build_benchmark_items, call_openai_with_retry


def _example_float_field() -> dict:
    return {
        'key': 'bandgap_eV',
        'type': 'float',
        'description': 'Measured bandgap',
        'distribution': 'log_normal',
        'p10': 0.6,
        'p50': 1.23,
        'p90': 2.5,
    }


def _example_predicted_float_field(result: float = 1.5) -> dict:
    return {
        'type': 'float',
        'description': 'Measured bandgap',
        'result': result,
        'distribution': 'log_normal',
        'p10': result / 2.0,
        'p50': result,
        'p90': result * 2.0,
    }


class PredictionSchemaTests(unittest.TestCase):
    def test_prediction_schema_uses_anyof_union(self) -> None:
        schema = build_prediction_format_signature()
        experiment_results_schema = schema['$defs']['BenchmarkPredictionExperiment']['properties']['experiment_results']

        self.assertEqual(experiment_results_schema['type'], 'array')
        item_schema = experiment_results_schema['items']
        # OpenAI structured output rejects `oneOf`; we deliberately emit `anyOf`
        # via a bare Union (no Field(discriminator=...)).
        self.assertIn('anyOf', item_schema)
        self.assertNotIn('oneOf', item_schema)
        refs = sorted(entry.get('$ref', '') for entry in item_schema['anyOf'])
        self.assertEqual(
            refs,
            sorted([
                '#/$defs/BoolResult',
                '#/$defs/CategoricalResult',
                '#/$defs/FloatResult',
                '#/$defs/FormulaResult',
                '#/$defs/IntegerResult',
            ]),
        )

    def test_uses_pydantic_parse_with_full_uncertainty_payload(self) -> None:
        text_format = build_prediction_text_format()
        parsed_payload = text_format(
            payload=[
                {
                    'experiment_description': 'Example experiment',
                    'experiment_results': [
                        _example_float_field(),
                        {
                            'key': 'stable',
                            'type': 'bool',
                            'description': 'Stable',
                            'result': True,
                            'prob_true': 0.8,
                        },
                        {
                            'key': 'phase',
                            'type': 'categorical',
                            'description': 'Observed phase',
                            'result': 'solid',
                            'allowed_categorial_values': ['solid', 'liquid'],
                            'probabilities': [
                                {'value': 'solid', 'probability': 0.7},
                                {'value': 'liquid', 'probability': 0.3},
                            ],
                        },
                        {
                            'key': 'dispersion',
                            'type': 'formula',
                            'description': 'Dispersion',
                            'result': 'E = h * nu',
                            'confidence': 0.6,
                        },
                    ],
                }
            ]
        )

        with patch('foresight_phys.prediction.OpenAI') as openai_cls:
            client = openai_cls.return_value
            client.responses.parse.return_value = SimpleNamespace(output_parsed=parsed_payload)

            payload = call_openai_with_retry(
                model='gpt-5.4-nano',
                system_prompt='Predict outcomes.',
                masked_payload=[
                    {
                        'experiment_description': 'Example experiment',
                        'experiment_results': {
                            'bandgap_eV': {
                                'type': 'float',
                                'description': 'Measured bandgap',
                                'result': 'TO_PREDICT',
                            }
                        },
                    }
                ],
                text_format=text_format,
            )

        client.responses.parse.assert_called_once()
        # Categorical probabilities list should be normalized into a dict.
        phase_pred = payload[0]['experiment_results']['phase']
        self.assertEqual(
            phase_pred['probabilities'],
            {'solid': 0.7, 'liquid': 0.3},
        )
        bandgap_pred = payload[0]['experiment_results']['bandgap_eV']
        self.assertEqual(bandgap_pred['distribution'], 'log_normal')
        self.assertEqual(bandgap_pred['p10'], 0.6)
        self.assertEqual(bandgap_pred['p50'], 1.23)
        self.assertEqual(bandgap_pred['p90'], 2.5)
        # normalize_prediction_payload synthesises `result` from p50 for downstream display.
        self.assertEqual(bandgap_pred['result'], 1.23)
        self.assertEqual(payload[0]['experiment_results']['stable']['prob_true'], 0.8)
        self.assertEqual(payload[0]['experiment_results']['dispersion']['confidence'], 0.6)

    def test_build_benchmark_items_skips_top_level_empty_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            (json_dir / 'empty.json').write_text('[]', encoding='utf-8')
            (json_dir / 'paper.json').write_text(
                json.dumps(
                    [
                        {
                            'experiment_description': 'Example experiment',
                            'experiment_results': {
                                'bandgap_eV': {
                                    'type': 'float',
                                    'description': 'Measured bandgap',
                                    'result': 1.23,
                                }
                            },
                        }
                    ]
                ),
                encoding='utf-8',
            )

            prediction_cache = PredictionCache(
                enabled=False,
                path=json_dir / 'cache.json',
                entries={},
            )

            with patch(
                'foresight_phys.prediction.call_openai_with_retry',
                return_value=[
                    {
                        'experiment_description': 'Example experiment',
                        'experiment_results': {
                            'bandgap_eV': _example_predicted_float_field(1.5),
                        },
                    }
                ],
            ) as call_openai:
                items = build_benchmark_items(
                    json_dir=json_dir,
                    system_prompt='Predict outcomes.',
                    model='gpt-5.4-nano',
                    max_files=None,
                    max_workers=1,
                    prediction_cache=prediction_cache,
                )

        self.assertEqual([item.file_name for item in items], ['paper.json'])
        call_openai.assert_called_once()

    def test_build_benchmark_items_calls_model_once_per_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            (json_dir / 'paper.json').write_text(
                json.dumps(
                    [
                        {
                            'experiment_description': 'Experiment one',
                            'experiment_results': {
                                'bandgap_eV': {
                                    'type': 'float',
                                    'description': 'Measured bandgap',
                                    'result': 1.23,
                                }
                            },
                        },
                        {
                            'experiment_description': 'Experiment two',
                            'experiment_results': {
                                'phase': {
                                    'type': 'categorical',
                                    'description': 'Observed phase',
                                    'result': 'solid',
                                    'allowed_categorial_values': ['solid', 'liquid'],
                                }
                            },
                        },
                    ]
                ),
                encoding='utf-8',
            )

            prediction_cache = PredictionCache(
                enabled=False,
                path=json_dir / 'cache.json',
                entries={},
            )

            with patch(
                'foresight_phys.prediction.call_openai_with_retry',
                side_effect=[
                    [
                        {
                            'experiment_description': 'Experiment one',
                            'experiment_results': {
                                'bandgap_eV': _example_predicted_float_field(1.5),
                            },
                        }
                    ],
                    [
                        {
                            'experiment_description': 'Experiment two',
                            'experiment_results': {
                                'phase': {
                                    'type': 'categorical',
                                    'description': 'Observed phase',
                                    'result': 'liquid',
                                    'allowed_categorial_values': ['solid', 'liquid'],
                                    'probabilities': {'solid': 0.4, 'liquid': 0.6},
                                }
                            },
                        }
                    ],
                ],
            ) as call_openai:
                items = build_benchmark_items(
                    json_dir=json_dir,
                    system_prompt='Predict outcomes.',
                    model='gpt-5.4-nano',
                    max_files=None,
                    max_workers=1,
                    prediction_cache=prediction_cache,
                )

        self.assertEqual(call_openai.call_count, 2)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].file_name, 'paper.json')
        self.assertEqual(
            items[0].actual_output[1]['experiment_results']['phase']['probabilities'],
            {'solid': 0.4, 'liquid': 0.6},
        )

    def test_build_benchmark_items_raises_on_cache_miss_in_cache_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            paper_path = json_dir / 'paper.json'
            ground_truth = [
                {
                    'experiment_description': 'Example experiment',
                    'experiment_results': {
                        'bandgap_eV': {
                            'type': 'float',
                            'description': 'Measured bandgap',
                            'result': 1.23,
                        }
                    },
                }
            ]
            paper_path.write_text(json.dumps(ground_truth), encoding='utf-8')

            prediction_cache = PredictionCache(
                enabled=True,
                path=json_dir / 'cache.json',
                entries={},
                cache_only=True,
            )

            with patch('foresight_phys.prediction.call_openai_with_retry') as call_openai:
                with self.assertRaisesRegex(
                    ValueError,
                    r'Cache-only mode is enabled, but cached predictions are missing for paper\.json experiment 1\.',
                ):
                    build_benchmark_items(
                        json_dir=json_dir,
                        system_prompt='Prompt A',
                        model='gpt-5.4-nano',
                        max_files=None,
                        max_workers=1,
                        prediction_cache=prediction_cache,
                        cache_only=True,
                    )

        call_openai.assert_not_called()

    def test_build_benchmark_items_can_ignore_system_prompt_for_cache_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            paper_path = json_dir / 'paper.json'
            ground_truth = [
                {
                    'experiment_description': 'Example experiment',
                    'experiment_results': {
                        'bandgap_eV': {
                            'type': 'float',
                            'description': 'Measured bandgap',
                            'result': 1.23,
                        }
                    },
                }
            ]
            paper_path.write_text(json.dumps(ground_truth), encoding='utf-8')

            masked_payload = build_masked_payload(ground_truth)
            cached_prediction = {
                'experiment_description': 'Example experiment',
                'experiment_results': {
                    'bandgap_eV': _example_predicted_float_field(1.5),
                },
            }
            cache_key = build_prediction_cache_key(
                model='gpt-5.4-nano',
                system_prompt='Prompt A',
                masked_payload=[masked_payload[0]],
                response_format=build_prediction_format_signature(),
                include_system_prompt=False,
            )
            prediction_cache = PredictionCache(
                enabled=True,
                path=json_dir / 'cache.json',
                entries={cache_key: cached_prediction},
                ignore_system_prompt=True,
            )

            with patch('foresight_phys.prediction.call_openai_with_retry') as call_openai:
                items = build_benchmark_items(
                    json_dir=json_dir,
                    system_prompt='Prompt B',
                    model='gpt-5.4-nano',
                    max_files=None,
                    max_workers=1,
                    prediction_cache=prediction_cache,
                    cache_ignore_system_prompt=True,
                )

        call_openai.assert_not_called()
        self.assertEqual(items[0].actual_output[0]['experiment_results']['bandgap_eV']['result'], 1.5)

    def test_build_benchmark_items_can_prime_prompt_agnostic_cache_from_benchmark_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_dir = tmp_path / 'jsons'
            docs_dir = tmp_path / 'docs'
            run_dir = docs_dir / 'prior-run'
            json_dir.mkdir()
            run_dir.mkdir(parents=True)

            ground_truth = [
                {
                    'experiment_description': 'Example experiment',
                    'experiment_results': {
                        'bandgap_eV': {
                            'type': 'float',
                            'description': 'Measured bandgap',
                            'result': 1.23,
                        }
                    },
                }
            ]
            predicted_output = [
                {
                    'experiment_description': 'Example experiment',
                    'experiment_results': {
                        'bandgap_eV': _example_predicted_float_field(1.5),
                    },
                }
            ]
            (json_dir / 'paper.json').write_text(json.dumps(ground_truth), encoding='utf-8')
            (run_dir / 'benchmark_results.json').write_text(
                json.dumps(
                    {
                        'model': 'gpt-5.4',
                        'per_file': [
                            {
                                'file': 'paper.json',
                                'expected_output': ground_truth,
                                'actual_output': predicted_output,
                            }
                        ],
                    }
                ),
                encoding='utf-8',
            )

            prediction_cache = PredictionCache(
                enabled=True,
                path=tmp_path / 'cache.json',
                entries={},
                cache_only=True,
                ignore_system_prompt=True,
            )

            with patch('foresight_phys.prediction.call_openai_with_retry') as call_openai:
                items = build_benchmark_items(
                    json_dir=json_dir,
                    system_prompt='Prompt B',
                    model='gpt-5.4',
                    max_files=None,
                    max_workers=1,
                    prediction_cache=prediction_cache,
                    cache_only=True,
                    cache_ignore_system_prompt=True,
                    docs_dir=docs_dir,
                )

        call_openai.assert_not_called()
        self.assertEqual(items[0].actual_output[0]['experiment_results']['bandgap_eV']['result'], 1.5)

    def test_build_benchmark_items_attaches_manifest_paper_titles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            paper_path = json_dir / 'paper.json'
            paper_path.write_text(
                json.dumps(
                    [
                        {
                            'experiment_description': 'Example experiment',
                            'experiment_results': {
                                'bandgap_eV': {
                                    'type': 'float',
                                    'description': 'Measured bandgap',
                                    'result': 1.23,
                                }
                            },
                        }
                    ]
                ),
                encoding='utf-8',
            )

            prediction_cache = PredictionCache(
                enabled=False,
                path=json_dir / 'cache.json',
                entries={},
            )

            manifest = {
                'runs': [
                    {
                        'paper_title': 'Manifest Title',
                        'filtered_output_path': str(paper_path.resolve()),
                    }
                ]
            }

            with patch(
                'foresight_phys.prediction.call_openai_with_retry',
                return_value=[
                    {
                        'experiment_description': 'Example experiment',
                        'experiment_results': {
                            'bandgap_eV': _example_predicted_float_field(1.5),
                        },
                    }
                ],
            ), patch(
                'foresight_phys.prediction.load_response_id_manifest',
                return_value=manifest,
            ):
                items = build_benchmark_items(
                    json_dir=json_dir,
                    system_prompt='Predict outcomes.',
                    model='gpt-5.4-nano',
                    max_files=None,
                    max_workers=1,
                    prediction_cache=prediction_cache,
                )

        self.assertEqual(items[0].paper_title, 'Manifest Title')

    def test_build_benchmark_items_backfills_arxiv_titles_from_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            paper_path = json_dir / '2601.11796.json'
            paper_path.write_text(
                json.dumps(
                    [
                        {
                            'experiment_description': 'Example experiment',
                            'experiment_results': {
                                'bandgap_eV': {
                                    'type': 'float',
                                    'description': 'Measured bandgap',
                                    'result': 1.23,
                                }
                            },
                        }
                    ]
                ),
                encoding='utf-8',
            )

            prediction_cache = PredictionCache(
                enabled=False,
                path=json_dir / 'cache.json',
                entries={},
            )

            with patch(
                'foresight_phys.prediction.call_openai_with_retry',
                return_value=[
                    {
                        'experiment_description': 'Example experiment',
                        'experiment_results': {
                            'bandgap_eV': _example_predicted_float_field(1.5),
                        },
                    }
                ],
            ), patch(
                'foresight_phys.prediction.resolve_arxiv_titles',
                return_value={'2601.11796': 'Backfilled ArXiv Title'},
            ):
                items = build_benchmark_items(
                    json_dir=json_dir,
                    system_prompt='Predict outcomes.',
                    model='gpt-5.4-nano',
                    max_files=None,
                    max_workers=1,
                    prediction_cache=prediction_cache,
                )

        self.assertEqual(items[0].paper_title, 'Backfilled ArXiv Title')


class PredictionCachePersistenceTests(unittest.TestCase):
    def test_set_persists_entries_without_explicit_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / 'cache.json'
            args = argparse.Namespace(cache_path=str(cache_path), disable_cache=False)
            prediction_cache = PredictionCache.from_args(args)

            prediction_cache.set(
                'paper-a',
                {
                    'experiment_results': {
                        'bandgap_eV': {
                            'result': 1.23,
                        }
                    }
                },
            )

            persisted_payload = json.loads(cache_path.read_text(encoding='utf-8'))
            reloaded_cache = PredictionCache.from_args(args)

        self.assertEqual(
            persisted_payload,
            {
                'paper-a': {
                    'experiment_results': {
                        'bandgap_eV': {
                            'result': 1.23,
                        }
                    }
                }
            },
        )
        self.assertEqual(reloaded_cache.get('paper-a'), persisted_payload['paper-a'])

    def test_overlapping_cache_instances_merge_new_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / 'cache.json'
            args = argparse.Namespace(cache_path=str(cache_path), disable_cache=False)
            first_cache = PredictionCache.from_args(args)
            second_cache = PredictionCache.from_args(args)

            first_cache.set('paper-a', {'prediction': 'first'})
            second_cache.set('paper-b', {'prediction': 'second'})

            persisted_payload = json.loads(cache_path.read_text(encoding='utf-8'))

        self.assertEqual(
            persisted_payload,
            {
                'paper-a': {'prediction': 'first'},
                'paper-b': {'prediction': 'second'},
            },
        )

    def test_summary_reports_cache_modes(self) -> None:
        prediction_cache = PredictionCache(
            enabled=True,
            path=Path('cache.json'),
            entries={'paper-a': {'prediction': 'first'}},
            cache_only=True,
            ignore_system_prompt=True,
        )

        self.assertEqual(
            prediction_cache.summary(),
            {
                'enabled': True,
                'path': 'cache.json',
                'cache_only': True,
                'ignore_system_prompt': True,
                'hits': 0,
                'misses': 0,
                'entries': 1,
                'warning': None,
            },
        )


if __name__ == '__main__':
    unittest.main()
