from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from foresight_phys.cache import PredictionCache
from foresight_phys.json_payloads import build_prediction_format_signature, build_prediction_text_format
from foresight_phys.prediction import build_benchmark_items, call_openai_with_retry


class PredictionCallTests(unittest.TestCase):
    def test_prediction_schema_uses_list_based_result_entries(self) -> None:
        schema = build_prediction_format_signature()

        experiment_results_schema = schema['$defs']['BenchmarkPredictionExperiment']['properties']['experiment_results']

        self.assertEqual(experiment_results_schema['type'], 'array')
        self.assertEqual(
            experiment_results_schema['items']['$ref'],
            '#/$defs/BenchmarkPredictionResultField',
        )

    def test_uses_pydantic_parse_for_prediction_payloads(self) -> None:
        text_format = build_prediction_text_format()
        parsed_payload = text_format(
            payload=[
                {
                    'experiment_description': 'Example experiment',
                    'experiment_results': [
                        {
                            'key': 'bandgap_eV',
                            'type': 'float',
                            'description': 'Measured bandgap',
                            'result': 1.23,
                        }
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
        self.assertEqual(
            client.responses.parse.call_args.kwargs['text_format'],
            text_format,
        )
        self.assertEqual(
            payload,
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
            ],
        )

    def test_build_benchmark_items_skips_top_level_empty_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_dir = Path(tmp_dir)
            (json_dir / 'empty.json').write_text('[]', encoding='utf-8')
            (json_dir / 'paper.json').write_text(
                """
[
  {
    "experiment_description": "Example experiment",
    "experiment_results": {
      "bandgap_eV": {
        "type": "float",
        "description": "Measured bandgap",
        "result": 1.23
      }
    }
  }
]
""".strip(),
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
                            'bandgap_eV': {
                                'type': 'float',
                                'description': 'Measured bandgap',
                                'result': 1.5,
                            }
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
                                'bandgap_eV': {
                                    'type': 'float',
                                    'description': 'Measured bandgap',
                                    'result': 1.5,
                                }
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
            call_openai.call_args_list[0].kwargs['masked_payload'],
            [
                {
                    'experiment_description': 'Experiment one',
                    'experiment_results': {
                        'bandgap_eV': {
                            'type': 'float',
                            'description': 'Measured bandgap',
                            'result': 'TO_PREDICT',
                        }
                    },
                }
            ],
        )
        self.assertEqual(
            call_openai.call_args_list[1].kwargs['masked_payload'],
            [
                {
                    'experiment_description': 'Experiment two',
                    'experiment_results': {
                        'phase': {
                            'type': 'categorical',
                            'description': 'Observed phase',
                            'result': 'TO_PREDICT',
                            'allowed_categorial_values': ['solid', 'liquid'],
                        }
                    },
                }
            ],
        )
        self.assertEqual(
            items[0].actual_output,
            [
                {
                    'experiment_description': 'Experiment one',
                    'experiment_results': {
                        'bandgap_eV': {
                            'type': 'float',
                            'description': 'Measured bandgap',
                            'result': 1.5,
                        }
                    },
                },
                {
                    'experiment_description': 'Experiment two',
                    'experiment_results': {
                        'phase': {
                            'type': 'categorical',
                            'description': 'Observed phase',
                            'result': 'liquid',
                            'allowed_categorial_values': ['solid', 'liquid'],
                        }
                    },
                },
            ],
        )

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
                            'bandgap_eV': {
                                'type': 'float',
                                'description': 'Measured bandgap',
                                'result': 1.5,
                            }
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
                            'bandgap_eV': {
                                'type': 'float',
                                'description': 'Measured bandgap',
                                'result': 1.5,
                            }
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


if __name__ == '__main__':
    unittest.main()