from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from foresight_phys.json_payloads import build_prediction_format_signature, build_prediction_text_format
from foresight_phys.prediction import call_openai_with_retry


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


if __name__ == '__main__':
    unittest.main()