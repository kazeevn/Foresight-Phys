import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from foresight_phys.extract_cli import extract_single_url, parse_args
from foresight_phys.extraction import ExtractionResult, FilteringResult


def write_manifest(
    ids_path: Path,
    *,
    source_url: str,
    raw_output_path: Path,
    filtered_output_path: Path,
    paper_title: str = 'Recorded Paper',
) -> None:
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        json.dumps(
            {
                'runs': [
                    {
                        'source_url': source_url,
                        'paper_title': paper_title,
                        'raw_output_path': str(raw_output_path),
                        'filtered_output_path': str(filtered_output_path),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )


class ExtractCliPreflightTests(unittest.TestCase):
    def test_parse_args_accepts_service_tier(self) -> None:
        with patch(
            'sys.argv',
            [
                'foresight-phys-extract',
                '--paper-url',
                'https://example.com/paper.pdf',
                '--service-tier',
                'priority',
            ],
        ):
            args = parse_args()

        self.assertEqual(args.paper_url, 'https://example.com/paper.pdf')
        self.assertEqual(args.service_tier, 'priority')

    def test_skips_explicit_existing_outputs_before_openai(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_output_path = tmp_path / 'raw.json'
            filtered_output_path = tmp_path / 'filtered.json'
            raw_output_path.write_text('[]', encoding='utf-8')
            filtered_output_path.write_text('[]', encoding='utf-8')
            ids_path = tmp_path / '.cache' / 'ids.json'

            with patch(
                'foresight_phys.extract_cli.extract_experiments_from_url',
                side_effect=AssertionError('extract_experiments_from_url should not run'),
            ), patch(
                'foresight_phys.extract_cli.filter_experiments_for_benchmark',
                side_effect=AssertionError('filter_experiments_for_benchmark should not run'),
            ):
                result = extract_single_url(
                    paper_url='https://arxiv.org/pdf/2601.12345',
                    model='gpt-5.5',
                    service_tier='flex',
                    extraction_system_prompt='prompt',
                    extraction_system_prompt_path=tmp_path / 'prompt.txt',
                    raw_output_override=str(raw_output_path),
                    filtered_output_override=str(filtered_output_path),
                    ids_path=ids_path,
                    overwrite=False,
                )

            self.assertEqual(result['status'], 'skipped_existing_output')
            self.assertEqual(result['path_resolution'], 'explicit_paths')
            self.assertEqual(result['skipped_paths'], [str(raw_output_path), str(filtered_output_path)])

    def test_skips_arxiv_derived_existing_outputs_before_openai(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_output_path = tmp_path / 'JSONs' / 'raw' / '2601.12345.json'
            filtered_output_path = tmp_path / 'JSONs' / 'filtered' / '2601.12345.json'
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path.write_text('[]', encoding='utf-8')
            filtered_output_path.write_text('[]', encoding='utf-8')
            ids_path = tmp_path / '.cache' / 'ids.json'
            paper_url = 'https://arxiv.org/pdf/2601.12345'

            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                with patch(
                    'foresight_phys.extract_cli.extract_experiments_from_url',
                    side_effect=AssertionError('extract_experiments_from_url should not run'),
                ), patch(
                    'foresight_phys.extract_cli.filter_experiments_for_benchmark',
                    side_effect=AssertionError('filter_experiments_for_benchmark should not run'),
                ):
                    result = extract_single_url(
                        paper_url=paper_url,
                        model='gpt-5.5',
                        service_tier='flex',
                        extraction_system_prompt='prompt',
                        extraction_system_prompt_path=tmp_path / 'prompt.txt',
                        raw_output_override=None,
                        filtered_output_override=None,
                        ids_path=ids_path,
                        overwrite=False,
                    )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result['status'], 'skipped_existing_output')
            self.assertEqual(result['path_resolution'], 'default_arxiv_id')
            self.assertEqual(result['arxiv_id'], '2601.12345')

    def test_skip_payload_keeps_manifest_paper_title(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_output_path = tmp_path / 'JSONs' / 'raw' / '2601.12345.json'
            filtered_output_path = tmp_path / 'JSONs' / 'filtered' / '2601.12345.json'
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path.write_text('[]', encoding='utf-8')
            filtered_output_path.write_text('[]', encoding='utf-8')
            ids_path = tmp_path / '.cache' / 'ids.json'
            paper_url = 'https://arxiv.org/pdf/2601.12345'
            write_manifest(
                ids_path,
                source_url=paper_url,
                raw_output_path=raw_output_path,
                filtered_output_path=filtered_output_path,
            )

            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                with patch(
                    'foresight_phys.extract_cli.extract_experiments_from_url',
                    side_effect=AssertionError('extract_experiments_from_url should not run'),
                ), patch(
                    'foresight_phys.extract_cli.filter_experiments_for_benchmark',
                    side_effect=AssertionError('filter_experiments_for_benchmark should not run'),
                ):
                    result = extract_single_url(
                        paper_url=paper_url,
                        model='gpt-5.5',
                        service_tier='flex',
                        extraction_system_prompt='prompt',
                        extraction_system_prompt_path=tmp_path / 'prompt.txt',
                        raw_output_override=None,
                        filtered_output_override=None,
                        ids_path=ids_path,
                        overwrite=False,
                    )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result['paper_title'], 'Recorded Paper')

    def test_partial_existing_outputs_fail_before_openai(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_output_path = tmp_path / 'raw.json'
            filtered_output_path = tmp_path / 'filtered.json'
            raw_output_path.write_text('[]', encoding='utf-8')
            ids_path = tmp_path / '.cache' / 'ids.json'

            with patch(
                'foresight_phys.extract_cli.extract_experiments_from_url',
                side_effect=AssertionError('extract_experiments_from_url should not run'),
            ), patch(
                'foresight_phys.extract_cli.filter_experiments_for_benchmark',
                side_effect=AssertionError('filter_experiments_for_benchmark should not run'),
            ):
                with self.assertRaises(FileExistsError) as error:
                    extract_single_url(
                        paper_url='https://arxiv.org/pdf/2601.12345',
                        model='gpt-5.5',
                        service_tier='flex',
                        extraction_system_prompt='prompt',
                        extraction_system_prompt_path=tmp_path / 'prompt.txt',
                        raw_output_override=str(raw_output_path),
                        filtered_output_override=str(filtered_output_path),
                        ids_path=ids_path,
                        overwrite=False,
                    )

            self.assertIn(str(raw_output_path), str(error.exception))

    def test_overwrite_forces_processing_even_when_outputs_exist(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_output_path = tmp_path / 'raw.json'
            filtered_output_path = tmp_path / 'filtered.json'
            raw_output_path.write_text('[{"stale": true}]', encoding='utf-8')
            filtered_output_path.write_text('[{"stale": true}]', encoding='utf-8')
            ids_path = tmp_path / '.cache' / 'ids.json'

            extraction_result = ExtractionResult(
                paper_title='Fresh Paper',
                experiments=[
                    {
                        'experiment_description': 'desc',
                        'experiment_results': {
                            'bandgap_eV': {
                                'type': 'float',
                                'description': 'desc',
                                'result': 1.23,
                            }
                        },
                    }
                ],
                response_id='response-1',
            )
            filtering_result = FilteringResult(
                filtered_experiments=extraction_result.experiments,
                validity_by_experiment=[True],
                response_id='response-2',
            )

            with patch(
                'foresight_phys.extract_cli.extract_experiments_from_url',
                return_value=extraction_result,
            ) as extract_mock, patch(
                'foresight_phys.extract_cli.filter_experiments_for_benchmark',
                return_value=filtering_result,
            ) as filter_mock, patch(
                'foresight_phys.extract_cli.record_response_id',
            ) as record_mock:
                result = extract_single_url(
                    paper_url='https://arxiv.org/pdf/2601.12345',
                    model='gpt-5.5',
                    service_tier='priority',
                    extraction_system_prompt='prompt',
                    extraction_system_prompt_path=tmp_path / 'prompt.txt',
                    raw_output_override=str(raw_output_path),
                    filtered_output_override=str(filtered_output_path),
                    ids_path=ids_path,
                    overwrite=True,
                )

            extract_mock.assert_called_once()
            filter_mock.assert_called_once()
            record_mock.assert_called_once()
            self.assertEqual(extract_mock.call_args.kwargs['service_tier'], 'priority')
            self.assertEqual(filter_mock.call_args.kwargs['service_tier'], 'priority')
            self.assertEqual(result['experiments_extracted'], 1)
            self.assertEqual(result['arxiv_id'], '2601.12345')
            self.assertEqual(json.loads(raw_output_path.read_text(encoding='utf-8')), extraction_result.experiments)
            self.assertEqual(
                json.loads(filtered_output_path.read_text(encoding='utf-8')),
                filtering_result.filtered_experiments,
            )

    def test_non_arxiv_url_raises_not_implemented_before_openai(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            with patch(
                'foresight_phys.extract_cli.extract_experiments_from_url',
                side_effect=AssertionError('extract_experiments_from_url should not run'),
            ), patch(
                'foresight_phys.extract_cli.filter_experiments_for_benchmark',
                side_effect=AssertionError('filter_experiments_for_benchmark should not run'),
            ):
                with self.assertRaises(NotImplementedError):
                    extract_single_url(
                        paper_url='https://example.com/paper.pdf',
                        model='gpt-5.5',
                        service_tier='flex',
                        extraction_system_prompt='prompt',
                        extraction_system_prompt_path=tmp_path / 'prompt.txt',
                        raw_output_override=None,
                        filtered_output_override=None,
                        ids_path=tmp_path / '.cache' / 'ids.json',
                        overwrite=False,
                    )


if __name__ == '__main__':
    unittest.main()