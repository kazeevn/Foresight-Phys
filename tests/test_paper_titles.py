from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foresight_phys.paper_titles import (
    extract_arxiv_id_from_file_name,
    parse_arxiv_title_html,
    resolve_arxiv_titles,
)


class PaperTitleTests(unittest.TestCase):
    def test_extract_arxiv_id_from_file_name(self) -> None:
        self.assertEqual(extract_arxiv_id_from_file_name('2601.11796.json'), '2601.11796')
        self.assertIsNone(extract_arxiv_id_from_file_name('not-an-arxiv-id.json'))

    def test_parse_arxiv_title_from_citation_meta_tag(self) -> None:
        html_text = (
            '<html><head>'
            '<meta name="citation_title" content="Example &amp; Test Title">'
            '</head><body></body></html>'
        )

        self.assertEqual(
            parse_arxiv_title_html(html_text),
            'Example & Test Title',
        )

    def test_resolve_arxiv_titles_uses_and_updates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / 'arxiv_titles.json'

            with patch(
                'foresight_phys.paper_titles.fetch_arxiv_title',
                return_value='Fetched Title',
            ) as fetch_title:
                titles = resolve_arxiv_titles(['2601.11796'], cache_path=cache_path)

            self.assertEqual(titles, {'2601.11796': 'Fetched Title'})
            self.assertTrue(cache_path.exists())
            fetch_title.assert_called_once_with('2601.11796')

            with patch('foresight_phys.paper_titles.fetch_arxiv_title') as fetch_title:
                cached_titles = resolve_arxiv_titles(['2601.11796'], cache_path=cache_path)

            self.assertEqual(cached_titles, {'2601.11796': 'Fetched Title'})
            fetch_title.assert_not_called()


if __name__ == '__main__':
    unittest.main()