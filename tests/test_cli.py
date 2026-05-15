from __future__ import annotations

import unittest
from unittest.mock import patch

from foresight_phys.cli import parse_args


class BenchmarkCliTests(unittest.TestCase):
    def test_parse_args_accepts_cache_mode_flags(self) -> None:
        with patch(
            'sys.argv',
            [
                'foresight-phys',
                '--cache-only',
                '--cache-ignore-system-prompt',
            ],
        ):
            args = parse_args()

        self.assertTrue(args.cache_only)
        self.assertTrue(args.cache_ignore_system_prompt)

    def test_parse_args_rejects_cache_only_with_disable_cache(self) -> None:
        with patch(
            'sys.argv',
            [
                'foresight-phys',
                '--cache-only',
                '--disable-cache',
            ],
        ):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
