from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkItem:
    file_name: str
    masked_input: Any
    expected_output: Any
    actual_output: Any