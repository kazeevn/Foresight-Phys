from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = PACKAGE_ROOT / 'system_prompt.txt'


def resolve_system_prompt_path(path: str | None) -> Path:
    if path:
        return Path(path)
    return DEFAULT_SYSTEM_PROMPT_PATH