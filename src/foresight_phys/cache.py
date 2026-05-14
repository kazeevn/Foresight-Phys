from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PredictionCache:
    enabled: bool
    path: Path
    entries: dict[str, Any]
    warning: str | None = None
    hits: int = 0
    misses: int = 0
    _dirty: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PredictionCache":
        cache_path = Path(args.cache_path)
        if args.disable_cache:
            return cls(enabled=False, path=cache_path, entries={})

        if cache_path.exists():
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return cls(enabled=True, path=cache_path, entries=loaded)
                return cls(
                    enabled=True,
                    path=cache_path,
                    entries={},
                    warning=f"Cache reset: expected JSON object at {cache_path}.",
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return cls(
                    enabled=True,
                    path=cache_path,
                    entries={},
                    warning=f"Cache reset: failed reading {cache_path} ({exc}).",
                )

        return cls(enabled=True, path=cache_path, entries={})

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        if key in self.entries:
            self.hits += 1
            return copy.deepcopy(self.entries[key])
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.entries[key] = copy.deepcopy(value)
        self._dirty = True

    def flush(self) -> None:
        if not self.enabled or not self._dirty:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temp_path.write_text(
                json.dumps(self.entries, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temp_path.replace(self.path)
            self._dirty = False
        except (OSError, TypeError, ValueError) as exc:
            self.warning = f"Cache write warning: {exc}"

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "warning": self.warning,
        }