from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl


@dataclass
class PredictionCache:
    enabled: bool
    path: Path
    entries: dict[str, Any]
    cache_only: bool = False
    ignore_system_prompt: bool = False
    warning: str | None = None
    hits: int = 0
    misses: int = 0
    _dirty: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PredictionCache":
        cache_path = Path(args.cache_path)
        cache_only = getattr(args, "cache_only", False)
        ignore_system_prompt = getattr(args, "cache_ignore_system_prompt", False)
        if args.disable_cache:
            return cls(
                enabled=False,
                path=cache_path,
                entries={},
                cache_only=cache_only,
                ignore_system_prompt=ignore_system_prompt,
            )

        if cache_path.exists():
            try:
                return cls(
                    enabled=True,
                    path=cache_path,
                    entries=cls._load_entries(cache_path),
                    cache_only=cache_only,
                    ignore_system_prompt=ignore_system_prompt,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return cls(
                    enabled=True,
                    path=cache_path,
                    entries={},
                    cache_only=cache_only,
                    ignore_system_prompt=ignore_system_prompt,
                    warning=f"Cache reset: failed reading {cache_path} ({exc}).",
                )

        return cls(
            enabled=True,
            path=cache_path,
            entries={},
            cache_only=cache_only,
            ignore_system_prompt=ignore_system_prompt,
        )

    @staticmethod
    def _load_entries(cache_path: Path) -> dict[str, Any]:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"expected JSON object at {cache_path}")
        return loaded

    @contextmanager
    def _exclusive_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_entries_for_flush(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}

        try:
            return self._load_entries(self.path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.warning = (
                f"Cache write warning: ignoring unreadable cache at {self.path} ({exc})"
            )
            return {}

    def _write_entries_atomically(self, entries: dict[str, Any]) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(entries, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)

            temp_path.replace(self.path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

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
        self.flush()

    def prime(self, entries: dict[str, Any]) -> None:
        if not self.enabled:
            return

        changed = False
        for key, value in entries.items():
            if key in self.entries:
                continue
            self.entries[key] = copy.deepcopy(value)
            changed = True

        if changed:
            self._dirty = True

    def flush(self) -> None:
        if not self.enabled or not self._dirty:
            return

        try:
            with self._exclusive_lock():
                on_disk_entries = self._load_entries_for_flush()
                merged_entries = dict(on_disk_entries)
                for key, value in self.entries.items():
                    merged_entries.setdefault(key, copy.deepcopy(value))

                if merged_entries != on_disk_entries or not self.path.exists():
                    self._write_entries_atomically(merged_entries)

                self.entries = merged_entries
            self._dirty = False
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.warning = f"Cache write warning: {exc}"

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "cache_only": self.cache_only,
            "ignore_system_prompt": self.ignore_system_prompt,
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "warning": self.warning,
        }
