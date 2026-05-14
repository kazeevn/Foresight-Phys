"""Central paths used by the analysis pipeline.

All paths are anchored at the project root (current working directory by
default) so the same code works from CLI, scripts, and notebooks. Override
``project_root`` for tests if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path

    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    @property
    def analysis_docs_dir(self) -> Path:
        """User-facing analysis artifacts (committed): summary, writeup, plots."""
        return self.docs_dir / "analysis"

    @property
    def plots_dir(self) -> Path:
        """One PDF per figure goes here."""
        return self.analysis_docs_dir / "plots"

    @property
    def cache_dir(self) -> Path:
        """Local-only intermediate artifacts (parquet)."""
        return self.project_root / ".cache" / "analysis"

    @property
    def json_dir(self) -> Path:
        return self.project_root / "JSONs" / "filtered"

    @property
    def predictions_parquet(self) -> Path:
        return self.cache_dir / "predictions.parquet"

    @property
    def scored_parquet(self) -> Path:
        return self.cache_dir / "scored.parquet"

    @property
    def per_field_parquet(self) -> Path:
        return self.cache_dir / "per_field.parquet"

    @property
    def summary_json(self) -> Path:
        return self.analysis_docs_dir / "summary.json"

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_docs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


def default_paths() -> AnalysisPaths:
    return AnalysisPaths(project_root=Path.cwd().resolve())
