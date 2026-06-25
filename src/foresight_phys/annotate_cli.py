"""CLI: annotate extracted benchmark papers with centrality / surprise / leakage
labels and decision-relevant comparison sets.

Writes one ``JSONs/annotations/<arxiv id>.json`` per non-empty filtered paper.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from .annotation import (
    DEFAULT_ANNOTATION_MODEL,
    annotate_paper,
    resolve_annotation_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate extracted benchmark papers (centrality, surprise, leakage, comparison sets)."
    )
    parser.add_argument("--json-dir", default="JSONs/filtered", help="Directory of filtered benchmark JSON files.")
    parser.add_argument("--annotations-dir", default="JSONs/annotations", help="Output directory for annotation JSON.")
    parser.add_argument("--papers-dir", default="papers", help="Directory of paper markdown files (<arxiv id>.md).")
    parser.add_argument("--model", default=DEFAULT_ANNOTATION_MODEL, help="OpenAI model for annotation.")
    parser.add_argument("--service-tier", default="flex", help="OpenAI Responses API service tier.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of papers processed.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel annotation requests.")
    parser.add_argument("--overwrite", action="store_true", help="Re-annotate papers even if an output file exists.")
    return parser.parse_args()


def _discover_papers(json_dir: Path, annotations_dir: Path, *, overwrite: bool, max_files: int | None) -> list[tuple[str, list[dict[str, Any]]]]:
    papers: list[tuple[str, list[dict[str, Any]]]] = []
    for json_path in sorted(json_dir.glob("*.json")):
        experiments = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(experiments, list) or not experiments:
            continue
        arxiv_id = json_path.stem
        if not overwrite and resolve_annotation_output_path(arxiv_id, annotations_dir=annotations_dir).exists():
            continue
        papers.append((arxiv_id, experiments))
    if max_files is not None:
        papers = papers[:max_files]
    return papers


def main() -> None:
    args = parse_args()
    load_dotenv()

    json_dir = Path(args.json_dir)
    annotations_dir = Path(args.annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = Path(args.papers_dir)

    papers = _discover_papers(json_dir, annotations_dir, overwrite=args.overwrite, max_files=args.max_files)
    if not papers:
        print(json.dumps({"papers_annotated": 0, "note": "nothing to do (all annotated or no input)."}))
        return

    results: list[dict[str, Any]] = []

    def _run(arxiv_id: str, experiments: list[dict[str, Any]]) -> dict[str, Any]:
        annotation = annotate_paper(
            arxiv_id=arxiv_id,
            experiments=experiments,
            paper_title="",
            model=args.model,
            papers_dir=papers_dir,
            service_tier=args.service_tier,
        )
        out_path = resolve_annotation_output_path(arxiv_id, annotations_dir=annotations_dir)
        out_path.write_text(json.dumps(annotation.record, ensure_ascii=False, indent=2), encoding="utf-8")
        n_fields = len(annotation.record["field_annotations"])
        n_headline = sum(1 for f in annotation.record["field_annotations"] if f["is_headline"])
        n_sets = len(annotation.record["comparison_sets"])
        return {
            "arxiv_id": arxiv_id,
            "output_path": str(out_path),
            "n_fields": n_fields,
            "n_headline": n_headline,
            "n_comparison_sets": n_sets,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run, arxiv_id, experiments): arxiv_id for arxiv_id, experiments in papers}
        with tqdm(total=len(futures), desc="Annotating", unit="paper") as progress:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(1)

    results.sort(key=lambda r: r["arxiv_id"])
    print(json.dumps({"papers_annotated": len(results), "runs": results}, ensure_ascii=False, indent=2))
