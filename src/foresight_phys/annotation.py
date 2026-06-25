"""Post-hoc, model-independent annotation of extracted benchmark fields.

This pass answers the two interpretability questions the raw benchmark cannot:
*are the scored fields the crucial ones?* and *do they form decision-relevant
contrasts?* It runs once per paper, with full knowledge of the ground-truth
results (it is meta-analysis, not prediction), and labels each result field with

- ``centrality``: how central the quantity is to the paper's claims,
- ``is_headline``: whether it is one of the paper's headline findings,
- ``ex_ante_surprise``: how strongly the value follows from generic physics
  priors *before* the experiment (the axis the benchmark motivation cares about),
- ``leakage_sufficient``: whether the (masked) experiment description alone is
  enough to determine the value without doing the experiment.

It also groups fields into ``comparison_sets`` — swept series or baseline/
treatment contrasts — together with the ordering variable and the kind of
decision they encode, which the decisions analysis scores as selection accuracy,
regret, and direction.

A cheap, auditable code-side proxy (``appears_in_abstract``) is added on top of
the LLM labels as an independent cross-check of centrality.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


DEFAULT_ANNOTATION_MODEL = "gpt-5.5"

CENTRALITY_VALUES = ("headline", "key_supporting", "secondary", "setup_or_control")
SURPRISE_VALUES = ("implied_or_derivable", "uncertain", "surprising")
QUESTION_TYPE_VALUES = (
    "argmax_select",
    "monotonic_direction",
    "sign_vs_baseline",
    "order_of_magnitude",
)

ANNOTATION_SYSTEM_PROMPT = """\
You are a senior physicist auditing a benchmark built from a physics paper. You are
given the full paper text and the list of experiments that were extracted from it,
WITH their ground-truth result values shown. This is meta-analysis with full
knowledge of the answers, not a prediction task.

For EVERY result field of EVERY experiment, assign:
- centrality: how central this quantity is to the paper's scientific contribution.
  * headline: a primary finding the paper is fundamentally about (the result a
    reader would cite; typically appears in the title/abstract).
  * key_supporting: an important quantity that substantiates a headline claim.
  * secondary: a real measurement that is contextual or incidental, not a main point.
  * setup_or_control: a yield/"did the process work" success flag, a calibration or
    control value, a sweep endpoint, or a sanity check.
- is_headline: true only for the small set of fields that are the paper's headline findings.
- ex_ante_surprise: how strongly the value follows from general physics priors BEFORE
  doing the experiment.
  * implied_or_derivable: obvious, conventional, or directly implied by the description
    (e.g., a published growth "succeeded"; a value restating a stated range).
  * uncertain: a competent physicist could not confidently call it in advance.
  * surprising: the result is unexpected or runs against the naive prior.
- leakage_sufficient: true if the experiment description ALONE (without performing the
  experiment) is enough to determine this value — e.g., the value merely restates a
  sweep range, a fixed setting, or a definition given in the description.

Also identify comparison_sets: groups of fields that form a decision-relevant comparison
within ONE experiment — a quantity measured across a swept control variable, or a
baseline-versus-intervention contrast. For each set give:
- experiment_index (0-based, matching the input),
- member_keys: the result keys that belong to the set (>= 2),
- ordering_variable: the physical control that varies across members (e.g., "Co
  composition x", "temperature", "baseline vs intercalated"),
- question_type:
  * argmax_select: which member condition maximizes/optimizes the quantity,
  * monotonic_direction: the direction of the trend across the ordering variable,
  * sign_vs_baseline: the sign of an intervention's effect relative to a baseline,
  * order_of_magnitude: the scale/magnitude of the quantity is the decision,
- description: one sentence stating the decision a researcher would make from this set.

Use the exact experiment_index (0-based) and result keys from the input. Annotate every
field exactly once. Only create comparison_sets that are genuinely comparable; do not
force unrelated fields together.
"""


class FieldAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_index: int = Field(description="0-based index of the experiment in the input list.")
    key: str = Field(description="The result key, exactly as given in the input.")
    centrality: Literal["headline", "key_supporting", "secondary", "setup_or_control"]
    is_headline: bool
    ex_ante_surprise: Literal["implied_or_derivable", "uncertain", "surprising"]
    leakage_sufficient: bool


class ComparisonSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_index: int
    member_keys: list[str]
    ordering_variable: str
    question_type: Literal[
        "argmax_select", "monotonic_direction", "sign_vs_baseline", "order_of_magnitude"
    ]
    description: str


class PaperAnnotationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_annotations: list[FieldAnnotation]
    comparison_sets: list[ComparisonSet]


@dataclass
class AnnotationResult:
    arxiv_id: str
    paper_title: str
    record: dict[str, Any]
    response_id: str


# ---------------------------------------------------------------------------
# Abstract proxy (code-side, no LLM) — an independent centrality cross-check.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "were", "per",
    "into", "over", "under", "between", "within", "via", "use", "used", "using", "vs",
    "near", "along", "each", "both", "its", "their", "than", "then", "when", "while",
    "measured", "value", "values", "result", "results", "approx", "approximate",
    "relative", "total", "number", "fraction", "ratio", "average", "mean", "max", "min",
    "maximum", "minimum", "reported",
}
# Unit / scale tokens that carry no semantic content for an abstract match.
_UNIT_WORDS = {
    "ev", "mev", "kev", "gev", "kelvin", "cm", "mm", "nm", "um", "micron", "angstrom",
    "inverse", "percent", "dimensionless", "snu", "deg", "degrees", "rad", "hz", "khz",
    "mhz", "ghz", "thz", "torr", "pa", "kpa", "mpa", "tesla", "gauss", "wb", "watt",
    "amp", "volt", "second", "seconds", "minute", "hour", "day", "days",
}


def _content_words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in _STOPWORDS and token not in _UNIT_WORDS
    }


def _key_content_words(key: str) -> set[str]:
    # snake_case → words; drop tokens that contain digits (e.g. x0_32, co0).
    parts = [part for part in re.split(r"[_\W]+", key) if part and not any(c.isdigit() for c in part)]
    return _content_words(" ".join(parts))


def extract_title_and_abstract(markdown_text: str) -> tuple[str, str]:
    """Best-effort title + abstract extraction from the generated markdown.

    The converted markdown puts the title in the first heading and the abstract as
    the long prose paragraph before the first "introduction" heading. We take the
    longest paragraph in that leading window as the abstract.
    """
    lines = markdown_text.splitlines()
    title = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            title = stripped.lstrip("#").strip()
            break

    intro_match = re.search(r"(?im)^#{1,3}\s*introduction\b", markdown_text)
    head = markdown_text[: intro_match.start()] if intro_match else markdown_text[:4000]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", head) if p.strip()]
    abstract = max(paragraphs, key=len) if paragraphs else ""
    return title, abstract


def abstract_overlap(*, key: str, description: str, corpus_words: set[str]) -> tuple[float, bool]:
    """Fraction of a field's content words that appear in title+abstract.

    Heuristic cross-check only: ``appears_in_abstract`` is True when at least half
    of the key's content words (or, failing an informative key, the description's)
    occur in the abstract corpus.
    """
    target = _key_content_words(key) or _content_words(description)
    if not target:
        return 0.0, False
    overlap = len(target & corpus_words) / len(target)
    return overlap, overlap >= 0.5


# ---------------------------------------------------------------------------
# Paper / experiment loading and prompt assembly.
# ---------------------------------------------------------------------------


def load_paper_markdown(arxiv_id: str, *, papers_dir: Path) -> str:
    md_path = papers_dir / f"{arxiv_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"No paper markdown for {arxiv_id} at {md_path}; annotation needs the paper text."
        )
    return md_path.read_text(encoding="utf-8")


def _experiments_as_indexed_json(experiments: list[dict[str, Any]]) -> str:
    indexed = [
        {
            "experiment_index": ei,
            "experiment_description": exp.get("experiment_description", ""),
            "experiment_results": exp.get("experiment_results", {}),
        }
        for ei, exp in enumerate(experiments)
    ]
    return json.dumps(indexed, ensure_ascii=False, indent=2)


def _enumerate_field_keys(experiments: list[dict[str, Any]]) -> list[tuple[int, str]]:
    keys: list[tuple[int, str]] = []
    for ei, exp in enumerate(experiments):
        for key in exp.get("experiment_results", {}):
            keys.append((ei, key))
    return keys


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
)
def annotate_paper_call(
    *,
    model: str,
    paper_markdown: str,
    experiments: list[dict[str, Any]],
    service_tier: str = "flex",
) -> tuple[PaperAnnotationPayload, str]:
    client = OpenAI()
    user_text = (
        "PAPER TEXT (markdown):\n"
        f"{paper_markdown}\n\n"
        "EXTRACTED EXPERIMENTS (with ground-truth results), 0-based experiment_index:\n"
        f"{_experiments_as_indexed_json(experiments)}\n\n"
        "Annotate every result field exactly once and list the comparison sets."
    )
    response = client.responses.parse(
        model=model,
        service_tier=service_tier,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": ANNOTATION_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        text_format=PaperAnnotationPayload,
    )
    parsed = getattr(response, "output_parsed", None)
    if not isinstance(parsed, PaperAnnotationPayload):
        raise ValueError("OpenAI structured output did not return a PaperAnnotationPayload.")
    response_id = getattr(response, "id", None)
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("OpenAI response did not include a response ID.")
    return parsed, response_id


def _default_field_annotation(experiment_index: int, key: str) -> dict[str, Any]:
    return {
        "experiment_index": experiment_index,
        "key": key,
        "centrality": "secondary",
        "is_headline": False,
        "ex_ante_surprise": "uncertain",
        "leakage_sufficient": False,
        "annotation_filled": True,
    }


def build_annotation_record(
    *,
    arxiv_id: str,
    paper_title: str,
    experiments: list[dict[str, Any]],
    payload: PaperAnnotationPayload,
    model: str,
    response_id: str,
    paper_markdown: str,
) -> dict[str, Any]:
    """Merge LLM labels with the abstract proxy and normalise field coverage.

    Every (experiment_index, key) present in the dataset gets exactly one record;
    LLM annotations for unknown keys are dropped, and missing keys are filled with a
    neutral default flagged ``annotation_filled`` so coverage gaps stay auditable.
    """
    title, abstract = extract_title_and_abstract(paper_markdown)
    corpus_words = _content_words(f"{title}\n{abstract}")

    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for ann in payload.field_annotations:
        by_key[(ann.experiment_index, ann.key)] = ann.model_dump()

    field_rows: list[dict[str, Any]] = []
    descriptions = {
        (ei, key): str(meta.get("description", ""))
        for ei, exp in enumerate(experiments)
        for key, meta in exp.get("experiment_results", {}).items()
    }
    for ei, key in _enumerate_field_keys(experiments):
        row = by_key.get((ei, key)) or _default_field_annotation(ei, key)
        row.setdefault("annotation_filled", False)
        overlap, appears = abstract_overlap(
            key=key,
            description=descriptions.get((ei, key), ""),
            corpus_words=corpus_words,
        )
        row["abstract_term_overlap"] = round(overlap, 4)
        row["appears_in_abstract"] = appears
        field_rows.append(row)

    valid_keys = set(_enumerate_field_keys(experiments))
    comparison_sets = [
        cs.model_dump()
        for cs in payload.comparison_sets
        if len(cs.member_keys) >= 2
        and all((cs.experiment_index, mk) in valid_keys for mk in cs.member_keys)
    ]

    return {
        "arxiv_id": arxiv_id,
        "paper_title": paper_title or title,
        "model": model,
        "response_id": response_id,
        "abstract_proxy": {"title": title, "abstract_chars": len(abstract)},
        "field_annotations": field_rows,
        "comparison_sets": comparison_sets,
    }


def resolve_annotation_output_path(arxiv_id: str, *, annotations_dir: Path) -> Path:
    return annotations_dir / f"{arxiv_id}.json"


def annotate_paper(
    *,
    arxiv_id: str,
    experiments: list[dict[str, Any]],
    paper_title: str,
    model: str = DEFAULT_ANNOTATION_MODEL,
    papers_dir: Path = Path("papers"),
    service_tier: str = "flex",
) -> AnnotationResult:
    paper_markdown = load_paper_markdown(arxiv_id, papers_dir=papers_dir)
    payload, response_id = annotate_paper_call(
        model=model,
        paper_markdown=paper_markdown,
        experiments=experiments,
        service_tier=service_tier,
    )
    record = build_annotation_record(
        arxiv_id=arxiv_id,
        paper_title=paper_title,
        experiments=experiments,
        payload=payload,
        model=model,
        response_id=response_id,
        paper_markdown=paper_markdown,
    )
    return AnnotationResult(
        arxiv_id=arxiv_id,
        paper_title=record["paper_title"],
        record=record,
        response_id=response_id,
    )
