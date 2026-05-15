from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


FORMULA_JUDGE_MODEL = "gpt-5.4-nano"
FORMULA_JUDGE_SYSTEM_PROMPT = (
    "You are judging whether a predicted physics formula matches a reference formula for a "
    "specific experiment. Mark a prediction as equivalent only when it expresses the same "
    "physical relationship as the reference formula for the described result. Allow harmless "
    "formatting differences, equivalent notation, and algebraically equivalent rearrangements. "
    "Do not allow missing or extra terms, changed coefficients, wrong branches, wrong signs, "
    "wrong variables, or a different functional dependence. Use the experiment description and "
    "result description to interpret the variables, and be strict."
)


class FormulaJudgmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equivalent: StrictBool = Field(description="Whether the predicted formula should count as correct.")
    explanation: str = Field(description="A short explanation for the decision.")


@dataclass(frozen=True)
class FormulaJudgment:
    equivalent: bool
    explanation: str


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    ),
)
def judge_formula_equivalence(
    *,
    model: str,
    experiment_description: str,
    result_key: str,
    result_description: str,
    reference_formula: str,
    predicted_formula: str,
) -> FormulaJudgment:
    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": FORMULA_JUDGE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Experiment description:\n{experiment_description}\n\n"
                            f"Result key: {result_key}\n"
                            f"Result description: {result_description}\n\n"
                            f"Reference formula:\n{reference_formula}\n\n"
                            f"Predicted formula:\n{predicted_formula}"
                        ),
                    }
                ],
            },
        ],
        text_format=FormulaJudgmentPayload,
    )

    parsed = getattr(response, "output_parsed", None)
    if not isinstance(parsed, FormulaJudgmentPayload):
        raise ValueError("OpenAI formula judge did not return a FormulaJudgmentPayload.")

    explanation = parsed.explanation.strip() or "No explanation returned by formula judge."
    return FormulaJudgment(equivalent=bool(parsed.equivalent), explanation=explanation)


class FormulaJudge:
    def __init__(self, *, model: str = FORMULA_JUDGE_MODEL, cache_only: bool = False):
        self.model = model
        self.cache_only = cache_only
        self._memoized_judgments: dict[str, FormulaJudgment] = {}

    def _build_cache_key(
        self,
        *,
        experiment_description: str,
        result_key: str,
        result_description: str,
        reference_formula: str,
        predicted_formula: str,
    ) -> str:
        payload = {
            "model": self.model,
            "experiment_description": experiment_description,
            "result_key": result_key,
            "result_description": result_description,
            "reference_formula": reference_formula,
            "predicted_formula": predicted_formula,
        }
        canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def prime(self, judgments: dict[str, FormulaJudgment]) -> None:
        for key, judgment in judgments.items():
            self._memoized_judgments.setdefault(key, judgment)

    def prime_from_benchmark_summaries(self, *, docs_dir: Path = Path("docs")) -> None:
        if not docs_dir.exists():
            return

        primed_judgments: dict[str, FormulaJudgment] = {}
        summary_paths = sorted(docs_dir.glob('*/benchmark_results.json'), reverse=True)
        for summary_path in summary_paths:
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue

            if not isinstance(summary, dict) or summary.get("formula_judge_model") != self.model:
                continue

            per_file = summary.get("per_file")
            if not isinstance(per_file, list):
                continue

            for row in per_file:
                if not isinstance(row, dict):
                    continue

                report_experiments = row.get("report_experiments")
                if not isinstance(report_experiments, list):
                    continue

                for experiment in report_experiments:
                    if not isinstance(experiment, dict):
                        continue

                    experiment_description = experiment.get("experiment_description")
                    if not isinstance(experiment_description, str):
                        continue

                    result_rows = experiment.get("result_rows")
                    if not isinstance(result_rows, list):
                        continue

                    for result_row in result_rows:
                        if not isinstance(result_row, dict) or result_row.get("type") != "formula":
                            continue

                        result_key = result_row.get("result_key")
                        result_description = result_row.get("description")
                        reference_formula = result_row.get("ground_truth")
                        predicted_formula = result_row.get("predicted")
                        equivalent = result_row.get("equivalent")
                        if not (
                            isinstance(result_key, str)
                            and isinstance(result_description, str)
                            and isinstance(reference_formula, str)
                            and isinstance(predicted_formula, str)
                            and isinstance(equivalent, bool)
                        ):
                            continue

                        explanation = result_row.get("status_title") or result_row.get("status_text")
                        if not isinstance(explanation, str) or not explanation.strip():
                            explanation = "Recovered from benchmark report."

                        cache_key = self._build_cache_key(
                            experiment_description=experiment_description,
                            result_key=result_key,
                            result_description=result_description,
                            reference_formula=reference_formula,
                            predicted_formula=predicted_formula,
                        )
                        primed_judgments.setdefault(
                            cache_key,
                            FormulaJudgment(
                                equivalent=equivalent,
                                explanation=explanation,
                            ),
                        )

        self.prime(primed_judgments)

    def judge(
        self,
        *,
        experiment_description: str,
        result_key: str,
        result_description: str,
        reference_formula: str,
        predicted_formula: str,
    ) -> FormulaJudgment:
        cache_key = self._build_cache_key(
            experiment_description=experiment_description,
            result_key=result_key,
            result_description=result_description,
            reference_formula=reference_formula,
            predicted_formula=predicted_formula,
        )
        cached = self._memoized_judgments.get(cache_key)
        if cached is not None:
            return cached

        if self.cache_only:
            raise ValueError(
                "Cache-only mode is enabled, but cached formula judgment is missing for "
                f"{result_key}."
            )

        judgment = judge_formula_equivalence(
            model=self.model,
            experiment_description=experiment_description,
            result_key=result_key,
            result_description=result_description,
            reference_formula=reference_formula,
            predicted_formula=predicted_formula,
        )
        self._memoized_judgments[cache_key] = judgment
        return judgment
