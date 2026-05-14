from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

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
    def __init__(self, *, model: str = FORMULA_JUDGE_MODEL):
        self.model = model
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