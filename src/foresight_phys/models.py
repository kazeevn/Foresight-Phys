from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt


@dataclass
class BenchmarkItem:
    file_name: str
    masked_input: Any
    expected_output: Any
    actual_output: Any
    paper_title: str | None = None


class CategoryProbability(BaseModel):
    model_config = ConfigDict(extra='forbid')

    value: str
    probability: StrictFloat


class FloatResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str
    type: Literal['float']
    description: str
    result: StrictFloat | StrictInt
    distribution: Literal['normal', 'log_normal']
    sigma: StrictFloat


class IntegerResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str
    type: Literal['integer']
    description: str
    result: StrictFloat | StrictInt
    distribution: Literal['normal', 'log_normal']
    sigma: StrictFloat


class BoolResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str
    type: Literal['bool']
    description: str
    result: StrictBool
    prob_true: StrictFloat


class CategoricalResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str
    type: Literal['categorical']
    description: str
    result: str
    allowed_categorial_values: list[str]
    probabilities: list[CategoryProbability]


class FormulaResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str
    type: Literal['formula']
    description: str
    result: str
    confidence: StrictFloat


# NOTE: We deliberately do NOT use Field(discriminator='type') here. Pydantic
# emits `oneOf` for discriminated unions, but OpenAI's structured-output
# validator rejects `oneOf` — it only accepts `anyOf`. A bare Union produces
# `anyOf`, and Pydantic's smart-union mode still selects the correct variant
# based on the Literal `type` field at parse time.
BenchmarkPredictionResultField = Union[
    FloatResult, IntegerResult, BoolResult, CategoricalResult, FormulaResult,
]


class BenchmarkPredictionExperiment(BaseModel):
    model_config = ConfigDict(extra='forbid')

    experiment_description: str
    experiment_results: list[BenchmarkPredictionResultField]


class BenchmarkPredictionEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    payload: list[BenchmarkPredictionExperiment]
