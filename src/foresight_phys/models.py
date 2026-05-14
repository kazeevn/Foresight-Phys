from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt


@dataclass
class BenchmarkItem:
    file_name: str
    masked_input: Any
    expected_output: Any
    actual_output: Any
    paper_title: str | None = None


class BenchmarkPredictionResultField(BaseModel):
    model_config = ConfigDict(extra='forbid')

    key: str
    type: Literal['float', 'integer', 'bool', 'categorical', 'formula']
    description: str
    result: StrictFloat | StrictInt | StrictBool | str
    allowed_categorial_values: list[str] | None = Field(default=None)


class BenchmarkPredictionExperiment(BaseModel):
    model_config = ConfigDict(extra='forbid')

    experiment_description: str
    experiment_results: list[BenchmarkPredictionResultField]


class BenchmarkPredictionEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid')

    payload: list[BenchmarkPredictionExperiment]