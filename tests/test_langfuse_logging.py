from __future__ import annotations

# pylint: disable=redefined-builtin

import argparse
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from foresight_phys.langfuse_logging import LangfuseRunLogger
from foresight_phys.models import BenchmarkItem


@dataclass
class FakeObservation:
    kwargs: dict[str, Any]
    trace_id: str = "trace-123"
    id: str = "observation-123"
    scores: list[dict[str, Any]] = field(default_factory=list)
    trace_io_calls: list[dict[str, Any]] = field(default_factory=list)

    def set_trace_io(self, *, input: Any = None, output: Any = None) -> "FakeObservation":
        self.trace_io_calls.append({"input": input, "output": output})
        return self

    def score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)


class FakeScoreService:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, request: dict[str, Any]) -> None:
        self.requests.append(request)


class FakeClient:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []
        self.propagated_attributes: list[dict[str, Any]] = []
        self.api = type("FakeApi", (), {"score": FakeScoreService()})()

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        observation = FakeObservation(kwargs=kwargs)
        self.observations.append(observation)
        yield observation

    @contextmanager
    def propagate_attributes(self, **kwargs: Any):
        self.propagated_attributes.append(kwargs)
        yield

    def get_trace_url(self, *, trace_id: str | None = None) -> str | None:
        if trace_id is None:
            return None
        return f"https://langfuse.example/trace/{trace_id}"


class LangfuseRunLoggerTests(unittest.TestCase):
    def test_log_file_result_uses_supported_trace_apis(self) -> None:
        client = FakeClient()
        logger = LangfuseRunLogger(
            enabled=True,
            model="gpt-5.4-nano",
            run_name="benchmark-run",
            session_id="benchmark-run-1234abcd",
            client=client,
            trace_urls=[],
        )
        item = BenchmarkItem(
            file_name="paper.json",
            masked_input={"experiment_description": "masked"},
            expected_output={"experiment_results": {"temperature": {"result": 1.0}}},
            actual_output={"experiment_results": {"temperature": {"result": 1.5}}},
        )

        logger.log_file_result(
            item,
            {
                "prediction_quality": 0.9,
                "log_accuracy": 0.1,
                "normalized_log_accuracy_score": 0.9,
                "bool_categorical_accuracy": 1.0,
                "formula_accuracy": 1.0,
            },
        )

        self.assertIsNone(logger.warning)
        self.assertEqual(len(client.observations), 1)
        self.assertEqual(len(client.propagated_attributes), 1)
        self.assertEqual(
            client.propagated_attributes[0],
            {
                "session_id": "benchmark-run-1234abcd",
                "trace_name": "foresight-phys.file-eval",
                "tags": ["foresight-phys", "benchmark", "file-eval"],
                "metadata": {
                    "file": "paper.json",
                    "model": "gpt-5.4-nano",
                    "run_name": "benchmark-run",
                },
            },
        )
        self.assertEqual(
            client.observations[0].trace_io_calls,
            [
                {
                    "input": {
                        "file": "paper.json",
                        "masked_input": {"experiment_description": "masked"},
                    },
                    "output": {
                        "predicted": {"experiment_results": {"temperature": {"result": 1.5}}},
                    },
                }
            ],
        )
        self.assertEqual(len(client.observations[0].scores), 5)
        self.assertEqual(len(client.api.score.requests), 2)
        self.assertEqual(
            logger.trace_urls,
            ["https://langfuse.example/trace/trace-123"],
        )

    def test_log_run_summary_uses_observation_api(self) -> None:
        client = FakeClient()
        logger = LangfuseRunLogger(
            enabled=True,
            model="gpt-5.4-nano",
            run_name="benchmark-run",
            session_id="benchmark-run-1234abcd",
            client=client,
            trace_urls=[],
        )
        args = argparse.Namespace(
            json_dir="JSONs/filtered",
            system_prompt="src/foresight_phys/system_prompt.txt",
            model="gpt-5.4-nano",
            max_files=10,
            max_workers=4,
        )
        summary = {"files_evaluated": 10}

        logger.log_run_summary(summary, args)

        self.assertIsNone(logger.warning)
        self.assertEqual(len(client.observations), 1)
        self.assertEqual(
            client.observations[0].kwargs,
            {
                "name": "foresight-phys.run-summary",
                "input": {
                    "json_dir": "JSONs/filtered",
                    "system_prompt": "src/foresight_phys/system_prompt.txt",
                    "model": "gpt-5.4-nano",
                    "max_files": 10,
                    "max_workers": 4,
                },
                "output": {"files_evaluated": 10},
                "metadata": {"run_name": "benchmark-run"},
            },
        )
        self.assertEqual(
            client.propagated_attributes,
            [
                {
                    "session_id": "benchmark-run-1234abcd",
                    "trace_name": "foresight-phys.run-summary",
                    "tags": ["foresight-phys", "benchmark", "run-summary"],
                    "metadata": {"run_name": "benchmark-run"},
                }
            ],
        )
        self.assertEqual(
            logger.trace_urls,
            ["https://langfuse.example/trace/trace-123"],
        )


if __name__ == "__main__":
    unittest.main()