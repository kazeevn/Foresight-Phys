from __future__ import annotations

# pylint: disable=broad-exception-caught

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .models import BenchmarkItem

try:
    from langfuse import Langfuse
except ImportError as exc:
    Langfuse = None
    LANGFUSE_IMPORT_ERROR = str(exc)
else:
    LANGFUSE_IMPORT_ERROR = None


@dataclass
class LangfuseRunLogger:
    enabled: bool
    model: str
    run_name: str | None = None
    session_id: str | None = None
    host: str | None = None
    client: Any | None = None
    warning: str | None = None
    trace_urls: list[str] | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LangfuseRunLogger":
        if args.disable_langfuse:
            return cls(enabled=False, model=args.model)

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = args.langfuse_host or os.getenv("LANGFUSE_HOST")

        if not public_key or not secret_key:
            return cls(
                enabled=False,
                model=args.model,
                warning="Langfuse disabled: LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY not set.",
            )

        if Langfuse is None:
            import_detail = f" ({LANGFUSE_IMPORT_ERROR})" if LANGFUSE_IMPORT_ERROR else ""
            return cls(
                enabled=False,
                model=args.model,
                warning=f"Langfuse disabled: package import failed{import_detail}.",
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = args.langfuse_run_name or f"foresight-phys-{timestamp}"
        session_id = f"{run_name}-{uuid.uuid4().hex[:8]}"

        client_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            client_kwargs["host"] = host

        try:
            client = Langfuse(**client_kwargs)
        except Exception as exc:
            return cls(
                enabled=False,
                model=args.model,
                warning=f"Langfuse disabled: failed to initialize client ({exc}).",
            )

        return cls(
            enabled=True,
            model=args.model,
            run_name=run_name,
            session_id=session_id,
            host=host,
            client=client,
            trace_urls=[],
        )

    def log_file_result(self, item: BenchmarkItem, metrics: dict[str, Any]) -> None:
        if not self.enabled or self.client is None:
            return

        try:
            with self.client.start_as_current_observation(
                name="llm-prediction",
                as_type="generation",
                model=self.model,
                input={
                    "file": item.file_name,
                    "masked_input": item.masked_input,
                },
                output={
                    "predicted": item.actual_output,
                },
                metadata={
                    "file": item.file_name,
                    "run_name": self.run_name,
                    "metrics": metrics,
                },
            ) as generation:
                generation.update_trace(
                    name="foresight-phys.file-eval",
                    session_id=self.session_id,
                    input={
                        "file": item.file_name,
                        "masked_input": item.masked_input,
                    },
                    output={
                        "predicted": item.actual_output,
                    },
                    metadata={
                        "file": item.file_name,
                        "model": self.model,
                        "run_name": self.run_name,
                        **metrics,
                    },
                    tags=["foresight-phys", "benchmark", "file-eval"],
                )

                if metrics.get("mape") is not None:
                    generation.score(
                        name="mape",
                        value=float(metrics["mape"]),
                        data_type="NUMERIC",
                        comment="Lower is better",
                    )

                accuracy = metrics.get("bool_categorical_accuracy")
                if accuracy is not None:
                    generation.score(
                        name="bool_categorical_accuracy",
                        value=float(accuracy),
                        data_type="NUMERIC",
                        comment="Higher is better",
                    )

                correction_payload = json.dumps(
                    item.expected_output,
                    ensure_ascii=False,
                    indent=2,
                )
                correction_common = {
                    "name": "output",
                    "value": correction_payload,
                    "dataType": "CORRECTION",
                    "source": "ANNOTATION",
                    "comment": "Ground-truth expected output",
                    "metadata": {
                        "file": item.file_name,
                        "run_name": self.run_name,
                    },
                }

                self.client.api.score.create(
                    request={
                        **correction_common,
                        "traceId": generation.trace_id,
                    }
                )

                self.client.api.score.create(
                    request={
                        **correction_common,
                        "traceId": generation.trace_id,
                        "observationId": generation.id,
                    }
                )

                if self.trace_urls is not None:
                    trace_url = self.client.get_trace_url(trace_id=generation.trace_id)
                    if trace_url:
                        self.trace_urls.append(trace_url)
        except Exception as exc:
            self.warning = f"Langfuse logging warning: {exc}"

    def log_run_summary(self, summary: dict[str, Any], args: argparse.Namespace) -> None:
        if not self.enabled or self.client is None:
            return

        try:
            with self.client.start_as_current_span(
                name="foresight-phys.run-summary",
                input={
                    "json_dir": args.json_dir,
                    "system_prompt": args.system_prompt,
                    "model": args.model,
                    "max_files": args.max_files,
                    "max_workers": args.max_workers,
                },
                output=summary,
                metadata={
                    "run_name": self.run_name,
                },
            ) as span:
                span.update_trace(
                    name="foresight-phys.run-summary",
                    session_id=self.session_id,
                    tags=["foresight-phys", "benchmark", "run-summary"],
                    metadata={"run_name": self.run_name},
                )

                if self.trace_urls is not None:
                    trace_url = self.client.get_trace_url(trace_id=span.trace_id)
                    if trace_url:
                        self.trace_urls.append(trace_url)
        except Exception as exc:
            self.warning = f"Langfuse logging warning: {exc}"

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exc:
            self.warning = f"Langfuse flush warning: {exc}"

    def summary(self) -> dict[str, Any]:
        primary_trace_url = self.trace_urls[0] if self.trace_urls else None
        return {
            "enabled": self.enabled,
            "run_name": self.run_name,
            "session_id": self.session_id,
            "host": self.host,
            "trace_url": primary_trace_url,
            "warning": self.warning,
        }