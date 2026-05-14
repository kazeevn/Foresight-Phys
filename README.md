# Foresight-Phys Benchmark

A benchmark for AI ability to predict the results of physical experiments 

## Data
The descriptions were extracted from recent open-access papers by Kostya Novoselov. The extraction was done with Gemini 3.1 Pro.

## Technical workflow

- Loads each JSON file in `JSONs/`.
- Replaces every `experiment_results.*.result` value with `"TO_PREDICT"`.
- Prompts the LLM with:
	- system prompt from `src/foresight_phys/system_prompt.txt`
	- masked JSON as user input
- Runs OpenAI calls in parallel with retry/backoff using `tenacity`.
- Shows a live progress bar while files are being predicted.
- Parses the model JSON output and compares predicted `result` values to ground truth.
- Caches raw model predictions on disk so report/formatting changes can be iterated without re-calling the LLM.
- Computes per JSON file:
	- `mape` (Mean Absolute Percentage Error) for numeric predictions
	- `bool_categorical_accuracy` for boolean and categorical predictions
- Computes aggregate averages across files for both metrics.
- Generates a static HTML report with a left paper switcher and per-experiment tables (instead of raw markdown text).

## Setup (uv)

```bash
uv sync
```

This installs the project from the src layout and exposes the `foresight-phys` CLI.

Ensure `.env` contains your provider API keys (for OpenAI-compatible use, set `OPENAI_API_KEY`).

To enable Langfuse tracing (optional), also set:

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
# Optional (defaults to cloud host)
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Run benchmark

```bash
uv run foresight-phys
```

This writes:

- JSON summary to `docs/benchmark_results.json`
- Offline human-readable report to `docs/benchmark_human_readable_report.html`

### Useful options

```bash
uv run foresight-phys --max-files 2
uv run foresight-phys --max-workers 8
uv run foresight-phys --langfuse-run-name paper-benchmark-run-01
uv run foresight-phys --disable-langfuse
uv run foresight-phys --cache-path .cache/llm_predictions.json
uv run foresight-phys --disable-cache
uv run foresight-phys --output docs/benchmark_results.json
uv run foresight-phys --html-output docs/benchmark_human_readable_report.html
```

To disable HTML report generation:

```bash
uv run foresight-phys --html-output ""
```

The run writes summary output to `docs/benchmark_results.json`.

## Extract benchmark JSON from arXiv PDFs

Use the dedicated extraction CLI to generate new files in the same format as `JSONs/`.
The command sends the PDF to OpenAI by public URL using `input_file`, asks `gpt-5.5`
for rich standalone experiment descriptions, writes the extracted experiments to disk,
and stores the OpenAI response ID locally for traceability.

```bash
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269
```

By default this will:

- derive the output filename from the extracted paper title and write it under `JSONs/`
- append the OpenAI response ID and run metadata to `.cache/extraction_response_ids.json`
- preserve the benchmark-compatible JSON shape already used in `JSONs/`

Useful options:

```bash
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --output JSONs/my-paper.json
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --model gpt-5.5
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --ids-path .cache/extraction_ids.json
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --overwrite
```

The output JSON remains a top-level list of experiments so it can be consumed by the
existing benchmark pipeline without any format conversion.

## Langfuse dashboard view

- If Langfuse keys are present, each file evaluation is logged as a trace.
- Traces are grouped by a `session_id` per benchmark run and include:
	- masked input JSON
	- predicted JSON (`predicted`)
	- reference JSON (`reference`)
	- correction entry (`Corrected Output`) populated from the reference JSON
	- per-file metrics (`mape`, `bool_categorical_accuracy`, etc.)
- The same `session_id` is written to the output summary under `langfuse.session_id`.
- In Langfuse UI, filter traces by that `session_id` to see the full run dashboard.
