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
	- `prediction_quality`, the average per-result score across the paper
	- `smape` (raw symmetric mean absolute percentage error) for numeric predictions with nonzero references
	- `normalized_smape_score`, the average of `1 - min(sMAPE, 1)` across numeric predictions with nonzero references
	- `bool_categorical_accuracy` for boolean and categorical predictions
	- `formula_accuracy` for formula predictions, judged by `gpt-5.4-nano`
- Uses the average per-paper `prediction_quality` as the main model-comparison result and also reports aggregate averages across files for all supporting metrics.
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

- JSON summary to `docs/<run-name>/benchmark_results.json`
- Offline human-readable report to `docs/<run-name>/benchmark_human_readable_report.html`

By default, `<run-name>` is generated as `<model>-<nice suffix>` and the same value is used for Langfuse grouping.

### Useful options

```bash
uv run foresight-phys --max-files 2
uv run foresight-phys --max-workers 8
uv run foresight-phys --run-name paper-benchmark-run-01
uv run foresight-phys --langfuse-run-name paper-benchmark-run-01
uv run foresight-phys --disable-langfuse
uv run foresight-phys --cache-path .cache/llm_predictions.json
uv run foresight-phys --disable-cache
uv run foresight-phys --output docs/custom-run/benchmark_results.json
uv run foresight-phys --html-output docs/custom-run/benchmark_human_readable_report.html
```

To disable HTML report generation:

```bash
uv run foresight-phys --html-output ""
```

The run writes summary output under `docs/<run-name>/` unless `--output` or `--html-output` overrides it.

## Extract benchmark JSON from arXiv PDFs

Use the dedicated extraction CLI to generate new files in the same format as the benchmark JSONs.
The command sends the PDF to OpenAI by public URL using `input_file`, asks the extraction
model for rich standalone experiment descriptions, then sends the extracted JSON through a
second `gpt-5.5` suitability pass that returns one validity boolean per experiment.
Formula-valued results must use `type: "formula"`, not `type: "string"`, and the
corresponding `experiment_description` must define every variable used in the formula.

```bash
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269
```

To process many papers from a text file with one PDF URL per line, use:

```bash
uv run foresight-phys-extract --paper-urls-file papers.txt
```

By default this will:

- derive the output filename from the extracted paper title
- write the raw parsed experiment list under `JSONs/raw/`
- write only the benchmark-suitable experiments under `JSONs/filtered/`
- append both OpenAI response IDs and run metadata to `.cache/extraction_response_ids.json`
- preserve the benchmark-compatible JSON shape in both output files
- check resolved output paths before calling OpenAI

Useful options:

```bash
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --output JSONs/filtered/my-paper.json
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --raw-output JSONs/raw/my-paper.json
uv run foresight-phys-extract --paper-urls-file papers.txt
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --model gpt-5.5
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --ids-path .cache/extraction_ids.json
uv run foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --overwrite
```

For `--paper-urls-file`, blank lines and lines starting with `#` are ignored. In batch mode,
the CLI writes each paper to its default title-derived path under `JSONs/raw/` and
`JSONs/filtered/`; `--output` and `--raw-output` remain single-paper-only overrides. Before
calling OpenAI, the CLI checks any output paths it can resolve up front. If both target JSONs
already exist and `--overwrite` is not set, that paper is skipped. If only one resolved target
already exists, the command fails fast without calling OpenAI. For default title-derived paths,
this preflight check uses the latest matching `source_url` entry in
`.cache/extraction_response_ids.json`, so a paper must have been processed once already before a
later run can skip it by URL alone.

This preflight behavior applies to both single-paper and batch runs.

If you want to regenerate outputs regardless of existing files, pass `--overwrite`.


The output JSON remains a top-level list of experiments so it can be consumed by the
existing benchmark pipeline without any format conversion. To benchmark newly filtered
files directly, point the benchmark CLI at `JSONs/filtered`, for example:

```bash
uv run foresight-phys --json-dir JSONs/filtered
```

## Langfuse dashboard view

- If Langfuse keys are present, each file evaluation is logged as a trace.
- Traces are grouped by a `session_id` per benchmark run and include:
	- masked input JSON
	- predicted JSON (`predicted`)
	- reference JSON (`reference`)
	- correction entry (`Corrected Output`) populated from the reference JSON
	- per-file metrics (`prediction_quality`, `smape`, `normalized_smape_score`, `bool_categorical_accuracy`, etc.)
- The same `session_id` is written to the output summary under `langfuse.session_id`.
- In Langfuse UI, filter traces by that `session_id` to see the full run dashboard.
