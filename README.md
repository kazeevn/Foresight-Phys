# Foresight-Phys Benchmark

Foresight-Phys benchmarks whether an LLM can predict the outcomes of physical
experiments from rich experiment descriptions. The repository also includes a
pipeline for extracting benchmark-format experiment JSON from public paper PDFs.

## Repository layout
- `src/foresight_phys/`: benchmark, extraction, reporting, and prompt code
- `JSONs/raw/`: raw extracted experiment JSONs
- `JSONs/filtered/`: benchmark-ready JSONs after suitability filtering
- `papers.txt`: batch input file with one public PDF URL per line
- `docs/<run-name>/`: benchmark outputs

The current checked-in snapshot contains 36 JSON files in `JSONs/raw/` and 36
JSON files in `JSONs/filtered/`.

`JSONs/filtered/` is the default input directory for the benchmark CLI.

## Benchmark JSON format
Each benchmark file is a top-level list of experiments. Each experiment contains
an `experiment_description` string and an `experiment_results` object keyed by
result name.

```json
[
	{
		"experiment_description": "Self-contained experiment description.",
		"experiment_results": {
			"bandgap_eV": {
				"type": "float",
				"description": "Measured optical bandgap.",
				"result": 1.42
			}
		}
	}
]
```

Supported result types are `float`, `integer`, `bool`, `categorical`, and
`formula`. Categorical results may also include `allowed_categorial_values`.
Formula results must use `type: "formula"`, and the experiment description
must define every symbol used in the formula.

## Setup

Install the project and dependencies with `uv`:

```bash
uv sync
```

Put required credentials in `.env`:

```bash
OPENAI_API_KEY=...
```

Langfuse is optional. To enable tracing, also set:

```bash
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

The commands below use `uv run --env-file .env ...` so local `.env` settings
are loaded consistently.

## Run the benchmark

Run the default benchmark:

```bash
uv run --env-file .env foresight-phys
```

By default this:

- reads benchmark inputs from `JSONs/filtered/`
- writes `docs/index.html`
- uses model `gpt-5.4-nano`
- writes `docs/<run-name>/benchmark_results.json`
- writes `docs/<run-name>/benchmark_human_readable_report.html`
- caches raw predictions in `.cache/llm_predictions.json`
- merges cache updates safely across overlapping benchmark runs and persists each completed prediction immediately
- enables Langfuse logging when Langfuse keys are present

`<run-name>` is auto-generated from the model name plus a random readable suffix.
`--run-name` and `--langfuse-run-name` are aliases for the same value.

Useful examples:

```bash
uv run --env-file .env foresight-phys --max-files 2
uv run --env-file .env foresight-phys --json-dir JSONs/filtered --max-workers 8
uv run --env-file .env foresight-phys --model gpt-5.4-nano
uv run --env-file .env foresight-phys --run-name paper-benchmark-run-01
uv run --env-file .env foresight-phys --disable-langfuse
uv run --env-file .env foresight-phys --cache-path .cache/custom_predictions.json
uv run --env-file .env foresight-phys --disable-cache
uv run --env-file .env foresight-phys --output docs/custom-run/benchmark_results.json
uv run --env-file .env foresight-phys --html-output docs/custom-run/benchmark_human_readable_report.html
uv run --env-file .env foresight-phys --html-output ""
```

Setting `--html-output ""` disables HTML report generation.

## Benchmark workflow

For each JSON file, the benchmark pipeline:

1. Loads the ground-truth experiment JSON.
2. Replaces every `experiment_results.*.result` value with `"TO_PREDICT"`.
3. Splits the masked payload into one LLM request per experiment and sends each
	 experiment with the system prompt from `src/foresight_phys/system_prompt.txt`
	 to the OpenAI Responses API.
4. Runs requests in parallel with retry and exponential backoff.
5. Reuses cached experiment predictions when available and persists each new prediction to the shared cache as soon as it completes.
6. Compares predicted results against the reference JSON.
7. Writes a machine-readable JSON summary and an offline HTML report.

## Forecasts with uncertainty

Predictions must include calibrated uncertainty per field. The model returns,
on top of the point value `result`:

- `float` / `integer`: `distribution` (`normal` or `log_normal`) plus `sigma`
  (1σ width — in dex for log-normal, in linear units for normal). `result` is
  the median for log-normal and the mean for normal.
- `bool`: `prob_true ∈ [0, 1]`.
- `categorical`: `probabilities` — a list of `{value, probability}` covering
  every entry in `allowed_categorial_values`, summing to ≈1.
- `formula`: `confidence ∈ [0, 1]`.

The prediction cache key includes the schema, so changing this schema
invalidates `.cache/llm_predictions.json` and the cache will be repopulated on
the next run.

## Metrics

Predictions are scored with proper scoring rules. Per file, the benchmark
computes:

- `prediction_quality`: mean of per-field quality across all result fields.
  Quality is bounded in [0, 1] (`exp(-z²/2)` for numeric, `1 - brier` for
  bool / formula, `1 - ½·brier` for categorical).
- `numeric_nll`: mean NLL of numeric predictions under the chosen
  distribution. `0.5 z² + log(sigma) + 0.5 log(2π)` (capped at 30).
- `coverage_1sigma`, `coverage_2sigma`: fraction of numeric predictions with
  `|z| < 1` and `|z| < 2`. With well-calibrated uncertainty these target
  ≈0.68 and ≈0.95.
- `bool_log_loss`, `categorical_log_loss`, `formula_log_loss`: average
  log-loss of the predicted probability against the realised outcome.
- `bool_categorical_accuracy`: argmax accuracy over boolean and categorical
  fields (sanity check; not a proper score).
- `formula_accuracy`: fraction of formula fields judged equivalent.

The summary JSON also includes counts such as total result fields, bool,
categorical, numeric, formula fields, and missing predictions.

## Reports and outputs

`benchmark_results.json` contains:

- run metadata (`run_name`, `model`, `max_workers`)
- aggregate metrics across files
- formula judge model name
- cache metadata
- Langfuse metadata
- a `per_file` list with one metrics row per JSON file

The HTML report is a static offline file with:

- a left-hand paper switcher
- per-paper metric chips
- one table per experiment showing ground truth, prediction, and match status

## Extract benchmark JSON from paper PDFs

Use the extraction CLI to generate new benchmark-format JSON files from public
PDF URLs:

```bash
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269
```

To process many papers from a text file:

```bash
uv run --env-file .env foresight-phys-extract --paper-urls-file papers.txt
```

By default the extraction pipeline:

- uses model `gpt-5.5`
- only supports arXiv `abs`/`pdf` URLs
- sends the public PDF URL to OpenAI as an `input_file`
- extracts a paper title plus a top-level list of experiments
- runs a second suitability pass with `gpt-5.5` to keep only benchmark-ready experiments
- writes raw output to `JSONs/raw/<arXiv id>.json`
- writes filtered output to `JSONs/filtered/<arXiv id>.json`
- records response IDs, arXiv IDs, and output-path metadata in `.cache/extraction_response_ids.json`

Useful examples:

```bash
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --output JSONs/filtered/my-paper.json
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --raw-output JSONs/raw/my-paper.json
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --model gpt-5.5
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --service-tier priority
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --ids-path .cache/extraction_ids.json
uv run --env-file .env foresight-phys-extract --paper-url https://arxiv.org/pdf/2511.14269 --overwrite
uv run --env-file .env foresight-phys-extract --paper-urls-file papers.txt
```

Notes:

- `--output` and `--raw-output` are only valid for single-paper runs.
- Blank lines and lines starting with `#` are ignored in `papers.txt`.
- Non-arXiv URLs currently raise `NotImplementedError`.
- Raw and filtered outputs must be different files.
- If both resolved output files already exist and `--overwrite` is not set, the
	command skips that paper before calling OpenAI.
- If only one resolved output file already exists and `--overwrite` is not set,
	the command fails fast.
- Default output paths are derived directly from the arXiv ID in the URL, so
	early skip detection does not depend on a prior manifest entry.

To benchmark newly filtered files directly:

```bash
uv run --env-file .env foresight-phys --json-dir JSONs/filtered
```

## Cleanup empty filtered outputs

The repository also includes a small cleanup utility:

```bash
uv run --env-file .env foresight-phys-cleanup
```

This command mutates the dataset in place. It scans `JSONs/filtered/` for files
whose content is empty or `[]`, deletes those filtered files, and also deletes
the same-named file in `JSONs/raw/` when present. The cleanup utility currently
has no command-line options.

## Langfuse tracing

If Langfuse is enabled, the benchmark logs:

- one file-level generation per evaluated JSON file
- one run-summary trace for the benchmark run

The file-level traces include:

- the masked benchmark input
- the predicted output
- per-file metrics
- the ground-truth output as a correction payload

The summary JSON includes Langfuse metadata such as `enabled`, `run_name`,
`session_id`, `host`, `trace_url`, and any warning message. Filtering Langfuse
by `session_id` is the easiest way to inspect one complete benchmark run.
