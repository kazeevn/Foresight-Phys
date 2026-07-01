![Static Badge](https://img.shields.io/badge/Forecast%40ICML26-Paper-blue?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DGhUK6VGW67)


# Foresight-Phys Benchmark

Foresight-Phys benchmarks whether an LLM can predict the outcomes of physical
experiments from rich experiment descriptions. The repository also includes a
pipeline for extracting benchmark-format experiment JSON from public paper PDFs
and a post-hoc analysis pipeline for comparing benchmark runs.

## Repository layout
- `src/foresight_phys/`: benchmark, extraction, cleanup, analysis, reporting, and prompt code
- `JSONs/raw/`: raw extracted experiment JSONs
- `JSONs/filtered/`: benchmark-ready JSONs after suitability filtering
- `papers.txt`: batch input file with one public PDF URL per line
- `docs/<run-name>/`: benchmark outputs
- `docs/analysis/`: committed cross-run analysis summaries and plots
- `.cache/analysis/`: local intermediate parquet files for the analysis pipeline
- `papers/`: local paper PDFs and markdown conversions used during development

The current checked-in snapshot contains 38 JSON files in `JSONs/raw/` and 38
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
- uses OpenAI service tier `flex`
- sends up to 5 prediction requests in parallel
- writes `docs/<run-name>/benchmark_results.json`
- writes `docs/<run-name>/benchmark_human_readable_report.html`
- caches raw predictions in `.cache/llm_predictions.json`
- merges cache updates safely across overlapping benchmark runs and persists each completed prediction immediately
- skips benchmark files whose top-level JSON payload is `[]`
- enables Langfuse logging when Langfuse keys are present

`<run-name>` is auto-generated from the model name plus a random readable suffix.
`--run-name` and `--langfuse-run-name` are aliases for the same value.

Useful examples:

```bash
uv run --env-file .env foresight-phys --max-files 2
uv run --env-file .env foresight-phys --json-dir JSONs/filtered --max-workers 8
uv run --env-file .env foresight-phys --model gpt-5.4-nano
uv run --env-file .env foresight-phys --ablation name-only
uv run --env-file .env foresight-phys --service-tier priority
uv run --env-file .env foresight-phys --run-name paper-benchmark-run-01
uv run --env-file .env foresight-phys --disable-langfuse
uv run --env-file .env foresight-phys --cache-path .cache/custom_predictions.json
uv run --env-file .env foresight-phys --disable-cache
uv run --env-file .env foresight-phys --cache-only
uv run --env-file .env foresight-phys --cache-ignore-system-prompt
uv run --env-file .env foresight-phys --output docs/custom-run/benchmark_results.json
uv run --env-file .env foresight-phys --html-output docs/custom-run/benchmark_human_readable_report.html
uv run --env-file .env foresight-phys --html-output ""
```

Setting `--html-output ""` disables HTML report generation.

## Benchmark workflow

For each JSON file, the benchmark pipeline:

1. Loads the ground-truth experiment JSON and skips files whose top-level payload is `[]`.
2. Replaces every `experiment_results.*.result` value with `"TO_PREDICT"`.
3. Splits the masked payload into one LLM request per experiment and sends each
	 experiment with the system prompt from `src/foresight_phys/system_prompt.txt`
	 to the OpenAI Responses API using a Pydantic structured-output schema.
4. Runs requests in parallel with retry and exponential backoff.
5. Reuses cached experiment predictions when available and persists each new prediction to the shared cache as soon as it completes.
   `--cache-only` disables OpenAI calls entirely and raises on any missing cached prediction or formula judgment.
6. Compares predicted results against the reference JSON, using an LLM judge for formula equivalence when needed.
7. Writes a machine-readable JSON summary, an offline HTML report, and refreshes `docs/index.html` as a run index.

## Forecasts with uncertainty

Predictions must include calibrated uncertainty per field. The structured
prediction schema is not identical to the benchmark input schema:

- `float` / `integer`: `distribution` (`normal` or `log_normal`) plus `p10`, `p50`, and `p90`.
	For `log_normal`, these quantiles must stay strictly positive. The benchmark
	derives `sigma` from the `p10`/`p90` span during scoring and stores `result = p50`
	in normalized outputs for downstream display.
- `bool`: `result` plus `prob_true ∈ [0, 1]`.
- `categorical`: `probabilities` — a list of `{value, probability}` covering
	every entry in `allowed_categorial_values`, summing to ≈1, plus a best-guess `result`.
- `formula`: `result` plus `confidence ∈ [0, 1]`.

By default, the prediction cache key includes the model, system prompt, masked
payload, and response schema, so changing any of those invalidates
`.cache/llm_predictions.json` and the cache will be repopulated on the next run.
Use `--cache-ignore-system-prompt` to reuse cache entries across different
system prompts by omitting the prompt from the cache key. When older cache
entries only exist in prior benchmark reports, the benchmark backfills those
prompt-agnostic entries from `docs/*/benchmark_results.json`.

## Metrics

Predictions are scored with proper scoring rules. Per file, the benchmark
computes:

- `numeric_crps_scaled`: **primary numeric metric** — mean of capped
  normalized CRPS over numeric predictions. Lower is better; 0 is a Dirac
  at the truth. For `normal` targets this is `CRPS / |y|` (only defined
  when `y ≠ 0`); for `log_normal` it equals the raw dex-CRPS (already
  unitless). Capped at 3 so a single catastrophic field cannot dominate
  the average. CRPS is a proper scoring rule, so it penalises both
  miscalibration and over-spread.
- `numeric_crps`: mean raw CRPS in the units of the predicted variable
  (linear for `normal`, dex for `log_normal`), capped at 30. Not
  comparable across experiments with different scales — use
  `numeric_crps_scaled` for cross-experiment aggregation.
  `CRPS = sigma * [z * (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)]`.
- `numeric_quality`: mean of per-field numeric quality, where
  `quality = 1 - crps_scaled / 3` is `numeric_crps_scaled` mapped onto
  `[0, 1]` (higher is better; 1 is a Dirac at the truth). Linear in
  CRPS, so it preserves the ranking induced by the proper score.
  Undefined (and excluded from the mean) when `crps_scaled` is undefined.
- `prediction_quality`: mean of per-field quality across all result fields.
  Quality is bounded in [0, 1] (`1 - crps_scaled / 3` for numeric,
  `1 - brier` for bool / formula, `1 - ½·brier` for categorical).
- `coverage_1sigma`, `coverage_2sigma`: fraction of numeric predictions with
  `|z| < 1` and `|z| < 2`. With well-calibrated uncertainty these target
  ≈0.68 and ≈0.95.
- `bool_brier`, `categorical_brier`, `formula_brier`: mean Brier score of the
  predicted probability against the realised outcome. Bounded in [0, 1] for
  bool / formula and [0, 2] for categorical. Brier is a proper scoring rule
  that, unlike log-loss, stays finite under catastrophic overconfidence.
- `bool_categorical_accuracy`: argmax accuracy over boolean and categorical
  fields (sanity check; not a proper score).
- `formula_accuracy`: fraction of formula fields judged equivalent.

Each per-file row in the summary JSON also includes counts such as
`result_count`, `numeric_count`, `bool_count`, `categorical_count`,
`formula_count`, and `missing_predictions`.

## Reports and outputs

`benchmark_results.json` contains:

- run metadata (`run_name`, `model`, `service_tier`, `max_workers`)
- aggregate metrics across files
- formula judge model name
- cache metadata
- Langfuse metadata
- a `per_file` list with one metrics row per JSON file
- per-file `expected_output`, `actual_output`, and `report_experiments` payloads used by the HTML report and analysis pipeline

The HTML report is a static offline file with:

- a left-hand paper switcher
- per-paper metric chips
- one table per experiment showing ground truth, prediction, and match status

`docs/index.html` is also regenerated on each benchmark run and serves as a
local index of discovered runs under `docs/`.

## Analyze benchmark runs

Use the analysis CLI to parse all saved benchmark runs under `docs/`, build a
cross-run dataset, compute model-level summaries, and render plots:

```bash
uv run --env-file .env foresight-phys-analysis all
```

Individual steps are also available:

```bash
uv run --env-file .env foresight-phys-analysis parse
uv run --env-file .env foresight-phys-analysis build
uv run --env-file .env foresight-phys-analysis analyze
uv run --env-file .env foresight-phys-analysis plots
```

The analysis pipeline reads `docs/<run-name>/benchmark_results.json` files and
writes:

- `.cache/analysis/predictions.parquet`
- `.cache/analysis/scored.parquet`
- `.cache/analysis/per_field.parquet`
- `docs/analysis/summary.json`
- `docs/analysis/plots/*.pdf`

The current checked-in `docs/analysis/plots/` directory contains figures such
as aggregate quality bars, numeric calibration plots, reliability diagrams, and
difficulty distributions.

## Interpretability analyses

Aggregate quality alone cannot say whether models predict the *crucial* variables
or just trivial/secondary ones, nor whether they are *useful for deciding which
experiments to run*. These analyses add those axes. They build on a per-paper
annotation pass and are surfaced in `docs/analysis/summary.json`.

### Annotate papers

```bash
uv run --env-file .env foresight-phys-annotate
```

For each non-empty filtered paper this writes `JSONs/annotations/<arXiv id>.json`
with, per result field: `centrality`
(`headline`/`key_supporting`/`secondary`/`setup_or_control`), `is_headline`,
`ex_ante_surprise` (`implied_or_derivable`/`uncertain`/`surprising`), and
`leakage_sufficient` (is the description alone enough to answer it?). It also
emits `comparison_sets` — swept series or baseline/intervention contrasts — each
with an `ordering_variable` and a `question_type`
(`argmax_select`/`monotonic_direction`/`sign_vs_baseline`/`order_of_magnitude`).
A code-side `appears_in_abstract` proxy is attached as an objective, reproducible
signal: it heads the centrality scale as the top tier (the LLM labels apply only
beneath it) and independently corroborates the headline labels. Annotation runs
once and is consumed by `analyze`.

### What the analysis adds

Running `foresight-phys-analysis all` (after annotation) augments
`summary.json` with:

- `decisions`: decision-usefulness from the comparison sets — `selection_accuracy`
  and normalised `mean_regret`/`decision_value` (vs a no-model pick) for
  `argmax_select`, `direction_accuracy` and a proper `sign_brier` for
  direction/sign questions, and order-of-magnitude coverage. `decisions.parquet`
  holds the per-set rows.
- `headline`, `quality_by_centrality`, `quality_by_surprise`: quality restricted
  to crucial / by-importance / by-surprise fields (paper-macro, comparable to the
  headline number).
- `foresight_lift`: full-context minus name-only quality per base model. Produce
  the name-only run with `foresight-phys --ablation name-only` (a distinct
  `<model> [name-only]` run); the lift is the predictive value the experiment
  description adds over generic typed-key priors.
- `leakage`: a leakage-excluded "clean" aggregate using the annotation's
  `leakage_sufficient` flag (`detect_leakage.py` is a complementary literal +
  normalized-numeric audit).
- `dataset_composition` and `dedup_aggregate`: field-count composition by
  type/centrality/surprise and fields-per-paper skew, plus a quality recomputed
  with each comparison set collapsed to one trend vote.

Numeric scoring also records per-field order-of-magnitude bands
(`abs_log10_error`, `within_factor_3`, `within_decade`) and the run aggregates
`oom_coverage_factor3` / `oom_coverage_decade`.

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
- uses OpenAI service tier `flex`
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
