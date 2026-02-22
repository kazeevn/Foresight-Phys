# Foresight-Phys Benchmark

A benchmark for AI ability to predict the results of physical experiments 

## Data
The descriptions were extracted from recent open-access papers by Kostya Novoselov. The extraction was done with Gemini 3.1 Pro.

## Technical workflow

- Loads each JSON file in `JSONs/`.
- Replaces every `experiment_results.*.result` value with `"TO_PREDICT"`.
- Prompts the LLM with:
	- system prompt from `system_prompt.txt`
	- masked JSON as user input
- Runs OpenAI calls in parallel with retry/backoff using `tenacity`.
- Shows a live progress bar while files are being predicted.
- Runs DeepEval reporting by default for each benchmark run.
- Parses the model JSON output and compares predicted `result` values to ground truth.
- Computes per JSON file:
	- `mape` (Mean Absolute Percentage Error) for numeric predictions
	- `bool_categorical_accuracy` for boolean and categorical predictions
- Computes aggregate averages across files for both metrics.

## Setup (uv)

```bash
uv sync
```

Ensure `.env` contains your provider API keys (for OpenAI-compatible use, set `OPENAI_API_KEY`).

## Run benchmark

```bash
uv run python main.py
```

### Useful options

```bash
uv run python main.py --max-files 2
uv run python main.py --max-workers 8
uv run python main.py --disable-deepeval
uv run python main.py --output benchmark_results.json
```

The run writes summary output to `benchmark_results.json`.
