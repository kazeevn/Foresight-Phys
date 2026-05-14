Use `uv` for Python environment management and task execution:
 - Install the packages you need with `uv add`
 - Run code with `uv run --env-file .env`

When getting structured output from LLMs, use pydantic, not raw json schemas

When calling OpenAI, use the Responses API, not Completions