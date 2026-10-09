# ai_ingeneering_ch

An async LLM manager that puts OpenAI, Anthropic and Gemini behind one interface.
You describe the provider and model in an `LLMConfig`, and `AsyncLLMManager` gives
you the same two methods for all of them: `generate` (full response) and
`generate_stream` (token by token). API errors are caught and returned in the
response instead of crashing the program, and a manager can be given a fallback
config (Gemini in the test script) that answers when the main provider fails.

## Requirements

- Python 3.12 or newer (`.python-version` pins 3.12, so `uv` picks it automatically)
- [uv](https://docs.astral.sh/uv/)

## Setup

1. Install the dependencies:

   ```bash
   uv sync
   ```

2. Create your local environment file from the template:

   ```bash
   cp .env.local.template .env.local
   ```

3. Fill in your API keys in `.env.local`:

   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   ```

`.env.local` is ignored by git. If any of the three keys is missing or empty, the
program asks for it in the terminal on every run.

## Running

Run the commands from the project root. The entry point is a module, so use
`python -m backend.main`; `python backend/main.py` fails with
`ModuleNotFoundError: No module named 'backend'`.

```bash
uv run python -m backend.main <prueba>
```

| `<prueba>`    | What it does                                                                        | Keys it uses                 |
| ------------- | ----------------------------------------------------------------------------------- | ---------------------------- |
| `normal`      | Sends one question to Anthropic and prints the full response.                       | Anthropic (Google if it falls back) |
| `streaming`   | Streams the answer from OpenAI, then from Anthropic, printing as it arrives.        | OpenAI, Anthropic (Google if it falls back) |
| `resiliencia` | Calls OpenAI with an invalid key on purpose, no fallback: shows the error was caught. | None (uses a fake)           |
| `fallback`    | Calls OpenAI with an invalid key on purpose, with Gemini as fallback: Gemini answers. | Google                       |

Examples:

```bash
uv run python -m backend.main normal
```

```bash
uv run python -m backend.main streaming
```

```bash
uv run python -m backend.main resiliencia
```

```bash
uv run python -m backend.main fallback
```

The tests use `gpt-4o-mini` for OpenAI, `claude-sonnet-4-6` for Anthropic and
`gemini-3.5-flash-lite` for Gemini. All three are set in `backend/main.py`.

## Fallback

Pass a second `LLMConfig` as `fallback` and the manager uses it whenever the main
provider returns an error:

```python
manager = AsyncLLMManager(config_openai, fallback=config_gemini)
resultado = await manager.generate(pregunta)
```

- `generate`: if the main provider fails, the result comes from the fallback.
  `resultado.provider` tells you who answered and `resultado.fallback_reason`
  holds the original error. If the fallback fails too, `resultado.error` holds
  its error and `fallback_reason` still holds the first one.
- `generate_stream`: if the main provider fails before sending any text, the
  stream silently continues from the fallback. If it fails mid-answer, a warning
  line is yielded and the fallback's answer streams from the beginning.
- Each fallback is also logged as a `WARNING` through the standard `logging`
  module.

Without a `fallback`, errors are returned as before: in `resultado.error` for
`generate`, and as a final `[⚠️ ...]` chunk for `generate_stream`.

## Project structure

```
backend/
├── main.py          # entry point: picks the test to run from the argument
├── settings.py      # loads .env.local and asks for any missing key
├── schemas.py       # Provider, ChatMessage, LLMConfig, ModelResponse
├── base.py          # BaseLLMClient: the contract every provider client implements
├── llm_manager.py   # AsyncLLMManager: builds the right client from an LLMConfig
└── clients/
    ├── openai.py
    ├── anthropic.py
    └── gemini.py
```

## Development

Linting and formatting use [ruff](https://docs.astral.sh/ruff/), run automatically
on every commit through pre-commit.

Install the git hook once:

```bash
uv run pre-commit install
```

Run the checks by hand:

```bash
uv run ruff check --fix .
```

```bash
uv run ruff format .
```
