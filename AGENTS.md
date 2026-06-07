# Repository Guidelines

## Project Structure & Module Organization
Core experiment code lives at the repository root. Use [`main.py`](main.py) as the entrypoint, [`Conversation.py`](Conversation.py) for negotiation flow, [`LanguageModel.py`](LanguageModel.py) for LiteLLM provider integration, and [`MarkAnomaly.py`](MarkAnomaly.py) for post-run anomaly labeling. Current refresh settings live in [`configs/model_refresh.json`](configs/model_refresh.json). Datasets live in [`dataset/`](dataset), notebooks in [`data_postprocess/`](data_postprocess), static figures in [`asset/`](asset), and utility scripts in [`scripts/`](scripts). Treat `results/`, `logs/`, and `artifacts/` as generated output.

## Build, Test, and Development Commands
Create a Python 3.9 environment, then run `pip install -r requirements.txt`. Main workflows:

- `python main.py --products-file dataset/products_consumer_electronics.json --buyer-model qwen/qwen3.7-max --seller-model openai/gpt-5.5 --summary-model openai/gpt-5.4-mini --num-experiments 1 --max-turns 10 --budget-scenarios wholesale,mid,retail --product-limit 1 --dry-run`
  Runs a small end-to-end negotiation smoke test.
- `python scripts/run_model_refresh.py --dry-run`
  Prints the configured frontier/bridge sweep without calling model APIs.
- `python scripts/validate_openrouter_models.py`
  Checks configured model IDs against the current OpenRouter model API.
- `bash scripts/run_model_refresh.sh`
  Launches the configured model-refresh sweep through `configs/model_refresh.json`.
- `bash run_all.sh`
  Launches the legacy full model-grid experiment sweep and then runs anomaly marking.
- `bash train.sh`
  Trains the RL prompt bandit and writes logs/artifacts under `logs/` and `artifacts/`.
- `bash eval.sh`
  Evaluates the saved RL policy across seller models.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, simple module-level functions, and concise docstrings where behavior is non-obvious. Preserve current filenames for public modules even though legacy files use `CamelCase.py`; prefer `snake_case` for new functions, variables, CLI flags, and internal helpers. No formatter or linter is configured, so keep imports tidy and changes minimal.

## Testing Guidelines
There is no formal `pytest` suite in this repository. Validate runner changes with `python scripts/run_model_refresh.py --dry-run` and a targeted dry run against `dataset/products_consumer_electronics.json`; for live provider changes, run one product and one budget before starting a sweep. For notebook or analysis edits, re-run only the affected cells and confirm output paths still match the documented directory structure.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Update README.md`, but many older commits are too terse. Prefer clear one-line summaries like `Add retry handling for Gemini responses`. PRs should state the experiment or bug being addressed, list any required config or dataset assumptions, and include representative output paths or screenshots for notebook/figure changes. Never commit API keys from `.env` / `Config.py` or bulk-generated results unless the change is explicitly about curated artifacts.
