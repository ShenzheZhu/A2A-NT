# Repository Guidelines

## Project Structure & Module Organization
Core experiment code lives at the repository root. Use [`main.py`](/home/zhush/Research/stanford/A2A/main.py) as the entrypoint, [`Conversation.py`](/home/zhush/Research/stanford/A2A/Conversation.py) for negotiation flow, [`LanguageModel.py`](/home/zhush/Research/stanford/A2A/LanguageModel.py) for provider/API integration, and [`MarkAnomaly.py`](/home/zhush/Research/stanford/A2A/MarkAnomaly.py) for post-run anomaly labeling. RL prompt-optimization code is isolated in [`rl/`](/home/zhush/Research/stanford/A2A/rl). Datasets live in [`dataset/`](/home/zhush/Research/stanford/A2A/dataset), notebooks in [`data_postprocess/`](/home/zhush/Research/stanford/A2A/data_postprocess), static figures in [`asset/`](/home/zhush/Research/stanford/A2A/asset), and paper templates/drafts in `Template/` plus the numbered manuscript directories. Treat `results/`, `logs/`, and `artifacts/` as generated output.

## Build, Test, and Development Commands
Create a Python 3.9 environment, then run `pip install -r requirements.txt`. Main workflows:

- `python main.py --products-file dataset/products_mini.json --buyer-model gpt-4o-mini --seller-model gpt-4o-mini --summary-model gpt-4o-mini --num-experiments 1 --max-turns 10 --output-dir results`
  Runs a small end-to-end negotiation smoke test.
- `bash run_all.sh`
  Launches the full model-grid experiment sweep and then runs anomaly marking.
- `bash train.sh`
  Trains the RL prompt bandit and writes logs/artifacts under `logs/` and `artifacts/`.
- `bash eval.sh`
  Evaluates the saved RL policy across seller models.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, simple module-level functions, and concise docstrings where behavior is non-obvious. Preserve current filenames for public modules even though legacy files use `CamelCase.py`; prefer `snake_case` for new functions, variables, CLI flags, and internal helpers. No formatter or linter is configured, so keep imports tidy and changes minimal.

## Testing Guidelines
There is no formal `pytest` suite in this repository. Validate changes with a targeted smoke run against `dataset/products_mini.json`, then inspect generated JSON, `logs/`, and RL artifacts for regressions. For notebook or analysis edits, re-run only the affected cells and confirm output paths still match the documented directory structure.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Update README.md`, but many older commits are too terse. Prefer clear one-line summaries like `Add retry handling for Gemini responses`. PRs should state the experiment or bug being addressed, list any required config or dataset assumptions, and include representative output paths or screenshots for notebook/figure changes. Never commit API keys from `Config.py` or bulk-generated results unless the change is explicitly about curated artifacts.
