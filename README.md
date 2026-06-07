# The Automated but Risky Game: Modeling and Benchmarking Agent-to-Agent Negotiations and Transactions in Consumer Markets

[Shenzhe Zhu](https://shenzhezhu.github.io) $^{1}$, [Jiao Sun](https://sunjiao123sun.github.io/) $^{2}$, Yi Nian $^{3}$, [Tobin South](https://tobin.page/) $^4$, [Alex Pentland](https://www.media.mit.edu/people/sandy/overview/) $^{4,5}$, [Jiaxin Pei](https://jiaxin-pei.github.io/) $^{5,\dagger}$

$^1$ University of Toronto, $^2$ Google DeepMind, $^3$ University of Southern California
$^4$ Massachusetts Institute of Technology, $^5$ Stanford University
($^{\dagger}$ Corresponding Author)

[Project Page](https://shenzhezhu.github.io/A2A-NT/) | [arXiv](https://arxiv.org/abs/2506.00073) | [Dataset](https://huggingface.co/datasets/Chouoftears/Agent2Agent-Negotiation-in-Consumer-Setting-Dataset)

![teaser](asset/teaser.png)

## Overview

A2A-NT simulates consumer-market negotiations where both buyers and sellers delegate price negotiation to LLM agents. The benchmark records full dialogue traces, extracted seller offers, final deal outcomes, budget and wholesale constraints, and post-run anomaly labels.

## Setup

Create a Python 3.9 environment and install dependencies:

```bash
pip install -r requirements.txt
```

For the current model refresh, calls are routed through LiteLLM and OpenRouter. Put the key in `.env`:

```bash
OPENROUTER_API_KEY=...
```

`Config.py` is still supported as a local fallback for older workflows, but should not be committed.

## Current Refresh Design

The experiment refresh uses a smaller, cleaner design before any expensive full sweep:

- Products: all 30 consumer electronics and appliance records from `dataset/products.json` (`id=41..70`), written to `dataset/products_consumer_electronics.json`.
- Main budgets: `wholesale`, `mid`, and `retail`.
- Frontier leaderboard: full 6 x 6 buyer-seller grid.
- Bridge comparison: selected legacy Qwen models only, instead of pairing every new model against every old model.

Build or refresh the product subset:

```bash
python3 scripts/build_product_subset.py
```

Inspect the planned run without calling model APIs:

```bash
python3 scripts/validate_openrouter_models.py
python3 scripts/run_model_refresh.py --dry-run
```

Run the configured sweep:

```bash
bash scripts/run_model_refresh.sh
```

Useful overrides:

```bash
python3 scripts/run_model_refresh.py --mode frontier-grid --product-limit 2 --dry-run
python3 scripts/run_model_refresh.py --mode bridge --dry-run
python3 main.py --products-file dataset/products_consumer_electronics.json --buyer-model qwen/qwen3.7-max --seller-model openai/gpt-5.5 --summary-model openai/gpt-5.4-mini --budget-scenarios wholesale,mid,retail --product-limit 1 --dry-run
```

The current OpenRouter model API lists `qwen/qwen-2.5-7b-instruct`, but not a standard `qwen2.5-14b-instruct` route. The 14B bridge entry is kept disabled in `configs/model_refresh.json` until a live provider or exact OpenRouter ID is confirmed.

## Budget Scenarios

Supported scenarios are:

- `low`: `0.8 * wholesale`
- `wholesale`: `wholesale`
- `mid`: `(retail + wholesale) / 2`
- `retail`: `retail`
- `high`: `1.2 * retail`

The main refresh uses `wholesale,mid,retail`. `low` and `high` are reserved for targeted risk audits rather than the main leaderboard aggregate.

## Results

Results are saved under:

```text
results/
└── seller_{seller_model}/
    └── {buyer_model}/
        └── product_{product_id}/
            └── budget_{scenario}/
                └── product_{product_id}_exp_{experiment_num}.json
```

Each result file contains the conversation history, extracted seller offers, negotiation outcome, budget scenario, and model metadata.

Summarize a run:

```bash
python3 scripts/summarize_results.py --results-dir results/model_refresh_2026
```

## Project Structure

```text
.
├── main.py                         # Experiment runner
├── Conversation.py                 # Negotiation flow
├── LanguageModel.py                # LiteLLM model gate
├── MarkAnomaly.py                  # Post-run anomaly labeling
├── configs/model_refresh.json      # Current model-refresh plan
├── dataset/
│   ├── products.json
│   ├── products_mini.json
│   └── products_consumer_electronics.json
├── scripts/
│   ├── build_product_subset.py
│   ├── run_model_refresh.py
│   ├── summarize_results.py
│   └── validate_openrouter_models.py
└── data_postprocess/               # Analysis notebooks and plotting scripts
```

Treat `results/`, `logs/`, and `artifacts/` as generated output.

## Citation

```bibtex
@misc{zhu2025automatedriskygamemodeling,
      title={The Automated but Risky Game: Modeling and Benchmarking Agent-to-Agent Negotiations and Transactions in Consumer Markets},
      author={Shenzhe Zhu and Jiao Sun and Yi Nian and Tobin South and Alex Pentland and Jiaxin Pei},
      year={2025},
      eprint={2506.00073},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.00073},
}
```
