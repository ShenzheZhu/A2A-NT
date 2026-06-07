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

Model calls are routed through LiteLLM. For OpenRouter-backed models, put the key in `.env`:

```bash
OPENROUTER_API_KEY=...
```

`Config.py` is still supported as a local fallback for older workflows, but should not be committed.

## Running Experiments

An experiment is defined by a product file, buyer model, seller model, summary model, budget scenarios, turn limit, and output directory. The runner accepts explicit LiteLLM/OpenRouter model IDs, so the same code can be used for small smoke tests, focused model comparisons, or full buyer-seller grids.

Run a single dry-run smoke test:

```bash
python3 main.py \
  --products-file dataset/products_mini.json \
  --buyer-model openrouter/openai/gpt-4o-mini \
  --seller-model openrouter/openai/gpt-4o-mini \
  --summary-model openrouter/openai/gpt-4o-mini \
  --budget-scenarios wholesale,mid,retail \
  --product-limit 1 \
  --dry-run
```

Build a product subset when a run should use only part of the released product catalog:

```bash
python3 scripts/build_product_subset.py
```

Inspect a configured sweep without calling model APIs:

```bash
python3 scripts/validate_openrouter_models.py
python3 scripts/run_sweep.py --dry-run
```

Run a configured sweep:

```bash
bash scripts/run_sweep.sh
```

Useful commands:

```bash
python3 scripts/run_sweep.py --config configs/sweep_example.json --max-pairs 2 --product-limit 1 --dry-run
python3 scripts/run_sweep.py --config configs/sweep_example.json --mode all --dry-run
```

## Budget Scenarios

Supported scenarios are:

- `low`: `0.8 * wholesale`
- `wholesale`: `wholesale`
- `mid`: `(retail + wholesale) / 2`
- `retail`: `retail`
- `high`: `1.2 * retail`

Use `--budget-scenarios` to choose which scenarios to include in a run. For example, `wholesale,mid,retail` evaluates feasible market settings, while `low` and `high` can be used for targeted stress tests.

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
python3 scripts/summarize_results.py --results-dir results/sweep
```

## Project Structure

```text
.
├── main.py                         # Experiment runner
├── experiment_utils.py             # Shared runner utilities
├── Conversation.py                 # Negotiation flow
├── LanguageModel.py                # LiteLLM model gate
├── MarkAnomaly.py                  # Post-run anomaly labeling
├── configs/sweep_example.json      # Example sweep configuration
├── dataset/
│   ├── products.json
│   ├── products_mini.json
│   └── products_consumer_electronics.json
├── scripts/
│   ├── build_product_subset.py
│   ├── run_sweep.py
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
