import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkAnomaly import PostDataProcessor
from experiment_utils import parse_price
from scripts.summarize_results import final_offer, result_metrics


BUDGET_ORDER = ["low", "wholesale", "mid", "retail", "high"]
BUDGET_LABELS = ["Low", "Wholesale", "Mid", "Retail", "High"]
PRR_BUDGET_LABELS = ["low", "wholesale", "mid", "retail", "high"]
MODEL_SHORT_LABELS = {
    "openai/gpt-5.5": "GPT-5.5",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "qwen/qwen3.7-max": "Qwen3.7 Max",
    "x-ai/grok-4.20": "Grok 4.20",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "qwen/qwen-2.5-7b-instruct": "Qwen2.5-7B",
}
PALETTE = [
    "#4C78A8",
    "#72B7B2",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#E15759",
    "#76B7B2",
    "#9C755F",
    "#EDC948",
]

WHOLESALE_CMAP = LinearSegmentedColormap.from_list(
    "paper_wholesale",
    ["#c9e7db", "#91c9c5", "#6f99b7", "#625084", "#32133f"],
)
BUDGET_CMAP = LinearSegmentedColormap.from_list(
    "paper_budget",
    ["#f3bb82", "#ee8368", "#d43d69", "#81306f", "#1e1835"],
)
OVERPAYMENT_CMAP = LinearSegmentedColormap.from_list(
    "paper_overpayment",
    ["#dfe8c7", "#a9d0ae", "#62aaa5", "#5c4f7c", "#29123d"],
)
STALL_CMAP = LinearSegmentedColormap.from_list(
    "paper_stall",
    ["#eee5c8", "#d6b29e", "#bd7596", "#7a2c78", "#211334"],
)
TOP_LABEL_POSITIONS = {
    "GPT-5.5": (20.32, 52.05),
    "Claude Sonnet 4.6": (16.45, 53.75),
    "Gemini 3.5 Flash": (17.82, 61.0),
    "DeepSeek V4 Pro": (14.72, 59.0),
    "DeepSeek V4 Flash": (14.55, 56.35),
    "Qwen3.7 Max": (17.82, 54.35),
    "Grok 4.20": (20.55, 59.25),
    "Claude Opus 4.8": (19.15, 56.9),
    "Qwen2.5-7B": (9.15, 80.25),
}
BOTTOM_LABEL_POSITIONS = {
    "GPT-5.5": (70.05, 14.45),
    "Claude Sonnet 4.6": (67.85, 12.75),
    "Gemini 3.5 Flash": (75.1, 11.25),
    "DeepSeek V4 Pro": (70.2, 11.95),
    "DeepSeek V4 Flash": (61.65, 13.15),
    "Qwen3.7 Max": (66.2, 14.2),
    "Grok 4.20": (55.0, 12.75),
    "Claude Opus 4.8": (72.55, 13.15),
    "Qwen2.5-7B": (74.65, 6.85),
}


def enabled_model_ids(config: Dict[str, Any]) -> List[str]:
    models = []
    seen = set()
    for group in ("frontier_models", "bridge_models"):
        for entry in config.get(group, []):
            if not entry.get("enabled", True):
                continue
            model_id = entry["model"]
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(model_id)
    return models


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_result_files(results_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in results_dir.rglob("*.json") if path.is_file())


def parse_product_price(data: Dict[str, Any], key: str) -> Optional[float]:
    try:
        return parse_price(data.get("product_data", {}).get(key))
    except (TypeError, ValueError):
        return None


def first_offer(data: Dict[str, Any]) -> Optional[float]:
    offers = data.get("seller_price_offers")
    if not isinstance(offers, list) or not offers:
        return None
    try:
        return float(offers[0])
    except (TypeError, ValueError):
        return None


def short_label(model_id: str) -> str:
    return MODEL_SHORT_LABELS.get(model_id, model_id)


def load_clean_records(
    results_dir: Path,
    model_ids: List[str],
    max_experiment_num_exclusive: Optional[int],
) -> List[Dict[str, Any]]:
    model_set = set(model_ids)
    processor = PostDataProcessor(base_dir=str(results_dir))
    records = []
    for path in iter_result_files(results_dir):
        data = load_json(path)
        if max_experiment_num_exclusive is not None:
            try:
                experiment_num = int(data.get("experiment_num", 0))
            except (TypeError, ValueError):
                experiment_num = 0
            if experiment_num >= max_experiment_num_exclusive:
                continue
        models = data.get("models", {})
        seller = models.get("seller")
        buyer = models.get("buyer")
        if seller not in model_set or buyer not in model_set:
            continue
        anomalies = processor.calculate_anomalies(data)
        if anomalies.get("system_data_error", False):
            continue
        metrics = result_metrics(data, anomalies)
        retail = parse_product_price(data, "Retail Price")
        wholesale = parse_product_price(data, "Wholesale Price")
        final = final_offer(data)
        first = first_offer(data)
        records.append(
            {
                "seller": seller,
                "buyer": buyer,
                "budget": data.get("budget_scenario", "unknown"),
                "accepted": data.get("negotiation_result") == "accepted",
                "clean_deal": bool(metrics.get("clean_deal")),
                "retail": retail,
                "wholesale": wholesale,
                "final": final,
                "first_offer": first,
                "flags": anomalies,
            }
        )
    return records


def mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def rate(num: int, den: int) -> float:
    return (num / den * 100.0) if den else 0.0


def budget_matrix(
    records: List[Dict[str, Any]],
    model_ids: List[str],
    owner: str,
    flag: str,
) -> np.ndarray:
    counts = {(model, budget): 0 for model in model_ids for budget in BUDGET_ORDER}
    denominators = {(model, budget): 0 for model in model_ids for budget in BUDGET_ORDER}
    for record in records:
        budget = record["budget"]
        if budget not in BUDGET_ORDER:
            continue
        if owner == "buyer":
            model = record["buyer"]
            denominators[(model, budget)] += 1
            if record["flags"].get(flag, False):
                counts[(model, budget)] += 1
        elif owner == "seller":
            model = record["seller"]
            denominators[(model, budget)] += 1
            if record["flags"].get(flag, False):
                counts[(model, budget)] += 1
        else:
            for model in (record["seller"], record["buyer"]):
                denominators[(model, budget)] += 1
                if record["flags"].get(flag, False):
                    counts[(model, budget)] += 1
    return np.array(
        [
            [rate(counts[(model, budget)], denominators[(model, budget)]) for budget in BUDGET_ORDER]
            for model in model_ids
        ]
    )


def overpayment_matrix(records: List[Dict[str, Any]], model_ids: List[str]) -> np.ndarray:
    matrix = budget_matrix(records, model_ids, "buyer", "overpayment")
    return matrix.T


def buyer_prr_by_budget(records: List[Dict[str, Any]], model_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    model_budget_values = {budget: [] for budget in BUDGET_ORDER}
    for budget in BUDGET_ORDER:
        for model in model_ids:
            values = []
            for record in records:
                if record["buyer"] != model or record["budget"] != budget or not record["accepted"]:
                    continue
                retail = record["retail"]
                final = record["final"]
                if retail and retail > 0 and final is not None:
                    values.append((retail - final) / retail * 100.0)
            if values:
                model_budget_values[budget].append(mean(values))
    means = np.array([mean(model_budget_values[budget]) for budget in BUDGET_ORDER])
    errors = []
    for budget in BUDGET_ORDER:
        values = model_budget_values[budget]
        if len(values) <= 1:
            errors.append(0.0)
        else:
            errors.append(float(np.std(values, ddof=1) / math.sqrt(len(values)) * 1.96))
    return means, np.array(errors)


def performance_rows(records: List[Dict[str, Any]], model_ids: List[str], baseline_model: str) -> List[Dict[str, Any]]:
    rows = []
    baseline_profit = None
    model_profit_values = {}
    for model in model_ids:
        buyer_prrs = []
        seller_prrs = []
        profit_rates = []
        profits = []
        seller_accepted = 0
        seller_episodes = 0
        for record in records:
            if record["buyer"] == model and record["clean_deal"]:
                retail = record["retail"]
                final = record["final"]
                if retail and retail > 0 and final is not None:
                    buyer_prrs.append((retail - final) / retail * 100.0)
            if record["seller"] == model:
                seller_episodes += 1
                seller_accepted += int(record["accepted"])
                if record["clean_deal"]:
                    retail = record["retail"]
                    wholesale = record["wholesale"]
                    final = record["final"]
                    if retail and wholesale is not None and retail > wholesale and final is not None:
                        seller_prrs.append((retail - final) / (retail - wholesale) * 100.0)
                    if retail and retail > 0 and wholesale is not None and final is not None:
                        profit_rates.append((final - wholesale) / retail * 100.0)
                    if wholesale is not None and final is not None:
                        profits.append(final - wholesale)
        avg_profit = mean(profits)
        model_profit_values[model] = avg_profit
        rows.append(
            {
                "model": model,
                "label": short_label(model),
                "buyer_prr": mean(buyer_prrs),
                "seller_prr": mean(seller_prrs),
                "deal_rate": rate(seller_accepted, seller_episodes),
                "avg_profit_rate": mean(profit_rates),
                "avg_profit": avg_profit,
            }
        )
    baseline_profit = model_profit_values.get(baseline_model)
    if not baseline_profit or math.isnan(baseline_profit):
        baseline_profit = next((value for value in model_profit_values.values() if value and not math.isnan(value)), None)
    for row in rows:
        row["relative_profit"] = (row["avg_profit"] / baseline_profit) if baseline_profit else 1.0
    return rows


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.edgecolor": "#777777",
            "grid.color": "#dddddd",
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_performance_scatter(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.9))
    paper_colors = sns.color_palette("viridis", len(rows))
    colors = {row["model"]: paper_colors[index] for index, row in enumerate(rows)}

    for row in rows:
        axes[0].scatter(row["buyer_prr"], row["seller_prr"], s=90, color=colors[row["model"]], alpha=0.9)
        label_x, label_y = TOP_LABEL_POSITIONS.get(row["label"], (row["buyer_prr"], row["seller_prr"]))
        axes[0].annotate(
            row["label"],
            (row["buyer_prr"], row["seller_prr"]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=8.5,
            ha="left",
            va="center",
        )
    axes[0].set_xlabel("Buyer Price Reduction Rate (%)")
    axes[0].set_ylabel("Seller Price Reduction Rate (%)")
    top_x = [row["buyer_prr"] for row in rows]
    top_y = [row["seller_prr"] for row in rows]
    axes[0].set_xlim(min(top_x) - 0.6, max(top_x) + 2.0)
    axes[0].set_ylim(max(top_y) + 2.2, min(top_y) - 1.4)
    axes[0].grid(True, alpha=0.8)

    rel_values = [row["relative_profit"] for row in rows if row["relative_profit"] and not math.isnan(row["relative_profit"])]
    rel_min = min(rel_values) if rel_values else 1.0
    rel_max = max(rel_values) if rel_values else 1.0
    for row in rows:
        rel = row["relative_profit"] if row["relative_profit"] and not math.isnan(row["relative_profit"]) else 1.0
        size = 70 + 210 * (rel - rel_min) / (rel_max - rel_min or 1)
        axes[1].scatter(row["deal_rate"], row["avg_profit_rate"], s=size, color=colors[row["model"]], alpha=0.85)
        label_x, label_y = BOTTOM_LABEL_POSITIONS.get(row["label"], (row["deal_rate"], row["avg_profit_rate"]))
        label = f"{row['label']}\n({rel:.1f}x)"
        axes[1].annotate(
            label,
            (row["deal_rate"], row["avg_profit_rate"]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=8.5,
            linespacing=0.9,
            ha="left",
            va="center",
        )
    axes[1].set_xlabel("Deal Rate (%)")
    axes[1].set_ylabel("Average Profit Rate (%)")
    axes[1].set_xlim(53.8, 78.5)
    axes[1].set_ylim(5.6, 14.85)
    axes[1].grid(True, alpha=0.8)

    if rel_values:
        handles = []
        for rel in [rel_min, (rel_min + rel_max) / 2, rel_max]:
            size = 70 + 210 * (rel - rel_min) / (rel_max - rel_min or 1)
            handles.append(axes[1].scatter([], [], s=size, color="#6b8fb3", alpha=0.75, label=f"{rel:.1f}x"))
        axes[1].legend(
            handles=handles,
            title="Relative Profit",
            loc="center left",
            bbox_to_anchor=(0.01, 0.52),
            frameon=True,
        )
    fig.tight_layout(h_pad=1.4)
    save_figure(fig, output_dir, "fig1_negotiation_performance_scatter")


def draw_constraint_heatmaps(records: List[Dict[str, Any]], model_ids: List[str], output_dir: Path) -> None:
    row_labels = [short_label(model) for model in model_ids]
    out_wholesale = budget_matrix(records, model_ids, "seller", "out_of_wholesale")
    out_budget = budget_matrix(records, model_ids, "buyer", "out_of_budget")
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 10.2))
    sns.heatmap(
        out_wholesale,
        ax=axes[0],
        cmap=WHOLESALE_CMAP,
        annot=True,
        fmt=".1f",
        xticklabels=BUDGET_LABELS,
        yticklabels=row_labels,
        cbar_kws={"label": "Out of Wholesale Rate (%)"},
        linewidths=0.2,
        linecolor="#eeeeee",
    )
    axes[0].set_title("Out of Wholesale Rate by Budget Type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.heatmap(
        out_budget,
        ax=axes[1],
        cmap=BUDGET_CMAP,
        annot=True,
        fmt=".1f",
        xticklabels=BUDGET_LABELS,
        yticklabels=row_labels,
        cbar_kws={"label": "Out of Budget Rate (%)"},
        linewidths=0.2,
        linecolor="#eeeeee",
    )
    axes[1].set_title("Out of Budget Rate by Budget Type")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    for axis in axes:
        axis.tick_params(axis="x", rotation=24)
        axis.tick_params(axis="y", rotation=18)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig2_constraint_heatmaps")


def draw_overpayment_deadlock_heatmaps(records: List[Dict[str, Any]], model_ids: List[str], output_dir: Path) -> None:
    labels = [short_label(model) for model in model_ids]
    overpayment = overpayment_matrix(records, model_ids)
    deadlock = budget_matrix(records, model_ids, "shared", "deadlock")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2))
    sns.heatmap(
        overpayment,
        ax=axes[0],
        cmap=OVERPAYMENT_CMAP,
        annot=True,
        fmt=".1f",
        xticklabels=labels,
        yticklabels=BUDGET_LABELS,
        cbar_kws={"label": "Overpayment Rate (%)"},
        linewidths=0.2,
        linecolor="#eeeeee",
    )
    axes[0].set_title("Overpayment Rate by Budget Type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.heatmap(
        deadlock,
        ax=axes[1],
        cmap=STALL_CMAP,
        annot=True,
        fmt=".1f",
        xticklabels=BUDGET_LABELS,
        yticklabels=labels,
        cbar_kws={"label": "Deadlock Rate (%)"},
        linewidths=0.2,
        linecolor="#eeeeee",
    )
    axes[1].set_title("Deadlock (Max Turn) Rate (%) by Model and Budget")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=32)
    axes[0].tick_params(axis="y", rotation=32)
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].tick_params(axis="y", rotation=18)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig3_overpayment_stall_heatmaps")


def draw_buyer_prr_line(records: List[Dict[str, Any]], model_ids: List[str], output_dir: Path) -> None:
    means, errors = buyer_prr_by_budget(records, model_ids)
    x = np.arange(len(BUDGET_ORDER))
    fig, axis = plt.subplots(figsize=(9.2, 4.2))
    color = "#5E2A7E"
    axis.plot(x, means, marker="o", color=color, linewidth=2.2, markersize=5)
    axis.fill_between(x, means - errors, means + errors, color=color, alpha=0.18)
    axis.set_xticks(x)
    axis.set_xticklabels(PRR_BUDGET_LABELS)
    axis.set_ylabel("Buyer Price Reduction Rate (%)")
    axis.set_title("Average Buyer Price Reduction Rate by Budget Setting")
    axis.grid(True, alpha=0.8)
    if not math.isnan(means[0]) and not math.isnan(means[3]):
        delta = means[3] - means[0]
        axis.annotate(
            "",
            xy=(3, means[3]),
            xytext=(0, means[0]),
            arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "red"},
        )
        axis.text(
            1.25,
            (means[0] + means[3]) / 2 + 2.0,
            f"{delta:+.1f} pp",
            color="red",
            fontsize=14,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    fig.tight_layout()
    save_figure(fig, output_dir, "fig4_buyer_prr_by_budget")


def write_manifest(output_dir: Path, args: argparse.Namespace, model_ids: List[str], records: List[Dict[str, Any]]) -> None:
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results_dir": str(args.results_dir),
        "summary_json": str(args.summary_json),
        "config": str(args.config),
        "max_experiment_num_exclusive": args.max_experiment_num_exclusive,
        "model_count": len(model_ids),
        "models": [{"model": model, "label": short_label(model)} for model in model_ids],
        "clean_analyzed_conversations": len(records),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.suffix in {".png", ".pdf"}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style A2A-NT talk figures from refreshed sweep data.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/full_sweep_20260608_clean_w8"))
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/analysis/full_sweep_20260608_unified_w16/summary.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/sweep_example.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "A2A-NT-talk-figures-20260609",
    )
    parser.add_argument("--max-experiment-num-exclusive", type=int, default=1)
    args = parser.parse_args()

    setup_style()
    config = load_json(args.config)
    model_ids = enabled_model_ids(config)
    records = load_clean_records(args.results_dir, model_ids, args.max_experiment_num_exclusive)
    if not records:
        raise RuntimeError(f"No clean records found under {args.results_dir}")
    rows = performance_rows(records, model_ids, config.get("relative_profit_baseline", model_ids[-1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_performance_scatter(rows, args.output_dir)
    draw_constraint_heatmaps(records, model_ids, args.output_dir)
    draw_overpayment_deadlock_heatmaps(records, model_ids, args.output_dir)
    draw_buyer_prr_line(records, model_ids, args.output_dir)
    write_manifest(args.output_dir, args, model_ids, records)
    print(f"Wrote talk figures to {args.output_dir}")
    for path in sorted(args.output_dir.iterdir()):
        if path.suffix in {".png", ".pdf"}:
            print(path)


if __name__ == "__main__":
    main()
