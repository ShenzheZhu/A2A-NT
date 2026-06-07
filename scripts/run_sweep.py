import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import load_products, parse_csv


def load_plan(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def enabled_models(entries):
    return [entry for entry in entries if entry.get("enabled", True)]


def disabled_models(entries):
    return [entry for entry in entries if not entry.get("enabled", True)]


def build_pairs(plan, mode):
    frontier = enabled_models(plan.get("frontier_models", []))
    bridge = enabled_models(plan.get("bridge_models", []))
    pairs = []

    if mode in {"frontier-grid", "all"}:
        for seller in frontier:
            for buyer in frontier:
                pairs.append((seller, buyer, "frontier-grid"))

    if mode in {"bridge", "all"}:
        for frontier_model in frontier:
            for bridge_model in bridge:
                pairs.append((frontier_model, bridge_model, "bridge-new-seller"))
                pairs.append((bridge_model, frontier_model, "bridge-new-buyer"))

    return pairs


def count_products(products_file):
    return len(load_products(products_file))


def run_pair(plan, seller, buyer, args):
    cmd = [
        sys.executable,
        "main.py",
        "--products-file",
        args.products_file or plan["products_file"],
        "--buyer-model",
        buyer["model"],
        "--seller-model",
        seller["model"],
        "--summary-model",
        args.summary_model or plan["summary_model"],
        "--max-turns",
        str(args.max_turns or plan["max_turns"]),
        "--num-experiments",
        str(args.num_experiments or plan["num_experiments"]),
        "--budget-scenarios",
        ",".join(args.budgets or plan["budgets"]),
        "--output-dir",
        args.output_dir or plan["output_dir"],
        "--append",
    ]
    if args.product_limit is not None:
        cmd.extend(["--product-limit", str(args.product_limit)])
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"\n=== {seller['label']} seller vs {buyer['label']} buyer ===")
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run an A2A-NT experiment sweep.")
    parser.add_argument("--config", default="configs/sweep_example.json")
    parser.add_argument("--mode", choices=["frontier-grid", "bridge", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--product-limit", type=int, default=None)
    parser.add_argument("--products-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--budgets", default=None, help="Comma-separated budget scenario override")
    parser.add_argument("--skip-postprocess", action="store_true", help="Do not run result postprocessing after the sweep")
    parser.add_argument("--postprocess-move-errors", action="store_true", help="Move error files during postprocessing")
    parser.add_argument("--repair-price-scale", action="store_true", help="Apply price-scale repair suggestions during postprocessing")
    args = parser.parse_args()

    if args.budgets:
        args.budgets = parse_csv(args.budgets)

    plan = load_plan(args.config)
    products_file = args.products_file or plan["products_file"]
    product_count = args.product_limit or count_products(products_file)
    budgets = args.budgets or plan["budgets"]
    num_experiments = args.num_experiments or plan["num_experiments"]
    pairs = build_pairs(plan, args.mode)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    for entry in disabled_models(plan.get("bridge_models", [])):
        print(f"[skip disabled bridge] {entry['label']}: {entry.get('note', 'disabled')}")

    cells = len(pairs) * product_count * len(budgets) * num_experiments
    print(f"Plan: {plan['name']}")
    print(f"Mode: {args.mode}")
    print(f"Pairs: {len(pairs)} | products: {product_count} | budgets: {len(budgets)} | experiments: {num_experiments}")
    print(f"Conversation cells: {cells}")

    for seller, buyer, _kind in pairs:
        run_pair(plan, seller, buyer, args)

    if not args.dry_run and not args.skip_postprocess:
        from MarkAnomaly import run_postprocess

        run_postprocess(
            base_dir=args.output_dir or plan["output_dir"],
            move_error_files=args.postprocess_move_errors,
            repair_price_scale=args.repair_price_scale,
        )


if __name__ == "__main__":
    main()
