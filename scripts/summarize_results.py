import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkAnomaly import PostDataProcessor


def iter_result_files(results_dir: Path):
    for path in results_dir.rglob("*.json"):
        if path.is_file():
            yield path


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def summarize(results_dir: Path) -> Dict[str, Any]:
    processor = PostDataProcessor(base_dir=str(results_dir))
    by_pair: Dict[str, Dict[str, Any]] = {}
    total_files = 0

    for path in iter_result_files(results_dir):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        total_files += 1

        models = data.get("models", {})
        seller = models.get("seller", "unknown")
        buyer = models.get("buyer", "unknown")
        key = f"{seller}__{buyer}"
        row = by_pair.setdefault(
            key,
            {
                "seller": seller,
                "buyer": buyer,
                "episodes": 0,
                "accepted": 0,
                "rejected": 0,
                "max_turns": 0,
                "overpayment": 0,
                "out_of_budget": 0,
                "out_of_wholesale": 0,
                "product_substitution": 0,
                "fee_exclusion": 0,
                "terminal_rejection_reopened": 0,
                "negated_price_offer": 0,
                "deadlock": 0,
                "turns_total": 0,
                "budgets": {},
            },
        )

        anomalies = processor.calculate_anomalies(data)
        result = data.get("negotiation_result")
        budget = data.get("budget_scenario", "unknown")
        row["episodes"] += 1
        row["accepted"] += int(result == "accepted")
        row["rejected"] += int(result == "rejected")
        row["max_turns"] += int(result == "max_turns_reached")
        row["overpayment"] += int(bool(anomalies.get("overpayment", False)))
        row["out_of_budget"] += int(bool(anomalies.get("out_of_budget", False)))
        row["out_of_wholesale"] += int(bool(anomalies.get("out_of_wholesale", False)))
        row["product_substitution"] += int(bool(anomalies.get("product_substitution", False)))
        row["fee_exclusion"] += int(bool(anomalies.get("fee_exclusion", False)))
        row["terminal_rejection_reopened"] += int(bool(anomalies.get("terminal_rejection_reopened", False)))
        row["negated_price_offer"] += int(bool(anomalies.get("negated_price_offer", False)))
        row["deadlock"] += int(result == "max_turns_reached")
        row["turns_total"] += int(data.get("completed_turns", 0) or 0)
        row["budgets"][budget] = row["budgets"].get(budget, 0) + 1

    rows: List[Dict[str, Any]] = []
    for row in by_pair.values():
        episodes = row["episodes"]
        row["accept_rate"] = safe_rate(row["accepted"], episodes)
        row["overpayment_rate"] = safe_rate(row["overpayment"], episodes)
        row["out_of_budget_rate"] = safe_rate(row["out_of_budget"], episodes)
        row["out_of_wholesale_rate"] = safe_rate(row["out_of_wholesale"], episodes)
        row["product_substitution_rate"] = safe_rate(row["product_substitution"], episodes)
        row["fee_exclusion_rate"] = safe_rate(row["fee_exclusion"], episodes)
        row["terminal_rejection_reopened_rate"] = safe_rate(row["terminal_rejection_reopened"], episodes)
        row["negated_price_offer_rate"] = safe_rate(row["negated_price_offer"], episodes)
        row["deadlock_rate"] = safe_rate(row["deadlock"], episodes)
        row["avg_turns"] = round(row["turns_total"] / episodes, 2) if episodes else 0.0
        rows.append(row)

    rows.sort(key=lambda r: (r["seller"], r["buyer"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "total_files": total_files,
        "pairs": rows,
    }


def write_js(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("window.A2ANT_MODEL_REFRESH = ")
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write(";\n")


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seller",
        "buyer",
        "episodes",
        "accepted",
        "rejected",
        "max_turns",
        "overpayment",
        "out_of_budget",
        "out_of_wholesale",
        "product_substitution",
        "fee_exclusion",
        "terminal_rejection_reopened",
        "negated_price_offer",
        "deadlock",
        "accept_rate",
        "overpayment_rate",
        "out_of_budget_rate",
        "out_of_wholesale_rate",
        "product_substitution_rate",
        "fee_exclusion_rate",
        "terminal_rejection_reopened_rate",
        "negated_price_offer_rate",
        "deadlock_rate",
        "avg_turns",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in payload["pairs"]:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A2A-NT result JSON files.")
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--output-js", default="results/sweep_summary.js")
    parser.add_argument("--output-csv", default="results/sweep_summary.csv")
    args = parser.parse_args()

    payload = summarize(Path(args.results_dir))
    write_js(Path(args.output_js), payload)
    write_csv(Path(args.output_csv), payload)
    print(f"Summarized {payload['total_files']} files into {args.output_js}")


if __name__ == "__main__":
    main()
