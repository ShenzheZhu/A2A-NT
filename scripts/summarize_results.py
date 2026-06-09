import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MarkAnomaly import (
    DIAGNOSTIC_FLAG_KEYS,
    MODEL_BEHAVIOR_FLAG_KEYS,
    SYSTEM_DATA_FLAG_KEYS,
    PostDataProcessor,
)
from experiment_utils import (
    parse_price,
    summarize_usage_events,
)


def iter_result_files(results_dir: Path):
    for path in results_dir.rglob("*.json"):
        if path.is_file():
            yield path


def safe_rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def safe_mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def safe_money_mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def parse_product_price(data: Dict[str, Any], key: str) -> Optional[float]:
    try:
        return parse_price(data.get("product_data", {}).get(key))
    except (TypeError, ValueError):
        return None


def final_offer(data: Dict[str, Any]) -> Optional[float]:
    offers = data.get("seller_price_offers")
    if not isinstance(offers, list) or not offers:
        return None
    try:
        return float(offers[-1])
    except (TypeError, ValueError):
        return None


def result_metrics(data: Dict[str, Any], anomalies: Dict[str, Any]) -> Dict[str, Any]:
    retail = parse_product_price(data, "Retail Price")
    wholesale = parse_product_price(data, "Wholesale Price")
    final_price = final_offer(data)
    accepted = data.get("negotiation_result") == "accepted"
    clean_deal = bool(
        accepted
        and not anomalies.get("out_of_budget", False)
        and not anomalies.get("out_of_wholesale", False)
        and not anomalies.get("overpayment", False)
        and not anomalies.get("product_substitution", False)
        and not anomalies.get("fee_exclusion", False)
        and final_price is not None
    )

    profit = None
    seller_margin_rate = None
    buyer_prr = None
    seller_discount_rate = None
    if clean_deal and wholesale is not None:
        profit = final_price - wholesale
    if clean_deal and retail and retail > 0:
        buyer_prr = (retail - final_price) / retail
    if clean_deal and retail is not None and wholesale is not None and retail > wholesale:
        seller_margin_rate = (final_price - wholesale) / (retail - wholesale)
        seller_discount_rate = (retail - final_price) / (retail - wholesale)

    return {
        "accepted": accepted,
        "clean_deal": clean_deal,
        "final_price": final_price if clean_deal else None,
        "profit": profit,
        "seller_margin_rate": seller_margin_rate,
        "buyer_prr": buyer_prr,
        "seller_discount_rate": seller_discount_rate,
    }


def empty_group_row(label_key: str, label: str) -> Dict[str, Any]:
    return {
        label_key: label,
        "episodes": 0,
        "accepted": 0,
        "rejected": 0,
        "max_turns": 0,
        "clean_deals": 0,
        "overpayment": 0,
        "out_of_budget": 0,
        "out_of_wholesale": 0,
        "product_substitution": 0,
        "fee_exclusion": 0,
        "irrational_refuse": 0,
        "terminal_rejection_reopened": 0,
        "negated_price_offer": 0,
        "offer_over_budget": 0,
        "offer_over_first": 0,
        "model_behavior_anomaly": 0,
        "diagnostic_flag": 0,
        "data_error": 0,
        "model_error": 0,
        "price_scale_warning": 0,
        "price_scale_repaired": 0,
        "terminal_not_closed": 0,
        "price_extraction_false_offer": 0,
        "partial_payment_price_extraction": 0,
        "system_data_error": 0,
        "deadlock": 0,
        "rational_impasse": 0,
        "turns_total": 0,
        "_final_prices": [],
        "_profits": [],
        "_buyer_prrs": [],
        "_seller_margin_rates": [],
        "_seller_discount_rates": [],
        "budgets": {},
    }


def update_group_row(row: Dict[str, Any], data: Dict[str, Any], anomalies: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    result = data.get("negotiation_result")
    budget = data.get("budget_scenario", "unknown")
    row["episodes"] += 1
    row["accepted"] += int(result == "accepted")
    row["rejected"] += int(result == "rejected")
    row["max_turns"] += int(result == "max_turns_reached")
    row["clean_deals"] += int(bool(metrics.get("clean_deal")))
    row["overpayment"] += int(bool(anomalies.get("overpayment", False)))
    row["out_of_budget"] += int(bool(anomalies.get("out_of_budget", False)))
    row["out_of_wholesale"] += int(bool(anomalies.get("out_of_wholesale", False)))
    row["product_substitution"] += int(bool(anomalies.get("product_substitution", False)))
    row["fee_exclusion"] += int(bool(anomalies.get("fee_exclusion", False)))
    row["irrational_refuse"] += int(bool(anomalies.get("irrational_refuse", False)))
    row["terminal_rejection_reopened"] += int(bool(anomalies.get("terminal_rejection_reopened", False)))
    row["negated_price_offer"] += int(bool(anomalies.get("negated_price_offer", False)))
    row["offer_over_budget"] += int(bool(anomalies.get("offer_over_budget", False)))
    row["offer_over_first"] += int(bool(anomalies.get("offer_over_first", False)))
    row["model_behavior_anomaly"] += int(bool(anomalies.get("model_behavior_anomaly", False)))
    row["diagnostic_flag"] += int(bool(anomalies.get("diagnostic_flag", False)))
    system_data_flags = anomalies.get("system_data_flags", {})
    if not isinstance(system_data_flags, dict):
        system_data_flags = {}
    row["data_error"] += int(bool(system_data_flags.get("data_error", False)))
    row["model_error"] += int(bool(system_data_flags.get("model_error", False)))
    row["price_scale_warning"] += int(bool(system_data_flags.get("price_scale_warning", False)))
    row["price_scale_repaired"] += int(bool(system_data_flags.get("price_scale_repaired", False)))
    row["terminal_not_closed"] += int(bool(system_data_flags.get("terminal_not_closed", False)))
    row["price_extraction_false_offer"] += int(bool(system_data_flags.get("price_extraction_false_offer", False)))
    row["partial_payment_price_extraction"] += int(
        bool(system_data_flags.get("partial_payment_price_extraction", False))
    )
    row["system_data_error"] += int(bool(anomalies.get("system_data_error", False)))
    row["deadlock"] += int(bool(anomalies.get("deadlock", False)))
    row["rational_impasse"] += int(bool(anomalies.get("rational_impasse", False)))
    row["turns_total"] += int(data.get("completed_turns", 0) or 0)
    row["budgets"][budget] = row["budgets"].get(budget, 0) + 1

    if metrics.get("final_price") is not None:
        row["_final_prices"].append(metrics["final_price"])
    if metrics.get("profit") is not None:
        row["_profits"].append(metrics["profit"])
    if metrics.get("buyer_prr") is not None:
        row["_buyer_prrs"].append(metrics["buyer_prr"])
    if metrics.get("seller_margin_rate") is not None:
        row["_seller_margin_rates"].append(metrics["seller_margin_rate"])
    if metrics.get("seller_discount_rate") is not None:
        row["_seller_discount_rates"].append(metrics["seller_discount_rate"])


def finalize_group_row(row: Dict[str, Any]) -> Dict[str, Any]:
    episodes = row["episodes"]
    finalized = {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }
    finalized["accept_rate"] = safe_rate(row["accepted"], episodes)
    finalized["clean_deal_rate"] = safe_rate(row["clean_deals"], episodes)
    finalized["overpayment_rate"] = safe_rate(row["overpayment"], episodes)
    finalized["out_of_budget_rate"] = safe_rate(row["out_of_budget"], episodes)
    finalized["out_of_wholesale_rate"] = safe_rate(row["out_of_wholesale"], episodes)
    finalized["product_substitution_rate"] = safe_rate(row["product_substitution"], episodes)
    finalized["fee_exclusion_rate"] = safe_rate(row["fee_exclusion"], episodes)
    finalized["irrational_refuse_rate"] = safe_rate(row["irrational_refuse"], episodes)
    finalized["terminal_rejection_reopened_rate"] = safe_rate(row["terminal_rejection_reopened"], episodes)
    finalized["negated_price_offer_rate"] = safe_rate(row["negated_price_offer"], episodes)
    finalized["offer_over_budget_rate"] = safe_rate(row["offer_over_budget"], episodes)
    finalized["offer_over_first_rate"] = safe_rate(row["offer_over_first"], episodes)
    finalized["model_behavior_anomaly_rate"] = safe_rate(row["model_behavior_anomaly"], episodes)
    finalized["diagnostic_flag_rate"] = safe_rate(row["diagnostic_flag"], episodes)
    finalized["system_data_error_rate"] = safe_rate(row["system_data_error"], episodes)
    finalized["terminal_not_closed_rate"] = safe_rate(row["terminal_not_closed"], episodes)
    finalized["price_extraction_false_offer_rate"] = safe_rate(row["price_extraction_false_offer"], episodes)
    finalized["partial_payment_price_extraction_rate"] = safe_rate(
        row["partial_payment_price_extraction"], episodes
    )
    finalized["deadlock_rate"] = safe_rate(row["deadlock"], episodes)
    finalized["rational_impasse_rate"] = safe_rate(row["rational_impasse"], episodes)
    finalized["avg_turns"] = round(row["turns_total"] / episodes, 2) if episodes else 0.0
    finalized["avg_final_price"] = safe_money_mean(row["_final_prices"])
    finalized["avg_profit"] = safe_money_mean(row["_profits"])
    finalized["avg_buyer_prr"] = safe_mean(row["_buyer_prrs"])
    finalized["avg_seller_margin_rate"] = safe_mean(row["_seller_margin_rates"])
    finalized["avg_seller_discount_rate"] = safe_mean(row["_seller_discount_rates"])
    return finalized


def summarize(results_dir: Path, include_error_files: bool = False) -> Dict[str, Any]:
    processor = PostDataProcessor(base_dir=str(results_dir))
    by_pair: Dict[str, Dict[str, Any]] = {}
    by_seller: Dict[str, Dict[str, Any]] = {}
    by_buyer: Dict[str, Dict[str, Any]] = {}
    by_budget: Dict[str, Dict[str, Any]] = {}
    total_files = 0
    analyzed_files = 0
    skipped_data_error = 0
    skipped_system_data_error = 0
    skipped_terminal_not_closed = 0
    usage_events: List[Dict[str, Any]] = []
    files_with_usage = 0

    for path in iter_result_files(results_dir):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        total_files += 1
        anomalies = processor.calculate_anomalies(data)
        system_data_flags = anomalies.get("system_data_flags", {})
        if not isinstance(system_data_flags, dict):
            system_data_flags = {}
        if anomalies.get("system_data_error", False) and not include_error_files:
            if system_data_flags.get("data_error"):
                skipped_data_error += 1
            if system_data_flags.get("terminal_not_closed"):
                skipped_terminal_not_closed += 1
            skipped_system_data_error += 1
            continue
        analyzed_files += 1
        file_usage_events = data.get("usage_events")
        if isinstance(file_usage_events, list) and file_usage_events:
            files_with_usage += 1
            usage_events.extend(event for event in file_usage_events if isinstance(event, dict))

        models = data.get("models", {})
        seller = models.get("seller", "unknown")
        buyer = models.get("buyer", "unknown")
        key = f"{seller}__{buyer}"
        pair_row = by_pair.setdefault(key, empty_group_row("pair", key))
        pair_row["seller"] = seller
        pair_row["buyer"] = buyer
        seller_row = by_seller.setdefault(seller, empty_group_row("seller", seller))
        buyer_row = by_buyer.setdefault(buyer, empty_group_row("buyer", buyer))
        budget = data.get("budget_scenario", "unknown")
        budget_row = by_budget.setdefault(budget, empty_group_row("budget", budget))

        metrics = result_metrics(data, anomalies)
        for row in (pair_row, seller_row, buyer_row, budget_row):
            update_group_row(row, data, anomalies, metrics)

    pair_rows = [finalize_group_row(row) for row in by_pair.values()]
    seller_rows = [finalize_group_row(row) for row in by_seller.values()]
    buyer_rows = [finalize_group_row(row) for row in by_buyer.values()]
    budget_rows = [finalize_group_row(row) for row in by_budget.values()]

    pair_rows.sort(key=lambda r: (r["seller"], r["buyer"]))
    seller_rows.sort(key=lambda r: (-r["avg_profit"], -r["accept_rate"], r["seller"]))
    buyer_rows.sort(key=lambda r: (-r["avg_buyer_prr"], -r["accept_rate"], r["buyer"]))
    budget_order = {"low": 0, "wholesale": 1, "mid": 2, "retail": 3, "high": 4}
    budget_rows.sort(key=lambda r: (budget_order.get(r["budget"], 99), r["budget"]))

    model_behavior_keys = list(MODEL_BEHAVIOR_FLAG_KEYS)
    diagnostic_keys = list(DIAGNOSTIC_FLAG_KEYS)
    system_data_keys = list(SYSTEM_DATA_FLAG_KEYS) + ["system_data_error"]

    def summarize_keys(keys: List[str]) -> Dict[str, Any]:
        summary = {key: sum(row.get(key, 0) for row in pair_rows) for key in keys}
        summary.update({f"{key}_rate": safe_rate(summary[key], analyzed_files) for key in keys})
        return summary

    model_behavior_summary = summarize_keys(model_behavior_keys + ["model_behavior_anomaly"])
    diagnostic_summary = summarize_keys(diagnostic_keys + ["diagnostic_flag"])
    system_data_summary = summarize_keys(system_data_keys)
    risk_summary = summarize_keys(
        [
            "overpayment",
            "out_of_budget",
            "out_of_wholesale",
            "product_substitution",
            "fee_exclusion",
            "irrational_refuse",
            "terminal_rejection_reopened",
            "negated_price_offer",
            "deadlock",
            "rational_impasse",
        ]
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "total_files": total_files,
        "analyzed_files": analyzed_files,
        "skipped_data_error": skipped_data_error,
        "skipped_system_data_error": skipped_system_data_error,
        "skipped_terminal_not_closed": skipped_terminal_not_closed,
        "include_error_files": include_error_files,
        "files_with_usage": files_with_usage,
        "usage_summary": summarize_usage_events(usage_events),
        "pairs": pair_rows,
        "seller_leaderboard": seller_rows,
        "buyer_leaderboard": buyer_rows,
        "budget_breakdown": budget_rows,
        "risk_summary": risk_summary,
        "model_behavior_summary": model_behavior_summary,
        "diagnostic_summary": diagnostic_summary,
        "system_data_summary": system_data_summary,
    }


def write_js(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("window.A2ANT_MODEL_REFRESH = ")
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write(";\n")


CSV_FIELDS = [
    "seller",
    "buyer",
    "pair",
    "budget",
    "episodes",
    "accepted",
    "rejected",
    "max_turns",
    "clean_deals",
    "overpayment",
    "out_of_budget",
    "out_of_wholesale",
    "product_substitution",
    "fee_exclusion",
    "irrational_refuse",
    "terminal_rejection_reopened",
    "negated_price_offer",
    "offer_over_budget",
    "offer_over_first",
    "model_behavior_anomaly",
    "diagnostic_flag",
    "data_error",
    "model_error",
    "price_scale_warning",
    "price_scale_repaired",
    "terminal_not_closed",
    "price_extraction_false_offer",
    "partial_payment_price_extraction",
    "system_data_error",
    "deadlock",
    "rational_impasse",
    "accept_rate",
    "clean_deal_rate",
    "overpayment_rate",
    "out_of_budget_rate",
    "out_of_wholesale_rate",
    "product_substitution_rate",
    "fee_exclusion_rate",
    "irrational_refuse_rate",
    "terminal_rejection_reopened_rate",
    "negated_price_offer_rate",
    "offer_over_budget_rate",
    "offer_over_first_rate",
    "model_behavior_anomaly_rate",
    "diagnostic_flag_rate",
    "system_data_error_rate",
    "terminal_not_closed_rate",
    "price_extraction_false_offer_rate",
    "partial_payment_price_extraction_rate",
    "deadlock_rate",
    "rational_impasse_rate",
    "avg_turns",
    "avg_final_price",
    "avg_profit",
    "avg_buyer_prr",
    "avg_seller_margin_rate",
    "avg_seller_discount_rate",
]


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def write_csv_outputs(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(output_dir / "pairs.csv", payload["pairs"])
    write_rows_csv(output_dir / "seller_leaderboard.csv", payload["seller_leaderboard"])
    write_rows_csv(output_dir / "buyer_leaderboard.csv", payload["buyer_leaderboard"])
    write_rows_csv(output_dir / "budget_breakdown.csv", payload["budget_breakdown"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A2A-NT result JSON files.")
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--output-js", default="results/sweep_summary.js")
    parser.add_argument("--output-csv", default="results/sweep_summary.csv", help="Backward-compatible pair summary CSV path")
    parser.add_argument("--output-json", default=None, help="Optional structured JSON summary path")
    parser.add_argument("--output-dir", default=None, help="Optional directory for pairs/seller/buyer/budget CSV outputs")
    parser.add_argument("--include-error-files", action="store_true", help="Include data_error result files in summaries")
    args = parser.parse_args()

    payload = summarize(Path(args.results_dir), include_error_files=args.include_error_files)
    write_js(Path(args.output_js), payload)
    write_rows_csv(Path(args.output_csv), payload["pairs"])
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_dir:
        write_csv_outputs(Path(args.output_dir), payload)
    print(
        f"Summarized {payload['analyzed_files']} analyzed files "
        f"({payload['skipped_system_data_error']} skipped system/data files) into {args.output_js}"
    )


if __name__ == "__main__":
    main()
