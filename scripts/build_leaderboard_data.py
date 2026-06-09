import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PUBLIC_BEHAVIOR_KEYS = [
    "model_behavior_anomaly",
    "fee_exclusion",
    "irrational_refuse",
    "out_of_budget",
    "out_of_wholesale",
    "product_substitution",
    "deadlock",
    "overpayment",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def enabled_models(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in entries if entry.get("enabled", True)]


def model_catalog(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    catalog: Dict[str, Dict[str, str]] = {}
    for group_name in ("frontier_models", "bridge_models"):
        for entry in enabled_models(config.get(group_name, [])):
            model_id = entry["model"]
            if model_id in catalog:
                continue
            catalog[model_id] = {
                "label": entry.get("label", model_id),
                "cohort": "model",
                "cohortLabel": "Model",
            }
    return catalog


def all_enabled_model_labels(config: Dict[str, Any]) -> List[str]:
    labels = []
    seen = set()
    for group_name in ("frontier_models", "bridge_models"):
        for entry in enabled_models(config.get(group_name, [])):
            model_id = entry["model"]
            if model_id in seen:
                continue
            seen.add(model_id)
            labels.append(entry.get("label", model_id))
    return labels


def default_baseline_model(config: Dict[str, Any]) -> Optional[str]:
    if config.get("relative_profit_baseline"):
        return config["relative_profit_baseline"]
    for entry in enabled_models(config.get("bridge_models", [])):
        return entry["model"]
    for group_name in ("frontier_models", "bridge_models"):
        for entry in enabled_models(config.get(group_name, [])):
            return entry["model"]
    return None


def index_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {row.get(key): row for row in rows if row.get(key)}


def pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None


def ratio(value: Any, baseline: Optional[float]) -> Optional[float]:
    if baseline is None or baseline <= 0 or value is None:
        return None
    try:
        return round(float(value) / baseline, 3)
    except (TypeError, ValueError):
        return None


def metric_summary(source: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    summary = {}
    for key in keys:
        count = source.get(key)
        rate = source.get(f"{key}_rate")
        summary[key] = {
            "count": count if count is not None else 0,
            "rate": rate if rate is not None else 0,
        }
    return summary


def numeric(value: Any, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def combined_count(left: Dict[str, Any], right: Dict[str, Any], key: str) -> int:
    return int(numeric(left.get(key)) + numeric(right.get(key)))


def combined_rate(
    left: Dict[str, Any],
    right: Dict[str, Any],
    key: str,
    episodes: int,
) -> Optional[float]:
    if episodes <= 0:
        return None
    return round((combined_count(left, right, key) / episodes) * 100, 2)


def responsible_count(row: Dict[str, Any], key: str) -> int:
    return int(numeric(row.get(f"responsible_{key}", row.get(key))))


def responsible_anomaly_count(row: Dict[str, Any]) -> int:
    return int(numeric(row.get("responsible_model_behavior_anomaly", row.get("model_behavior_anomaly"))))


def rate_from_count(count: int, episodes: int) -> Optional[float]:
    if episodes <= 0:
        return None
    return round((count / episodes) * 100, 2)


def responsible_role_rate(row: Dict[str, Any], key: str) -> Optional[float]:
    episodes = int(numeric(row.get("episodes")))
    return rate_from_count(responsible_count(row, key), episodes)


def combined_responsible_rate(
    seller: Dict[str, Any],
    buyer: Dict[str, Any],
    key: str,
    episodes: int,
) -> Optional[float]:
    return rate_from_count(responsible_count(seller, key) + responsible_count(buyer, key), episodes)


def build_risk_rows(
    catalog: Dict[str, Dict[str, str]],
    seller_rows: Dict[str, Dict[str, Any]],
    buyer_rows: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for model_id, meta in catalog.items():
        seller = seller_rows.get(model_id, {})
        buyer = buyer_rows.get(model_id, {})
        seller_episodes = int(numeric(seller.get("episodes")))
        buyer_episodes = int(numeric(buyer.get("episodes")))
        episodes = seller_episodes + buyer_episodes
        seller_risk_cases = responsible_anomaly_count(seller)
        buyer_risk_cases = responsible_anomaly_count(buyer)
        rows.append(
            {
                "model": meta["label"],
                "modelId": model_id,
                "cohort": meta["cohort"],
                "cohortLabel": meta["cohortLabel"],
                "riskEpisodes": episodes,
                "riskCases": seller_risk_cases + buyer_risk_cases,
                "riskRate": rate_from_count(seller_risk_cases + buyer_risk_cases, episodes),
                "feeExclusionRate": responsible_role_rate(buyer, "fee_exclusion"),
                "irrationalRefusalRate": responsible_role_rate(buyer, "irrational_refuse"),
                "outOfBudgetRate": responsible_role_rate(buyer, "out_of_budget"),
                "outOfWholesaleRate": responsible_role_rate(seller, "out_of_wholesale"),
                "productSubstitutionRate": combined_responsible_rate(
                    seller, buyer, "product_substitution", episodes
                ),
                "deadlockRate": combined_responsible_rate(seller, buyer, "deadlock", episodes),
                "overpaymentRate": responsible_role_rate(buyer, "overpayment"),
                "sellerRiskRate": pct(
                    seller.get("responsible_model_behavior_anomaly_rate", seller.get("model_behavior_anomaly_rate"))
                ),
                "buyerRiskRate": pct(
                    buyer.get("responsible_model_behavior_anomaly_rate", buyer.get("model_behavior_anomaly_rate"))
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["riskRate"] is None,
            row["riskRate"] or 0,
            row["model"],
        )
    )
    return rows


def enabled_labels(entries: List[Dict[str, Any]]) -> List[str]:
    return [entry.get("label", entry["model"]) for entry in enabled_models(entries)]


def infer_product_count(summary: Dict[str, Any]) -> Optional[int]:
    pair_count = len(summary.get("pairs", []))
    budget_count = len(summary.get("budget_breakdown", []))
    included_files = summary.get("included_files", summary.get("total_files"))
    if not pair_count or not budget_count or not included_files:
        return None
    try:
        product_count = int(included_files) / (pair_count * budget_count)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if product_count.is_integer():
        return int(product_count)
    return None


def avg_turns(summary: Dict[str, Any]) -> Optional[float]:
    episodes = sum(int(numeric(row.get("episodes"))) for row in summary.get("pairs", []))
    turns_total = sum(int(numeric(row.get("turns_total"))) for row in summary.get("pairs", []))
    if not episodes:
        return None
    return round(turns_total / episodes, 2)


def build_experiment_details(
    summary: Dict[str, Any],
    config: Dict[str, Any],
    baseline_label: Optional[str],
) -> Dict[str, Any]:
    budgets = config.get("budgets") or [row.get("budget") for row in summary.get("budget_breakdown", [])]
    models = all_enabled_model_labels(config)
    return {
        "modelSet": {
            "models": models,
            "count": len(models),
        },
        "pairCount": len(summary.get("pairs", [])),
        "productCount": infer_product_count(summary),
        "productsFile": config.get("products_file"),
        "budgetSettings": budgets,
        "budgetCount": len(budgets),
        "conversationCount": summary.get("included_files", summary.get("total_files")),
        "analyzedCount": summary.get("analyzed_files"),
        "skippedSystemDataError": summary.get("skipped_system_data_error"),
        "skippedExperimentNum": summary.get("skipped_experiment_num", 0),
        "avgTurns": avg_turns(summary),
        "summaryModel": config.get("summary_model"),
        "maxTurns": config.get("max_turns"),
        "numExperiments": config.get("num_experiments"),
        "baselineLabel": baseline_label,
    }


def build_payload(
    summary: Dict[str, Any],
    config: Dict[str, Any],
    baseline_model: Optional[str] = None,
) -> Dict[str, Any]:
    catalog = model_catalog(config)
    seller_rows = index_rows(summary.get("seller_leaderboard", []), "seller")
    buyer_rows = index_rows(summary.get("buyer_leaderboard", []), "buyer")
    baseline_model = baseline_model or default_baseline_model(config)
    baseline_profit = None
    if baseline_model and baseline_model in seller_rows:
        baseline_profit = seller_rows[baseline_model].get("avg_profit")
        try:
            baseline_profit = float(baseline_profit)
        except (TypeError, ValueError):
            baseline_profit = None

    rows = []
    for model_id, meta in catalog.items():
        seller = seller_rows.get(model_id, {})
        buyer = buyer_rows.get(model_id, {})
        avg_profit = seller.get("avg_profit")
        rows.append(
            {
                "model": meta["label"],
                "modelId": model_id,
                "cohort": meta["cohort"],
                "cohortLabel": meta["cohortLabel"],
                "relativeProfit": ratio(avg_profit, baseline_profit),
                "sellerPrr": pct(seller.get("avg_seller_discount_rate")),
                "buyerPrr": pct(buyer.get("avg_buyer_prr")),
                "avgProfit": avg_profit,
                "sellerEpisodes": seller.get("episodes", 0),
                "buyerEpisodes": buyer.get("episodes", 0),
                "cleanDealRate": seller.get("clean_deal_rate", 0),
                "acceptRate": seller.get("accept_rate", 0),
            }
        )

    rows.sort(
        key=lambda row: (
            row["relativeProfit"] is None,
            -(row["relativeProfit"] or 0),
            row["model"],
        )
    )
    baseline_label = catalog.get(baseline_model, {}).get("label") if baseline_model else None
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceGeneratedAt": summary.get("generated_at"),
        "sourceResultsDir": summary.get("results_dir"),
        "totalFiles": summary.get("total_files"),
        "includedFiles": summary.get("included_files", summary.get("total_files")),
        "analyzedFiles": summary.get("analyzed_files"),
        "skippedSystemDataError": summary.get("skipped_system_data_error"),
        "skippedExperimentNum": summary.get("skipped_experiment_num", 0),
        "baselineModel": baseline_model,
        "baselineLabel": baseline_label,
        "baselineAvgProfit": baseline_profit,
        "experimentDetails": build_experiment_details(summary, config, baseline_label),
        "modelBehaviorSummary": metric_summary(
            summary.get("model_behavior_summary", {}),
            PUBLIC_BEHAVIOR_KEYS,
        ),
        "riskRows": build_risk_rows(catalog, seller_rows, buyer_rows),
        "rows": rows,
    }


def write_js(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("window.A2ANT_LEADERBOARD_DATA = ")
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write(";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build website leaderboard data from an A2A-NT summary JSON.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--config", default="configs/sweep_example.json")
    parser.add_argument("--output-js", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--baseline-model", default=None)
    args = parser.parse_args()

    summary = load_json(Path(args.summary_json))
    config = load_json(Path(args.config))
    payload = build_payload(summary, config, baseline_model=args.baseline_model)
    write_js(Path(args.output_js), payload)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(payload['rows'])} leaderboard rows to {args.output_js} "
        f"using baseline {payload.get('baselineLabel') or payload.get('baselineModel')}"
    )


if __name__ == "__main__":
    main()
