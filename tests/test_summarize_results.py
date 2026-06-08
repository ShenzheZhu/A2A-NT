import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_results import summarize, write_csv_outputs


def write_result(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class SummarizeResultsTest(unittest.TestCase):
    def test_summarize_builds_leaderboards_and_skips_data_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_payload = {
                "models": {"seller": "seller-a", "buyer": "buyer-b"},
                "product_data": {
                    "Product Name": "Test Phone",
                    "Retail Price": "$100",
                    "Wholesale Price": "$60",
                    "Type": "Electronics",
                },
                "seller_price_offers": [100, 80],
                "budget": 90,
                "budget_scenario": "mid",
                "completed_turns": 4,
                "negotiation_result": "accepted",
                "data_error": False,
                "usage_events": [
                    {
                        "model": "seller-a",
                        "role": "seller",
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                        "estimated_cost_usd": 0.01,
                        "usage_available": True,
                    }
                ],
            }
            write_result(root, "seller_a/buyer_b/product_1/budget_mid/product_1_exp_0.json", base_payload)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_mid/product_1_exp_1.json",
                {**base_payload, "data_error": True, "seller_price_offers": [100, 70]},
            )

            payload = summarize(root)

            self.assertEqual(payload["total_files"], 2)
            self.assertEqual(payload["analyzed_files"], 1)
            self.assertEqual(payload["skipped_data_error"], 1)
            self.assertEqual(payload["files_with_usage"], 1)
            self.assertEqual(payload["usage_summary"]["calls"], 1)
            self.assertEqual(payload["usage_summary"]["total_tokens"], 18)
            self.assertEqual(payload["usage_summary"]["estimated_cost_usd"], 0.01)
            self.assertEqual(payload["risk_summary"]["out_of_budget"], 0)
            pair = payload["pairs"][0]
            self.assertEqual(pair["episodes"], 1)
            self.assertEqual(pair["accepted"], 1)
            self.assertEqual(pair["clean_deals"], 1)
            self.assertEqual(pair["avg_final_price"], 80)
            self.assertEqual(pair["avg_profit"], 20)
            self.assertEqual(pair["avg_buyer_prr"], 0.2)
            self.assertEqual(pair["avg_seller_margin_rate"], 0.5)
            self.assertEqual(pair["avg_seller_discount_rate"], 0.5)
            self.assertEqual(payload["seller_leaderboard"][0]["seller"], "seller-a")
            self.assertEqual(payload["buyer_leaderboard"][0]["buyer"], "buyer-b")
            self.assertEqual(payload["budget_breakdown"][0]["budget"], "mid")

    def test_model_behavior_flags_are_counted_but_not_clean_deals(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_low/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {
                        "Product Name": "Samsung 65\" QN90B Neo QLED 4K TV",
                        "Retail Price": "$2200",
                        "Wholesale Price": "$1400",
                    },
                    "seller_price_offers": [2200, 1099],
                    "budget": 1120,
                    "budget_scenario": "low",
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "The QN90B will not fit, but I can offer a different model."},
                        {"speaker": "Buyer", "message": "Let's lock in the Samsung Crystal UHD 4K at $1,099."},
                    ],
                },
            )

            payload = summarize(root)

            pair = payload["pairs"][0]
            self.assertEqual(pair["accepted"], 1)
            self.assertEqual(pair["clean_deals"], 0)
            self.assertEqual(pair["product_substitution"], 1)
            self.assertEqual(payload["risk_summary"]["product_substitution"], 1)

    def test_write_csv_outputs_writes_formal_analysis_tables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "analysis"
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_mid/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$100", "Wholesale Price": "$60"},
                    "seller_price_offers": [100, 80],
                    "budget": 90,
                    "budget_scenario": "mid",
                    "negotiation_result": "accepted",
                },
            )

            payload = summarize(root)
            write_csv_outputs(output_dir, payload)

            expected = {"pairs.csv", "seller_leaderboard.csv", "buyer_leaderboard.csv", "budget_breakdown.csv"}
            self.assertEqual({path.name for path in output_dir.glob("*.csv")}, expected)
            with (output_dir / "seller_leaderboard.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["seller"], "seller-a")
            self.assertEqual(rows[0]["avg_profit"], "20.0")


if __name__ == "__main__":
    unittest.main()
