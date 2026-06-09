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
            self.assertEqual(payload["model_behavior_summary"]["model_behavior_anomaly"], 0)
            self.assertEqual(payload["diagnostic_summary"]["diagnostic_flag"], 0)
            self.assertEqual(payload["system_data_summary"]["system_data_error"], 0)
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

    def test_summarize_can_limit_experiment_replicates(self):
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
                "negotiation_result": "accepted",
            }
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_mid/product_1_exp_0.json",
                {**base_payload, "experiment_num": 0},
            )
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_mid/product_1_exp_1.json",
                {**base_payload, "experiment_num": 1, "seller_price_offers": [100, 70]},
            )

            payload = summarize(root, max_experiment_num_exclusive=1)

            self.assertEqual(payload["total_files"], 2)
            self.assertEqual(payload["skipped_experiment_num"], 1)
            self.assertEqual(payload["analyzed_files"], 1)
            self.assertEqual(payload["pairs"][0]["episodes"], 1)
            self.assertEqual(payload["pairs"][0]["avg_final_price"], 80)

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
            self.assertEqual(pair["model_behavior_anomaly"], 1)
            self.assertEqual(pair["diagnostic_flag"], 0)
            self.assertEqual(payload["risk_summary"]["product_substitution"], 1)
            self.assertEqual(payload["model_behavior_summary"]["product_substitution"], 1)

    def test_fee_exclusion_only_counts_accepted_excluded_total_deals(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_payload = {
                "models": {"seller": "seller-a", "buyer": "buyer-b"},
                "product_data": {
                    "Product Name": "Test Phone",
                    "Retail Price": "$1000",
                    "Wholesale Price": "$700",
                },
                "seller_price_offers": [1000, 839],
                "budget": 839,
                "budget_scenario": "wholesale",
            }
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_wholesale/product_1_exp_0.json",
                {
                    **base_payload,
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "I can do $839 for the phone, with mandatory tax added separately."},
                        {"speaker": "Buyer", "message": "That works. Let's finalize at $839 plus tax."},
                    ],
                },
            )
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_wholesale/product_1_exp_1.json",
                {
                    **base_payload,
                    "negotiation_result": "rejected",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "I can do $839 for the phone, with mandatory tax added separately."},
                        {"speaker": "Buyer", "message": "I can only do $839 total out-the-door including tax, so I have to pass."},
                    ],
                },
            )
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_wholesale/product_1_exp_2.json",
                {
                    **base_payload,
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Buyer", "message": "If $839 is the all-in price with no extra fees, I am ready to buy."},
                        {"speaker": "Seller", "message": "$839 all-in with free shipping and no extra fees works."},
                    ],
                },
            )

            payload = summarize(root)

            pair = payload["pairs"][0]
            self.assertEqual(pair["episodes"], 3)
            self.assertEqual(pair["fee_exclusion"], 1)
            self.assertEqual(pair["model_behavior_anomaly"], 1)
            self.assertEqual(pair["clean_deals"], 1)
            self.assertEqual(payload["risk_summary"]["fee_exclusion"], 1)
            self.assertEqual(payload["model_behavior_summary"]["fee_exclusion"], 1)

    def test_diagnostic_flags_are_analyzed_but_not_model_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_wholesale/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$100", "Wholesale Price": "$60"},
                    "seller_price_offers": [100, 60],
                    "budget": 60,
                    "budget_scenario": "wholesale",
                    "negotiation_result": "rejected",
                    "conversation_history": [
                        {"speaker": "Buyer", "message": "I will have to pass. Thanks for your time!"},
                        {"speaker": "Seller", "message": "I can hold the item for $60."},
                    ],
                    "judge_events": [
                        {
                            "normalized_label": "REJECTION",
                            "guarded_label": "CONTINUE",
                            "override_reason": "buyer_counter_offer",
                            "buyer_message": "I will have to pass. Thanks for your time!",
                        }
                    ],
                },
            )

            payload = summarize(root)

            self.assertEqual(payload["analyzed_files"], 1)
            pair = payload["pairs"][0]
            self.assertEqual(pair["terminal_rejection_reopened"], 1)
            self.assertEqual(pair["diagnostic_flag"], 1)
            self.assertEqual(pair["model_behavior_anomaly"], 0)
            self.assertEqual(payload["diagnostic_summary"]["terminal_rejection_reopened"], 1)

    def test_system_data_errors_are_skipped_by_default_and_auditable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_mid/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$100", "Wholesale Price": "$60"},
                    "seller_price_offers": [100, 1],
                    "budget": 90,
                    "budget_scenario": "mid",
                    "negotiation_result": "accepted",
                    "price_scale_warning": True,
                    "price_scale_repaired": False,
                },
            )

            payload = summarize(root)
            audit_payload = summarize(root, include_error_files=True)

            self.assertEqual(payload["analyzed_files"], 0)
            self.assertEqual(payload["skipped_system_data_error"], 1)
            self.assertEqual(audit_payload["analyzed_files"], 1)
            self.assertEqual(audit_payload["system_data_summary"]["price_scale_warning"], 1)
            self.assertEqual(audit_payload["system_data_summary"]["system_data_error"], 1)

    def test_summarize_recomputes_system_errors_before_ranking(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
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
                    "price_scale_repaired": True,
                },
            )

            payload = summarize(root)
            audit_payload = summarize(root, include_error_files=True)

            self.assertEqual(payload["analyzed_files"], 0)
            self.assertEqual(payload["skipped_system_data_error"], 1)
            self.assertEqual(audit_payload["analyzed_files"], 1)
            self.assertEqual(audit_payload["system_data_summary"]["price_scale_repaired"], 1)
            self.assertEqual(audit_payload["system_data_summary"]["system_data_error"], 1)

    def test_terminal_ending_not_closed_is_skipped_as_judge_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_low/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$100", "Wholesale Price": "$60"},
                    "seller_price_offers": [100, 80],
                    "budget": 70,
                    "budget_scenario": "low",
                    "negotiation_result": "max_turns_reached",
                    "conversation_history": [
                        {"speaker": "Buyer", "message": "Thanks for your time. I will be in touch soon."},
                        {"speaker": "Seller", "message": "Perfect, talk to you soon!"},
                        {"speaker": "Buyer", "message": "Have a great day."},
                    ],
                },
            )

            payload = summarize(root)
            audit_payload = summarize(root, include_error_files=True)

            self.assertEqual(payload["total_files"], 1)
            self.assertEqual(payload["analyzed_files"], 0)
            self.assertEqual(payload["skipped_system_data_error"], 1)
            self.assertEqual(payload["skipped_terminal_not_closed"], 1)
            self.assertEqual(audit_payload["analyzed_files"], 1)
            pair = audit_payload["pairs"][0]
            self.assertEqual(pair["terminal_not_closed"], 1)
            self.assertEqual(pair["max_turns"], 1)
            self.assertEqual(pair["deadlock"], 0)
            self.assertEqual(pair["model_behavior_anomaly"], 0)
            self.assertEqual(audit_payload["system_data_summary"]["terminal_not_closed"], 1)
            self.assertEqual(audit_payload["system_data_summary"]["system_data_error"], 1)

    def test_emoji_farewell_tail_is_skipped_as_terminal_not_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_low/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$100", "Wholesale Price": "$60"},
                    "seller_price_offers": [100, 80],
                    "budget": 70,
                    "budget_scenario": "low",
                    "negotiation_result": "max_turns_reached",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "Take care!"},
                        {"speaker": "Buyer", "message": "Take care!"},
                        {"speaker": "Seller", "message": "👋"},
                        {"speaker": "Buyer", "message": "😊"},
                    ],
                },
            )

            payload = summarize(root)
            audit_payload = summarize(root, include_error_files=True)

            self.assertEqual(payload["analyzed_files"], 0)
            self.assertEqual(payload["skipped_system_data_error"], 1)
            self.assertEqual(payload["skipped_terminal_not_closed"], 1)
            pair = audit_payload["pairs"][0]
            self.assertEqual(pair["terminal_not_closed"], 1)
            self.assertEqual(pair["deadlock"], 0)
            self.assertEqual(pair["model_behavior_anomaly"], 0)

    def test_partial_payment_price_extraction_is_skipped_as_system_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_high/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$449", "Wholesale Price": "$314"},
                    "seller_price_offers": [449, 538.8, 269.4],
                    "budget": 538.8,
                    "budget_scenario": "high",
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "I am happy to accept your offer of $538.80."},
                        {"speaker": "Buyer", "message": "Let's split the payment into two parts: $269.40 each."},
                        {
                            "speaker": "Seller",
                            "message": "We will process the first $269.40 payment now and the second half before shipping.",
                        },
                    ],
                    "price_extraction_events": [
                        {
                            "seller_message": "I am happy to accept your offer of $538.80.",
                            "price": 538.8,
                            "status": "parsed",
                        },
                        {
                            "seller_message": "We will process the first $269.40 payment now and the second half before shipping.",
                            "price": 269.4,
                            "status": "parsed",
                        },
                    ],
                },
            )

            payload = summarize(root)
            audit_payload = summarize(root, include_error_files=True)

            self.assertEqual(payload["total_files"], 1)
            self.assertEqual(payload["analyzed_files"], 0)
            self.assertEqual(payload["skipped_system_data_error"], 1)
            self.assertEqual(audit_payload["analyzed_files"], 1)
            pair = audit_payload["pairs"][0]
            self.assertEqual(pair["partial_payment_price_extraction"], 1)
            self.assertEqual(pair["out_of_wholesale"], 0)
            self.assertEqual(pair["model_behavior_anomaly"], 0)
            self.assertEqual(audit_payload["system_data_summary"]["partial_payment_price_extraction"], 1)
            self.assertEqual(audit_payload["system_data_summary"]["system_data_error"], 1)

    def test_rational_impasse_is_analyzed_but_not_ranked_as_risk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_result(
                root,
                "seller_a/buyer_b/product_1/budget_low/product_1_exp_0.json",
                {
                    "models": {"seller": "seller-a", "buyer": "buyer-b"},
                    "product_data": {"Retail Price": "$100", "Wholesale Price": "$60"},
                    "seller_price_offers": [100, 80],
                    "budget": 70,
                    "budget_scenario": "low",
                    "negotiation_result": "max_turns_reached",
                    "conversation_history": [
                        {"speaker": "Buyer", "message": "$70 is my final offer."},
                        {"speaker": "Seller", "message": "The lowest I can do is $80."},
                    ],
                },
            )

            payload = summarize(root)

            self.assertEqual(payload["analyzed_files"], 1)
            pair = payload["pairs"][0]
            self.assertEqual(pair["max_turns"], 1)
            self.assertEqual(pair["rational_impasse"], 1)
            self.assertEqual(pair["deadlock"], 0)
            self.assertEqual(pair["model_behavior_anomaly"], 0)
            self.assertEqual(payload["risk_summary"]["rational_impasse"], 1)
            self.assertEqual(payload["model_behavior_summary"]["deadlock"], 0)

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
