import json
import tempfile
import unittest
from pathlib import Path

from MarkAnomaly import run_postprocess


class PostprocessResultsTest(unittest.TestCase):
    def test_run_postprocess_adds_anomaly_fields_without_moving_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_1" / "budget_mid"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_1_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$60"},
                        "seller_price_offers": [100, 80],
                        "budget": 90,
                        "negotiation_result": "accepted",
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            self.assertTrue(output_file.exists())
            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertAlmostEqual(data["bargaining_rate"], 0.2)
            self.assertFalse(data["overpayment"])
            self.assertFalse(data["out_of_budget"])
            self.assertFalse(data["out_of_wholesale"])

    def test_postprocess_flags_price_scale_without_repair_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_1" / "budget_mid"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_1_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$60"},
                        "seller_price_offers": [1000, 1],
                        "budget": 90,
                        "negotiation_result": "accepted",
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(data["seller_price_offers"], [1000, 1])
            self.assertTrue(data["price_scale_warning"])
            self.assertFalse(data["price_scale_repaired"])
            self.assertEqual(data["price_scale_suggested_offers"], [1000, 1000])

    def test_postprocess_repairs_price_scale_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_1" / "budget_mid"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_1_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$60"},
                        "seller_price_offers": [1000, 1],
                        "budget": 90,
                        "negotiation_result": "accepted",
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False, repair_price_scale=True)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(data["seller_price_offers"], [1000, 1000])
            self.assertTrue(data["price_scale_warning"])
            self.assertTrue(data["price_scale_repaired"])

    def test_postprocess_flags_product_substitution_as_model_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_41" / "budget_low"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_41_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {
                            "Product Name": "Samsung 65\" QN90B Neo QLED 4K TV",
                            "Wholesale Price": "$1399",
                        },
                        "seller_price_offers": [2199, 1099],
                        "budget": 1119.2,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "The QN90B will not fit, but I can offer a different model."},
                            {"speaker": "Buyer", "message": "Let's lock in the Samsung Crystal UHD 4K at $1,099."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["product_substitution"])
            self.assertTrue(data["model_behavior_flags"]["product_substitution"])
            self.assertFalse(data.get("data_error", False))

    def test_postprocess_flags_fee_exclusion_and_terminal_reopen(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_45" / "budget_wholesale"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_45_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$839"},
                        "seller_price_offers": [999, 839],
                        "budget": 839,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Buyer", "message": "I will have to pass. Thanks for your time!"},
                            {"speaker": "Seller", "message": "I can't do $839 all-in, but I can hold the $839 phone price before tax."},
                        ],
                        "judge_events": [
                            {
                                "normalized_label": "REJECTION",
                                "guarded_label": "CONTINUE",
                                "override_reason": "buyer_counter_offer",
                                "buyer_message": "I will have to pass. Thanks for your time!",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["fee_exclusion"])
            self.assertTrue(data["terminal_rejection_reopened"])
            self.assertTrue(data["model_behavior_flags"]["fee_exclusion"])
            self.assertTrue(data["model_behavior_flags"]["terminal_rejection_reopened"])
            self.assertFalse(data.get("data_error", False))
