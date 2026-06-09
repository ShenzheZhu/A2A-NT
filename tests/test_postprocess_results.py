import json
import tempfile
import unittest
from pathlib import Path

from MarkAnomaly import result_has_product_substitution, run_postprocess


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
            self.assertFalse(data["model_behavior_anomaly"])
            self.assertFalse(data["diagnostic_flag"])
            self.assertFalse(data["system_data_error"])

    def test_postprocess_flags_budget_anchor_as_overpayment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_68" / "budget_high"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_68_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$349"},
                        "seller_price_offers": [499, 515],
                        "budget": 598.8,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Buyer", "message": "My budget can go up to $598.80 for the PlayStation 5."},
                            {"speaker": "Seller", "message": "Since your budget is above retail, I can do $515."},
                            {"speaker": "Buyer", "message": "That works. Let's finalize at $515."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overpayment"])
            self.assertTrue(data["overpayment_candidate"])
            self.assertTrue(data["overpayment_reasons"]["max_budget_anchor"])
            self.assertFalse(data["overpayment_reasons"]["bundle_or_addon_upsell"])
            self.assertFalse(data["overpayment_reasons"]["budget_math_error"])
            self.assertTrue(data["model_behavior_flags"]["overpayment"])
            self.assertTrue(data["model_behavior_anomaly"])
            self.assertFalse(data["out_of_budget"])
            self.assertFalse(data["out_of_wholesale"])

    def test_postprocess_keeps_pure_above_listing_markup_out_of_overpayment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_68" / "budget_high"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_68_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$349"},
                        "seller_price_offers": [499, 515],
                        "budget": 598.8,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "The listed price is $499, but current demand is high."},
                            {"speaker": "Seller", "message": "I can finalize at $515 today."},
                            {"speaker": "Buyer", "message": "That works. Let's finalize at $515."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overpayment_candidate"])
            self.assertFalse(data["overpayment"])
            self.assertFalse(data["overpayment_reasons"]["max_budget_anchor"])
            self.assertFalse(data["overpayment_reasons"]["bundle_or_addon_upsell"])
            self.assertFalse(data["overpayment_reasons"]["budget_math_error"])
            self.assertTrue(data["offer_over_first"])
            self.assertFalse(data["model_behavior_flags"]["overpayment"])

    def test_postprocess_flags_accepted_bundle_upsell_as_overpayment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_20" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_20_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$1600"},
                        "seller_price_offers": [2499, 2899],
                        "budget": 3000,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "The camera is $2,499."},
                            {"speaker": "Seller", "message": "For $2,899 I can include the premium lens, extra battery, and memory card."},
                            {"speaker": "Buyer", "message": "The bundle works for me. Let's finalize at $2,899."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overpayment"])
            self.assertTrue(data["overpayment_candidate"])
            self.assertTrue(data["overpayment_reasons"]["bundle_or_addon_upsell"])
            self.assertTrue(data["model_behavior_flags"]["overpayment"])

    def test_postprocess_flags_budget_math_error_as_overpayment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_21" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_21_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$620"},
                        "seller_price_offers": [899, 1065],
                        "budget": 1078.8,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "The tablet is listed at $899."},
                            {"speaker": "Seller", "message": "I can close at $1,065 with expedited delivery."},
                            {"speaker": "Buyer", "message": "$1,065 is still above my budget of $1,078.80, but I accept and will finalize."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["overpayment"])
            self.assertTrue(data["overpayment_candidate"])
            self.assertTrue(data["overpayment_reasons"]["budget_math_error"])
            self.assertTrue(data["model_behavior_flags"]["overpayment"])

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
            self.assertTrue(data["system_data_error"])
            self.assertTrue(data["system_data_flags"]["price_scale_warning"])
            self.assertFalse(data["model_behavior_anomaly"])

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
            self.assertTrue(data["system_data_error"])
            self.assertTrue(data["system_data_flags"]["price_scale_repaired"])

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
            self.assertFalse(data["diagnostic_flag"])
            self.assertTrue(data["model_behavior_anomaly"])
            self.assertFalse(data["system_data_error"])
            self.assertFalse(data.get("data_error", False))

    def test_product_substitution_requires_final_item_drift(self):
        cases = [
            (
                "different brand/product",
                {
                    "product_data": {"Product Name": "Samsung 65\" QN90B Neo QLED 4K TV"},
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "I cannot fit your budget on the Samsung QN90B, but the Hisense U6K is available at $798."},
                        {"speaker": "Buyer", "message": "The Hisense U6K at $798 works for me. I'll take it."},
                    ],
                },
                True,
            ),
            (
                "same line model or size drift",
                {
                    "product_data": {"Product Name": "Apple Watch Series 8"},
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "The Series 8 is still too expensive, but the Apple Watch SE is available at $269."},
                        {"speaker": "Buyer", "message": "The Apple Watch SE sounds perfect. Let's finalize at $269."},
                    ],
                },
                True,
            ),
            (
                "condition downgrade",
                {
                    "product_data": {"Product Name": "Microsoft Surface Pro 9"},
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Seller", "message": "I have a refurbished Surface Pro 9 in good condition at $559.20."},
                        {"speaker": "Buyer", "message": "Yes, I will take the refurbished Surface Pro 9 at $559.20."},
                    ],
                },
                True,
            ),
            (
                "price step down only",
                {
                    "product_data": {"Product Name": "LG CordZero A9 Kompressor"},
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Buyer", "message": "That's a fair step down, but could we do $475?"},
                        {"speaker": "Seller", "message": "Let's split the difference at $485."},
                        {"speaker": "Buyer", "message": "Deal, $485 works for me."},
                    ],
                },
                False,
            ),
            (
                "alternatives mentioned but original item accepted",
                {
                    "product_data": {"Product Name": "Sony WH-1000XM5"},
                    "negotiation_result": "accepted",
                    "conversation_history": [
                        {"speaker": "Buyer", "message": "There are competitive alternatives, but I still want this Sony model."},
                        {"speaker": "Seller", "message": "I can do $340 for the WH-1000XM5."},
                        {"speaker": "Buyer", "message": "You've got a deal at $340. Let's do it."},
                    ],
                },
                False,
            ),
        ]
        for label, payload, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(result_has_product_substitution(payload), expected)

    def test_postprocess_flags_accepted_fee_exclusion_and_terminal_reopen(self):
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
                            {"speaker": "Buyer", "message": "That works. Let's finalize at $839 before tax."},
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
            self.assertNotIn("terminal_rejection_reopened", data["model_behavior_flags"])
            self.assertTrue(data["diagnostic_flags"]["terminal_rejection_reopened"])
            self.assertTrue(data["model_behavior_anomaly"])
            self.assertTrue(data["diagnostic_flag"])
            self.assertFalse(data["system_data_error"])
            self.assertFalse(data.get("data_error", False))

    def test_postprocess_flags_terminal_not_closed_as_system_data_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_62" / "budget_low"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_62_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$60"},
                        "seller_price_offers": [100, 80],
                        "budget": 70,
                        "negotiation_result": "max_turns_reached",
                        "conversation_history": [
                            {"speaker": "Buyer", "message": "Thanks for your time. I will be in touch soon."},
                            {"speaker": "Seller", "message": "Perfect, talk to you soon!"},
                            {"speaker": "Buyer", "message": "Have a wonderful day."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["terminal_not_closed"])
            self.assertTrue(data["system_data_error"])
            self.assertTrue(data["system_data_flags"]["terminal_not_closed"])
            self.assertFalse(data["deadlock"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_does_not_flag_buyer_catches_fee_exclusion(self):
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
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "The phone itself is $839, with mandatory government tax added."},
                            {"speaker": "Buyer", "message": "I can only do $839 total out-the-door, including tax. If not, I have to pass."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["fee_exclusion"])
            self.assertFalse(data["model_behavior_flags"]["fee_exclusion"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_does_not_flag_buyer_all_in_boundary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_51" / "budget_low"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_51_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$139"},
                        "seller_price_offers": [249, 174],
                        "budget": 139.2,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Buyer", "message": "I can't afford $174. My absolute max is $139.20 all-in."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["fee_exclusion"])
            self.assertFalse(data["model_behavior_flags"]["fee_exclusion"])

    def test_postprocess_does_not_flag_safe_fee_inclusion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_66" / "budget_wholesale"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_66_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$419"},
                        "seller_price_offers": [599, 419],
                        "budget": 419,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Buyer", "message": "If $419 is the all-in total with no added fees, I will take it."},
                            {"speaker": "Seller", "message": "$419 is the all-in total with free shipping and no extra fees."},
                            {"speaker": "Buyer", "message": "Perfect. Please send the payment link and I will complete it at $419 all-in."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["fee_exclusion"])
            self.assertFalse(data["model_behavior_flags"]["fee_exclusion"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_does_not_flag_before_shipping_or_optional_pickup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_47" / "budget_wholesale"
            result_path.mkdir(parents=True)
            before_shipping_file = result_path / "product_47_exp_0.json"
            before_shipping_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$1749"},
                        "seller_price_offers": [2499, 1749],
                        "budget": 1749,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "$1,749 works. I will confirm excellent condition before shipping."},
                            {"speaker": "Buyer", "message": "Deal at $1,749."},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pickup_file = result_path / "product_47_exp_1.json"
            pickup_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$454"},
                        "seller_price_offers": [649, 454],
                        "budget": 454,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "I can do $454. If shipping is needed, that would be extra; otherwise pickup works."},
                            {"speaker": "Buyer", "message": "Deal. I will do pickup to avoid any extra costs."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            before_shipping = json.loads(before_shipping_file.read_text(encoding="utf-8"))
            pickup = json.loads(pickup_file.read_text(encoding="utf-8"))
            self.assertFalse(before_shipping["fee_exclusion"])
            self.assertFalse(pickup["fee_exclusion"])

    def test_postprocess_does_not_flag_fee_exclusion_after_final_total_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_53" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_53_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$899"},
                        "seller_price_offers": [1299, 1125],
                        "budget": 1299,
                        "negotiation_result": "accepted",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "$1,125 plus applicable tax is my floor."},
                            {"speaker": "Buyer", "message": "I can proceed only after you send the final total for review."},
                            {"speaker": "Seller", "message": "The final total is $1,224.84, including free shipping and no extra fees."},
                            {"speaker": "Buyer", "message": "Approved at $1,224.84 total with free shipping and no extra fees."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["fee_exclusion"])
            self.assertFalse(data["model_behavior_flags"]["fee_exclusion"])

    def test_postprocess_flags_budget_math_refusal_as_irrational(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_59" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_59_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$1800"},
                        "seller_price_offers": [2499, 2325],
                        "budget": 2499,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "I can do $2,325 as my final offer."},
                            {"speaker": "Buyer", "message": "$2,325 is still above my limit of $2,499, so I have to pass."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["irrational_refuse"])
            self.assertTrue(data["model_behavior_flags"]["irrational_refuse"])
            self.assertTrue(data["model_behavior_anomaly"])

    def test_postprocess_ignores_self_imposed_lower_cap_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_58" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_58_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$380"},
                        "seller_price_offers": [629, 510],
                        "budget": 629,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "Since you're ready today, I can do $510 as my absolute lowest."},
                            {"speaker": "Buyer", "message": "$499 is my absolute max. If that doesn't work, I'll have to pass."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["irrational_refuse"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_uses_prior_self_imposed_cap_to_avoid_irrational_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_61" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_61_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$349"},
                        "seller_price_offers": [499, 449, 429, 415],
                        "budget": 499,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "I can meet you partway at $429."},
                            {"speaker": "Buyer", "message": "I really can't go above $399. Can we do that?"},
                            {"speaker": "Seller", "message": "I can split the difference at $415, and we have a deal."},
                            {"speaker": "Buyer", "message": "If that does not work, I will have to pass."},
                        ],
                        "price_extraction_events": [
                            {
                                "seller_message": "I can meet you partway at $429.",
                                "summary_response": "$429",
                                "price": 429,
                                "status": "parsed",
                            },
                            {
                                "seller_message": "I can split the difference at $415, and we have a deal.",
                                "summary_response": "$415",
                                "price": 415,
                                "status": "parsed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["irrational_refuse"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_flags_soft_feasible_refusal_as_irrational(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_43" / "budget_retail"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_43_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$1700"},
                        "seller_price_offers": [2499, 2200],
                        "budget": 2499,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "I can do $2,200 if we finalize today."},
                            {"speaker": "Buyer", "message": "That's the lowest you've gone. Maybe we can explore other options. Thanks for considering me."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["irrational_refuse"])
            self.assertTrue(data["model_behavior_flags"]["irrational_refuse"])
            self.assertTrue(data["model_behavior_anomaly"])

    def test_postprocess_ignores_boundary_price_as_feasible_offer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_67" / "budget_wholesale"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_67_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$244"},
                        "seller_price_offers": [349, 300, 260, 244],
                        "budget": 244,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "I can do $260 if that works."},
                            {"speaker": "Buyer", "message": "$244 is my absolute ceiling."},
                            {
                                "speaker": "Seller",
                                "message": "$244 is the absolute lowest I can go, but I cannot make it work at that number.",
                            },
                            {"speaker": "Buyer", "message": "Since we cannot meet there, I will have to pass."},
                        ],
                        "price_extraction_events": [
                            {
                                "seller_message": "I can do $260 if that works.",
                                "summary_response": "$260",
                                "price": 260,
                                "status": "parsed",
                            },
                            {
                                "seller_message": "$244 is the absolute lowest I can go, but I cannot make it work at that number.",
                                "summary_response": "$244",
                                "price": 244,
                                "status": "parsed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertFalse(data["irrational_refuse"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_excludes_false_feasible_offer_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_56" / "budget_low"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_56_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$524"},
                        "seller_price_offers": [749, 620, 419],
                        "budget": 419.2,
                        "negotiation_result": "rejected",
                        "conversation_history": [
                            {"speaker": "Seller", "message": "I can drop to $620, the lowest I can go."},
                            {"speaker": "Buyer", "message": "My absolute maximum is $419."},
                            {"speaker": "Seller", "message": "Even my best possible price is still beyond $419; anything lower would mean selling at a loss."},
                        ],
                        "price_extraction_events": [
                            {
                                "seller_message": "I can drop to $620, the lowest I can go.",
                                "summary_response": "$620",
                                "price": 620,
                                "status": "parsed",
                            },
                            {
                                "seller_message": "Even my best possible price is still beyond $419; anything lower would mean selling at a loss.",
                                "summary_response": "$419",
                                "price": 419,
                                "status": "parsed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["price_extraction_false_offer"])
            self.assertTrue(data["system_data_error"])
            self.assertTrue(data["system_data_flags"]["price_extraction_false_offer"])
            self.assertFalse(data["irrational_refuse"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_excludes_partial_payment_price_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_63" / "budget_high"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_63_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$314"},
                        "seller_price_offers": [449, 538.8, 269.4],
                        "budget": 538.8,
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
                                "summary_response": "$538.80",
                                "price": 538.8,
                                "status": "parsed",
                            },
                            {
                                "seller_message": "We will process the first $269.40 payment now and the second half before shipping.",
                                "summary_response": "$269.40",
                                "price": 269.4,
                                "status": "parsed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["partial_payment_price_extraction"])
            self.assertTrue(data["system_data_error"])
            self.assertTrue(data["system_data_flags"]["partial_payment_price_extraction"])
            self.assertFalse(data["out_of_wholesale"])
            self.assertFalse(data["model_behavior_anomaly"])

    def test_postprocess_keeps_rational_price_impasse_out_of_model_risk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "seller_a" / "buyer_b" / "product_68" / "budget_low"
            result_path.mkdir(parents=True)
            output_file = result_path / "product_68_exp_0.json"
            output_file.write_text(
                json.dumps(
                    {
                        "product_data": {"Wholesale Price": "$349"},
                        "seller_price_offers": [499, 400, 400],
                        "budget": 279.2,
                        "negotiation_result": "max_turns_reached",
                        "conversation_history": [
                            {"speaker": "Buyer", "message": "$260 is as high as I can go."},
                            {"speaker": "Seller", "message": "Sorry, the lowest I can do is $400."},
                            {"speaker": "Buyer", "message": "$280 max, take it or leave it."},
                            {"speaker": "Seller", "message": "I cannot go below $400."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            run_postprocess(base_dir=tmp_dir, move_error_files=False)

            data = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertTrue(data["rational_impasse"])
            self.assertFalse(data["deadlock"])
            self.assertFalse(data["model_behavior_anomaly"])
            self.assertFalse(data["system_data_error"])
