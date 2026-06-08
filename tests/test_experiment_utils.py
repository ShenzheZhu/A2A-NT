import json
import tempfile
import unittest
from pathlib import Path

from experiment_utils import (
    calculate_budget_scenarios,
    count_valid_results,
    extract_price_from_text,
    inspect_result_file,
    aggregate_usage_from_results,
    next_experiment_number,
    looks_like_no_price,
    parse_int_csv,
    parse_price,
    price_candidates_from_text,
    safe_path_name,
    select_budget_scenarios,
    select_products,
    summarize_usage_events,
)


class ExperimentUtilsTest(unittest.TestCase):
    def test_parse_price_handles_currency_and_commas(self):
        self.assertEqual(parse_price("$1,234.50"), 1234.5)

    def test_extract_price_from_model_output(self):
        self.assertEqual(extract_price_from_text("Price: 1,234", allow_bare_number=True), 1234)
        self.assertEqual(extract_price_from_text("USD 1,234.50"), 1234.5)
        self.assertIsNone(extract_price_from_text("None", allow_bare_number=True))

    def test_price_candidates_require_currency_by_default(self):
        self.assertEqual(price_candidates_from_text("It is a 65 inch 4K TV for $1,200."), [1200])
        self.assertEqual(price_candidates_from_text("It is a 65 inch 4K TV."), [])
        self.assertTrue(looks_like_no_price("No clear price offer."))

    def test_safe_path_name_preserves_readable_model_label(self):
        self.assertEqual(safe_path_name("openrouter/openai/gpt-4o-mini"), "openrouter__openai__gpt-4o-mini")

    def test_calculate_budget_scenarios(self):
        budgets = calculate_budget_scenarios("$100", "$60")
        self.assertEqual(budgets["high"], 120)
        self.assertEqual(budgets["retail"], 100)
        self.assertEqual(budgets["mid"], 80)
        self.assertEqual(budgets["wholesale"], 60)
        self.assertEqual(budgets["low"], 48)

    def test_select_budget_scenarios_preserves_requested_order(self):
        budgets = calculate_budget_scenarios("$100", "$60")
        selected = select_budget_scenarios(budgets, ["wholesale", "mid"])
        self.assertEqual(list(selected.keys()), ["wholesale", "mid"])
        self.assertEqual(selected["wholesale"], 60)
        self.assertEqual(selected["mid"], 80)

    def test_select_budget_scenarios_rejects_unknown_names(self):
        budgets = calculate_budget_scenarios("$100", "$60")
        with self.assertRaisesRegex(ValueError, "Unsupported budget scenarios: impossible"):
            select_budget_scenarios(budgets, ["impossible"])

    def test_parse_int_csv_allows_empty_input(self):
        self.assertEqual(parse_int_csv(None), [])
        self.assertEqual(parse_int_csv("1, 2,3"), [1, 2, 3])

    def test_select_products_preserves_original_indices(self):
        products = [
            {"id": 10, "Product Name": "A"},
            {"id": 20, "Product Name": "B"},
            {"id": 30, "Product Name": "C"},
            {"id": 40, "Product Name": "D"},
        ]
        selected = select_products(products, start_index=1, product_limit=2, product_ids=[20, 30, 40])
        self.assertEqual(selected, [(1, products[1]), (2, products[2])])

    def test_result_validation_excludes_corrupt_and_data_error_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            valid_dir = root / "seller_a" / "buyer_b" / "product_1" / "budget_mid"
            valid_dir.mkdir(parents=True)
            valid_payload = {
                "product_id": 1,
                "experiment_num": 0,
                "conversation_history": [],
                "data_error": False,
            }
            (valid_dir / "product_1_exp_0.json").write_text(json.dumps(valid_payload), encoding="utf-8")

            error_payload = dict(valid_payload, experiment_num=1, data_error=True)
            (valid_dir / "product_1_exp_1.json").write_text(json.dumps(error_payload), encoding="utf-8")
            (valid_dir / "product_1_exp_2.json").write_text("{bad json", encoding="utf-8")

            self.assertEqual(count_valid_results(root, product_id=1), 1)
            self.assertEqual(count_valid_results(root, product_id=1, include_error_files=True), 2)
            self.assertEqual(next_experiment_number(valid_dir, product_id=1), 3)
            self.assertFalse(inspect_result_file(valid_dir / "product_1_exp_2.json")["valid"])

    def test_summarize_usage_events_groups_by_model_and_role(self):
        summary = summarize_usage_events(
            [
                {
                    "model": "model-a",
                    "role": "buyer",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "estimated_cost_usd": 0.01,
                    "usage_available": True,
                },
                {
                    "model": "model-a",
                    "role": "summary",
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "estimated_cost_usd": 0.002,
                    "usage_available": True,
                },
            ]
        )

        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["prompt_tokens"], 13)
        self.assertEqual(summary["completion_tokens"], 7)
        self.assertEqual(summary["total_tokens"], 20)
        self.assertEqual(summary["estimated_cost_usd"], 0.012)
        self.assertEqual(summary["by_model"]["model-a"]["calls"], 2)
        self.assertEqual(summary["by_role"]["buyer"]["total_tokens"], 15)

    def test_aggregate_usage_from_results_includes_data_error_costs_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            result_dir = root / "seller_a" / "buyer_b" / "product_1" / "budget_mid"
            result_dir.mkdir(parents=True)
            result_payload = {
                "product_id": 1,
                "experiment_num": 0,
                "conversation_history": [],
                "data_error": True,
                "usage_events": [
                    {
                        "model": "model-a",
                        "role": "seller",
                        "prompt_tokens": 4,
                        "completion_tokens": 6,
                        "estimated_cost_usd": 0.004,
                        "usage_available": True,
                    }
                ],
            }
            (result_dir / "product_1_exp_0.json").write_text(json.dumps(result_payload), encoding="utf-8")

            included = aggregate_usage_from_results(root)
            excluded = aggregate_usage_from_results(root, include_error_files=False)

            self.assertEqual(included["result_files"], 1)
            self.assertEqual(included["files_with_usage"], 1)
            self.assertEqual(included["calls"], 1)
            self.assertEqual(included["total_tokens"], 10)
            self.assertEqual(included["estimated_cost_usd"], 0.004)
            self.assertEqual(excluded["result_files"], 0)
            self.assertEqual(excluded["calls"], 0)


if __name__ == "__main__":
    unittest.main()
