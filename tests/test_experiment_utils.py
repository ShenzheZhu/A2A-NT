import unittest

from experiment_utils import (
    calculate_budget_scenarios,
    extract_price_from_text,
    looks_like_no_price,
    parse_int_csv,
    parse_price,
    price_candidates_from_text,
    safe_path_name,
    select_budget_scenarios,
    select_products,
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


if __name__ == "__main__":
    unittest.main()
