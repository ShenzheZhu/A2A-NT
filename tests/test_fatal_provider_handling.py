import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import scripts.run_sweep as run_sweep
from LanguageModel import RUN_FATAL_EXIT_CODE


PRODUCT = {
    "id": 1,
    "Product Name": "AirPods Pro 2",
    "Retail Price": "$249",
    "Wholesale Price": "$174",
    "Features": "Noise cancellation and spatial audio",
}


class FakeFatalConversation:
    def __init__(self, *args, **kwargs):
        self.run_fatal_error = True
        self.completed_turns = 0
        self.negotiation_completed = True
        self.negotiation_result = "model_error"
        self.current_price_offer = None
        self.budget_scenario = None

    def run_negotiation(self):
        return []

    def save_conversation(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)


class FatalProviderHandlingTest(unittest.TestCase):
    def test_main_exits_after_saving_run_fatal_conversation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(main, "Conversation", FakeFatalConversation):
                with self.assertRaises(SystemExit) as ctx:
                    main._run_product_experiment(
                        product_index=0,
                        product=PRODUCT,
                        buyer_model="buyer",
                        seller_model="seller",
                        summary_model="summary",
                        max_turns=1,
                        num_experiments=1,
                        output_dir=tmp_dir,
                        budget_scenarios=["mid"],
                    )

        self.assertEqual(ctx.exception.code, RUN_FATAL_EXIT_CODE)

    def test_run_pair_raises_fatal_sweep_error_for_fatal_exit_code(self):
        args = SimpleNamespace(
            products_file=None,
            summary_model=None,
            max_turns=None,
            num_experiments=None,
            budgets=None,
            output_dir=None,
            product_limit=None,
            include_error_files=False,
            dry_run=False,
            continue_on_error=True,
            judge_confirmation_model=None,
        )
        plan = {
            "products_file": "dataset/products_mini.json",
            "summary_model": "summary",
            "max_turns": 1,
            "num_experiments": 1,
            "budgets": ["mid"],
            "output_dir": "results/test",
        }
        seller = {"label": "Seller", "model": "seller-model"}
        buyer = {"label": "Buyer", "model": "buyer-model"}

        with patch.object(run_sweep.subprocess, "run", return_value=SimpleNamespace(returncode=RUN_FATAL_EXIT_CODE)):
            with self.assertRaises(run_sweep.FatalSweepError) as ctx:
                run_sweep.run_pair(plan, seller, buyer, "test-kind", args)

        self.assertTrue(ctx.exception.result["run_fatal"])
        self.assertEqual(ctx.exception.result["returncode"], RUN_FATAL_EXIT_CODE)


if __name__ == "__main__":
    unittest.main()
