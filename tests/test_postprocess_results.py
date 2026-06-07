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
