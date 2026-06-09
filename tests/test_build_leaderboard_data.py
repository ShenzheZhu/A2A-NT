import unittest

from scripts.build_leaderboard_data import build_payload


class BuildLeaderboardDataTest(unittest.TestCase):
    def test_build_payload_uses_unified_model_pool_and_configured_baseline(self):
        config = {
            "relative_profit_baseline": "provider/bridge-model",
            "frontier_models": [
                {"label": "New Model", "model": "provider/new-model"},
            ],
            "bridge_models": [
                {"label": "Bridge Model", "model": "provider/bridge-model", "enabled": True},
            ],
        }
        summary = {
            "generated_at": "2026-06-08T00:00:00+00:00",
            "results_dir": "results/run",
            "total_files": 5,
            "included_files": 4,
            "analyzed_files": 4,
            "skipped_experiment_num": 1,
            "skipped_system_data_error": 0,
            "pairs": [{"pair": "provider/new-model__provider/bridge-model", "episodes": 4, "turns_total": 18}],
            "budget_breakdown": [{"budget": "mid"}, {"budget": "retail"}],
            "seller_leaderboard": [
                {
                    "seller": "provider/new-model",
                    "episodes": 2,
                    "avg_profit": 30,
                    "avg_seller_discount_rate": 0.25,
                    "clean_deal_rate": 1.0,
                    "accept_rate": 1.0,
                    "model_behavior_anomaly": 1,
                    "fee_exclusion": 1,
                    "out_of_budget": 0,
                    "out_of_wholesale": 1,
                    "overpayment": 0,
                    "responsible_model_behavior_anomaly": 1,
                    "responsible_fee_exclusion": 0,
                    "responsible_out_of_budget": 0,
                    "responsible_out_of_wholesale": 1,
                    "responsible_overpayment": 0,
                    "responsible_model_behavior_anomaly_rate": 0.5,
                },
                {
                    "seller": "provider/bridge-model",
                    "episodes": 2,
                    "avg_profit": 10,
                    "avg_seller_discount_rate": 0.4,
                    "clean_deal_rate": 0.5,
                    "accept_rate": 0.5,
                    "model_behavior_anomaly": 0,
                    "fee_exclusion": 0,
                    "out_of_budget": 0,
                    "out_of_wholesale": 0,
                    "overpayment": 0,
                    "responsible_model_behavior_anomaly": 0,
                    "responsible_fee_exclusion": 0,
                    "responsible_out_of_budget": 0,
                    "responsible_out_of_wholesale": 0,
                    "responsible_overpayment": 0,
                    "responsible_model_behavior_anomaly_rate": 0,
                },
            ],
            "buyer_leaderboard": [
                {
                    "buyer": "provider/new-model",
                    "episodes": 2,
                    "avg_buyer_prr": 0.12,
                    "model_behavior_anomaly": 1,
                    "fee_exclusion": 0,
                    "out_of_budget": 1,
                    "out_of_wholesale": 0,
                    "overpayment": 1,
                    "responsible_model_behavior_anomaly": 1,
                    "responsible_fee_exclusion": 0,
                    "responsible_out_of_budget": 1,
                    "responsible_out_of_wholesale": 0,
                    "responsible_overpayment": 1,
                    "responsible_model_behavior_anomaly_rate": 0.5,
                },
                {
                    "buyer": "provider/bridge-model",
                    "episodes": 2,
                    "avg_buyer_prr": 0.08,
                    "model_behavior_anomaly": 0,
                    "fee_exclusion": 0,
                    "out_of_budget": 0,
                    "out_of_wholesale": 0,
                    "overpayment": 0,
                    "responsible_model_behavior_anomaly": 0,
                    "responsible_fee_exclusion": 0,
                    "responsible_out_of_budget": 0,
                    "responsible_out_of_wholesale": 0,
                    "responsible_overpayment": 0,
                    "responsible_model_behavior_anomaly_rate": 0,
                },
            ],
            "model_behavior_summary": {
                "model_behavior_anomaly": 2,
                "model_behavior_anomaly_rate": 0.5,
                "fee_exclusion": 1,
                "fee_exclusion_rate": 0.25,
            },
        }

        payload = build_payload(summary, config)

        self.assertEqual(payload["baselineModel"], "provider/bridge-model")
        self.assertEqual(payload["baselineLabel"], "Bridge Model")
        self.assertEqual(payload["baselineAvgProfit"], 10)
        rows = {row["modelId"]: row for row in payload["rows"]}
        self.assertEqual(rows["provider/new-model"]["cohortLabel"], "Model")
        self.assertEqual(rows["provider/bridge-model"]["cohortLabel"], "Model")
        self.assertEqual(rows["provider/new-model"]["relativeProfit"], 3.0)
        self.assertEqual(rows["provider/new-model"]["sellerPrr"], 25.0)
        self.assertEqual(rows["provider/new-model"]["buyerPrr"], 12.0)
        self.assertEqual(payload["modelBehaviorSummary"]["model_behavior_anomaly"]["count"], 2)
        self.assertEqual(payload["modelBehaviorSummary"]["model_behavior_anomaly"]["rate"], 0.5)
        self.assertEqual(payload["modelBehaviorSummary"]["fee_exclusion"]["count"], 1)
        self.assertEqual(payload["modelBehaviorSummary"]["fee_exclusion"]["rate"], 0.25)
        self.assertEqual(payload["modelBehaviorSummary"]["deadlock"]["count"], 0)
        risk_rows = {row["modelId"]: row for row in payload["riskRows"]}
        self.assertEqual(risk_rows["provider/new-model"]["riskCases"], 2)
        self.assertEqual(risk_rows["provider/new-model"]["riskEpisodes"], 4)
        self.assertEqual(risk_rows["provider/new-model"]["riskRate"], 50.0)
        self.assertEqual(risk_rows["provider/new-model"]["feeExclusionRate"], 0.0)
        self.assertEqual(risk_rows["provider/new-model"]["outOfBudgetRate"], 50.0)
        self.assertEqual(risk_rows["provider/new-model"]["outOfWholesaleRate"], 50.0)
        self.assertEqual(risk_rows["provider/new-model"]["overpaymentRate"], 50.0)
        self.assertEqual(risk_rows["provider/new-model"]["sellerRiskRate"], 50.0)
        self.assertEqual(risk_rows["provider/new-model"]["buyerRiskRate"], 50.0)
        self.assertEqual(risk_rows["provider/bridge-model"]["riskRate"], 0.0)
        details = payload["experimentDetails"]
        self.assertEqual(details["modelSet"]["models"], ["New Model", "Bridge Model"])
        self.assertEqual(details["modelSet"]["count"], 2)
        self.assertEqual(details["pairCount"], 1)
        self.assertEqual(details["productCount"], 2)
        self.assertEqual(details["budgetSettings"], ["mid", "retail"])
        self.assertEqual(details["conversationCount"], 4)
        self.assertEqual(details["skippedExperimentNum"], 1)
        self.assertEqual(payload["totalFiles"], 5)
        self.assertEqual(payload["includedFiles"], 4)
        self.assertEqual(payload["skippedExperimentNum"], 1)
        self.assertEqual(details["analyzedCount"], 4)
        self.assertEqual(details["avgTurns"], 4.5)
        self.assertEqual(details["baselineLabel"], "Bridge Model")


if __name__ == "__main__":
    unittest.main()
