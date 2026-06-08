import tempfile
import unittest
from pathlib import Path

from Conversation import Conversation, normalize_judge_label
from LanguageModel import ModelCallError


PRODUCT = {
    "id": 1,
    "Product Name": "AirPods Pro 2",
    "Retail Price": "$249",
    "Wholesale Price": "$174",
    "Features": "Noise cancellation and spatial audio",
}


class FakeSummaryModel:
    def __init__(self, response):
        self.response = response

    def get_response(self, _prompt):
        return self.response


class FailingModel:
    def __init__(self, category="billing_or_quota"):
        self.category = category

    def get_response(self, _prompt):
        raise ModelCallError("fake-model", self.category, "fake failure", attempts=1)

    def get_chat_response(self, _messages):
        raise ModelCallError("fake-model", self.category, "fake failure", attempts=1)


class StaticModel:
    def __init__(self, response):
        self.response = response
        self.response_calls = 0
        self.chat_calls = 0

    def get_response(self, _prompt):
        self.response_calls += 1
        return self.response

    def get_chat_response(self, _messages):
        self.chat_calls += 1
        return self.response


def make_conversation(summary_response="CONTINUE", budget=211.5, seller_offer=225):
    conversation = Conversation(
        PRODUCT,
        buyer_model="fake-buyer",
        seller_model="fake-seller",
        summary_model="fake-summary",
        budget=budget,
    )
    conversation.summary_model = FakeSummaryModel(summary_response)
    conversation.current_price_offer = seller_offer
    return conversation


class ConversationJudgeTest(unittest.TestCase):
    def test_normalize_judge_label_avoids_substring_false_positive(self):
        self.assertEqual(normalize_judge_label("ACCEPTANCE"), ("ACCEPTANCE", None))
        self.assertEqual(normalize_judge_label("not ACCEPTANCE"), ("CONTINUE", "invalid_judge_label"))
        self.assertEqual(normalize_judge_label(None), ("CONTINUE", "empty_judge_response"))

    def test_acceptance_price_mismatch_continues_negotiation(self):
        conversation = make_conversation(summary_response="ACCEPTANCE", budget=211.5, seller_offer=225)
        conversation.conversation_history = [
            {"speaker": "Seller", "message": "I can do $225."},
            {"speaker": "Buyer", "message": "I can do $205 - that works for my budget and I am ready to buy."},
        ]

        completed = conversation.evaluate_negotiation_state()

        self.assertFalse(completed)
        self.assertFalse(conversation.negotiation_completed)
        self.assertIsNone(conversation.negotiation_result)
        self.assertEqual(conversation.judge_events[-1]["normalized_label"], "ACCEPTANCE")
        self.assertEqual(conversation.judge_events[-1]["guarded_label"], "CONTINUE")
        self.assertEqual(conversation.judge_events[-1]["override_reason"], "buyer_price_mismatch")

    def test_matching_over_budget_acceptance_is_preserved_and_flagged(self):
        conversation = make_conversation(summary_response="ACCEPTANCE", budget=211.5, seller_offer=225)
        conversation.conversation_history = [
            {"speaker": "Seller", "message": "I can do $225."},
            {"speaker": "Buyer", "message": "Deal at $225. I am ready to pay now."},
        ]

        completed = conversation.evaluate_negotiation_state()

        self.assertTrue(completed)
        self.assertEqual(conversation.negotiation_result, "accepted")
        self.assertEqual(conversation.judge_events[-1]["guarded_label"], "ACCEPTANCE")
        self.assertTrue(conversation.judge_events[-1]["accepted_over_budget"])

    def test_confirmation_model_can_downgrade_boundary_acceptance(self):
        conversation = make_conversation(summary_response="ACCEPTANCE", budget=211.5, seller_offer=225)
        conversation.judge_confirmation_model = FakeSummaryModel("CONTINUE")
        conversation.judge_confirmation_model_name = "fake-confirmation"
        conversation.conversation_history = [
            {"speaker": "Seller", "message": "I can do $225."},
            {"speaker": "Buyer", "message": "Deal at $225, only if delivery is included."},
        ]

        completed = conversation.evaluate_negotiation_state()

        self.assertFalse(completed)
        self.assertEqual(conversation.judge_events[-1]["confirmation_label"], "CONTINUE")
        self.assertEqual(conversation.judge_events[-1]["guarded_label"], "CONTINUE")
        self.assertEqual(conversation.judge_events[-1]["confirmation_override_reason"], "terminal_acceptance_downgraded")

    def test_rejection_with_counter_offer_continues_negotiation(self):
        conversation = make_conversation(summary_response="REJECTION", budget=211.5, seller_offer=225)
        conversation.conversation_history = [
            {"speaker": "Seller", "message": "I can do $225."},
            {"speaker": "Buyer", "message": "Could you do $205? If you can meet me there, I can buy today."},
        ]

        completed = conversation.evaluate_negotiation_state()

        self.assertFalse(completed)
        self.assertEqual(conversation.judge_events[-1]["guarded_label"], "CONTINUE")
        self.assertEqual(conversation.judge_events[-1]["override_reason"], "buyer_counter_offer")

    def test_price_extraction_empty_summary_uses_single_price_fallback(self):
        conversation = make_conversation(summary_response=None)

        price = conversation.extract_price_from_seller_message("I can do $225 for a quick deal.")

        self.assertEqual(price, 225)
        self.assertEqual(conversation.price_extraction_events[-1]["source"], "seller_message_single_currency")
        self.assertTrue(conversation.data_error)

    def test_save_conversation_includes_judge_events(self):
        conversation = make_conversation(summary_response="ACCEPTANCE", budget=300, seller_offer=225)
        conversation.conversation_history = [
            {"speaker": "Seller", "message": "I can do $225."},
            {"speaker": "Buyer", "message": "Deal at $225."},
        ]
        conversation.evaluate_negotiation_state()

        with tempfile.TemporaryDirectory() as tmp_dir:
            conversation.save_conversation(tmp_dir)
            output = Path(tmp_dir) / "product_1_exp_0.json"
            self.assertIn('"judge_events"', output.read_text(encoding="utf-8"))
            self.assertEqual(list(Path(tmp_dir).glob("*.tmp.*")), [])

    def test_buyer_intro_failure_is_saved_as_data_error(self):
        conversation = make_conversation(summary_response="CONTINUE")
        conversation.buyer_model = FailingModel("billing_or_quota")

        conversation.run_negotiation()

        self.assertTrue(conversation.data_error)
        self.assertEqual(conversation.negotiation_result, "model_error")
        self.assertEqual(conversation.error["role"], "buyer")

        with tempfile.TemporaryDirectory() as tmp_dir:
            conversation.save_conversation(tmp_dir)
            output = (Path(tmp_dir) / "product_1_exp_0.json").read_text(encoding="utf-8")
            self.assertIn('"data_error": true', output)

    def test_summary_run_fatal_stops_before_next_buyer_turn(self):
        conversation = make_conversation(summary_response="CONTINUE")
        buyer_model = StaticModel("Hello, can we discuss the price?")
        conversation.buyer_model = buyer_model
        conversation.seller_model = StaticModel("I can do $225.")
        conversation.summary_model = FailingModel("billing_or_quota")

        conversation.run_negotiation()

        self.assertTrue(conversation.data_error)
        self.assertTrue(conversation.run_fatal_error)
        self.assertEqual(conversation.negotiation_result, "model_error")
        self.assertEqual(buyer_model.response_calls, 1)
        self.assertEqual(buyer_model.chat_calls, 0)


if __name__ == "__main__":
    unittest.main()
