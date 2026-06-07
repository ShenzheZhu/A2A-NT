import tempfile
import unittest
from pathlib import Path

from Conversation import Conversation, normalize_judge_label


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


if __name__ == "__main__":
    unittest.main()
