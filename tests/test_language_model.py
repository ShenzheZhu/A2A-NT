import unittest
from types import SimpleNamespace

import LanguageModel
from LanguageModel import LanguageModel as ModelGate
from LanguageModel import ModelCallError, classify_model_error


class FakeProviderError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def fake_response(content, usage=None, hidden_params=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=usage,
        _hidden_params=hidden_params or {},
        model="provider/model-response",
    )


class LanguageModelTest(unittest.TestCase):
    def setUp(self):
        self._sleep = LanguageModel.time.sleep
        LanguageModel.time.sleep = lambda _seconds: None

    def tearDown(self):
        LanguageModel._LITELLM_COMPLETION = None
        LanguageModel.time.sleep = self._sleep

    def test_classify_billing_and_auth_errors(self):
        self.assertEqual(classify_model_error(FakeProviderError("Payment required", 402)), "billing_or_quota")
        self.assertEqual(classify_model_error(FakeProviderError("invalid api key", 401)), "auth")
        self.assertEqual(
            classify_model_error(FakeProviderError("Key limit exceeded (total limit)", 403)),
            "billing_or_quota",
        )

    def test_fatal_provider_error_does_not_retry(self):
        calls = []

        def fail_once(**_kwargs):
            calls.append("call")
            raise FakeProviderError("insufficient credits", 402)

        LanguageModel._LITELLM_COMPLETION = fail_once
        model = ModelGate("local-test-model")
        model._rate_limit_delay = 0
        model._max_retries = 3

        with self.assertRaises(ModelCallError) as ctx:
            model.get_response("hello")

        self.assertEqual(ctx.exception.category, "billing_or_quota")
        self.assertEqual(ctx.exception.attempts, 1)
        self.assertEqual(len(calls), 1)

    def test_transient_error_retries_then_returns_content(self):
        calls = []

        def flaky(**_kwargs):
            calls.append("call")
            if len(calls) == 1:
                raise TimeoutError("timed out")
            return fake_response("ok")

        LanguageModel._LITELLM_COMPLETION = flaky
        model = ModelGate("local-test-model")
        model._rate_limit_delay = 0
        model._max_retries = 2

        self.assertEqual(model.get_response("hello"), "ok")
        self.assertEqual(len(calls), 2)

    def test_success_records_usage_metadata(self):
        def respond(**_kwargs):
            return fake_response(
                "ok",
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
                hidden_params={"response_cost": 0.00123, "custom_llm_provider": "openrouter"},
            )

        LanguageModel._LITELLM_COMPLETION = respond
        model = ModelGate("local-test-model")
        model._rate_limit_delay = 0

        self.assertEqual(model.get_response("hello"), "ok")

        self.assertEqual(model.last_usage_event["requested_model"], "local-test-model")
        self.assertEqual(model.last_usage_event["model"], "local-test-model")
        self.assertEqual(model.last_usage_event["response_model"], "provider/model-response")
        self.assertEqual(model.last_usage_event["provider"], "openrouter")
        self.assertEqual(model.last_usage_event["prompt_tokens"], 11)
        self.assertEqual(model.last_usage_event["completion_tokens"], 7)
        self.assertEqual(model.last_usage_event["total_tokens"], 18)
        self.assertEqual(model.last_usage_event["estimated_cost_usd"], 0.00123)
        self.assertTrue(model.last_usage_event["usage_available"])

    def test_consume_usage_events_clears_last_call_events(self):
        def respond(**_kwargs):
            return fake_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 2})

        LanguageModel._LITELLM_COMPLETION = respond
        model = ModelGate("local-test-model")
        model._rate_limit_delay = 0

        model.get_response("hello")
        events = model.consume_usage_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["total_tokens"], 3)
        self.assertEqual(model.last_call_usage_events, [])
        self.assertIsNone(model.last_usage_event)

    def test_success_accepts_dict_response_shape(self):
        def respond(**_kwargs):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                "_hidden_params": {"response_cost": 0.005},
                "model": "dict-model-response",
            }

        LanguageModel._LITELLM_COMPLETION = respond
        model = ModelGate("local-test-model")
        model._rate_limit_delay = 0

        self.assertEqual(model.get_response("hello"), "ok")
        self.assertEqual(model.last_usage_event["response_model"], "dict-model-response")
        self.assertEqual(model.last_usage_event["total_tokens"], 5)
        self.assertEqual(model.last_usage_event["estimated_cost_usd"], 0.005)


if __name__ == "__main__":
    unittest.main()
