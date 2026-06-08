import unittest
from types import SimpleNamespace

import LanguageModel
from LanguageModel import LanguageModel as ModelGate
from LanguageModel import ModelCallError, classify_model_error


class FakeProviderError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def fake_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
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


if __name__ == "__main__":
    unittest.main()
