import json
import logging
import os
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    import Config as _config
except ImportError:
    _config = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LITELLM_COMPLETION = None


MODEL_ALIASES = {
    "gpt-4.1": "openrouter/openai/gpt-4.1",
    "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
    "o3": "openrouter/openai/o3",
    "o4-mini": "openrouter/openai/o4-mini",
    "deepseek-r1": "openrouter/deepseek/deepseek-r1",
    "deepseek-v3": "openrouter/deepseek/deepseek-chat",
    "qwen2.5-7b-instruct": "openrouter/qwen/qwen-2.5-7b-instruct",
}

OPENROUTER_MODEL_PREFIXES = (
    "openai/",
    "anthropic/",
    "google/",
    "deepseek/",
    "qwen/",
    "x-ai/",
    "meta-llama/",
    "moonshotai/",
    "z-ai/",
)


def _config_value(name: str) -> Optional[str]:
    if _config is None:
        return None
    value = getattr(_config, name, None)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _env_or_config(name: str) -> Optional[str]:
    return os.getenv(name) or _config_value(name)


def _completion_function():
    global _LITELLM_COMPLETION
    if _LITELLM_COMPLETION is None:
        try:
            import litellm
        except ImportError as exc:
            raise ImportError("LiteLLM is required for live model calls. Run `pip install -r requirements.txt`.") from exc
        litellm.drop_params = True
        litellm.suppress_debug_info = True
        _LITELLM_COMPLETION = litellm.completion
    return _LITELLM_COMPLETION


def normalize_model_name(model_name: str) -> str:
    """Return the LiteLLM model identifier used for completion calls."""
    model = MODEL_ALIASES.get(model_name, model_name)
    if model.startswith("openrouter/"):
        return model
    if model.startswith(OPENROUTER_MODEL_PREFIXES):
        return f"openrouter/{model}"
    return model


def _validate_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    valid_roles = {"system", "user", "assistant", "tool"}
    validated = []
    for msg in messages:
        role = msg.get("role", "user")
        if role not in valid_roles:
            role = "user"
        content = msg.get("content", "")
        validated.append({"role": role, "content": "" if content is None else str(content)})
    return validated


class LanguageModel:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.requested_model_name = model_name
        self.model_name = normalize_model_name(model_name)
        self._last_request_time = 0.0
        self._rate_limit_delay = float(os.getenv("A2ANT_REQUEST_DELAY_SECONDS", "1.0"))
        self._max_retries = int(os.getenv("A2ANT_MAX_RETRIES", "5"))
        self._setup_credentials()

    def _setup_credentials(self) -> None:
        if self.model_name.startswith("openrouter/"):
            api_key = _env_or_config("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is required for OpenRouter LiteLLM calls")
            os.environ.setdefault("OPENROUTER_API_KEY", api_key)
            os.environ.setdefault("OR_SITE_URL", "https://shenzhezhu.github.io/A2A-NT/")
            os.environ.setdefault("OR_APP_NAME", "A2A-NT Negotiation Experiments")

    def _enforce_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        messages = _validate_messages(messages)
        retry_delay = 2.0

        for attempt in range(1, self._max_retries + 1):
            try:
                self._enforce_rate_limit()
                logger.debug("LiteLLM request model=%s messages=%s", self.model_name, json.dumps(messages))
                response = _completion_function()(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not response or not getattr(response, "choices", None):
                    raise ValueError("LiteLLM returned an empty response")

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LiteLLM response message content is empty")
                return content
            except Exception as exc:
                logger.error(
                    "LiteLLM attempt %s/%s failed for %s: %s",
                    attempt,
                    self._max_retries,
                    self.model_name,
                    exc,
                )
                if attempt == self._max_retries:
                    return None
                time.sleep(retry_delay)
                retry_delay *= 2
        return None

    def get_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        return self._make_api_call(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        return self._make_api_call(messages, temperature=temperature, max_tokens=max_tokens)
