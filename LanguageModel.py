import os
import json
import random
import logging
from openai import OpenAI
from Config import OPENAI_API_KEY, DEEPSEEK_API_KEY, ZHI_API_KEY, GOOGLE_API_KEY
import time
from openai import OpenAIError
from typing import List, Dict, Optional, Union
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageModel:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self._api_key_index = 0
        self._last_request_time = 0
        self._rate_limit_delay = 1.0  # seconds between requests
        self._setup_provider()
        
    def _setup_provider(self) -> None:
        """Setup the appropriate provider based on model name."""
        if "gpt" in self.model_name.lower() or (self.model_name.lower().count("o") > 0 and any(c.isdigit() for c in self.model_name[self.model_name.lower().index("o")+1:])):
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            self.api_keys = [OPENAI_API_KEY]
        elif "deepseek" in self.model_name.lower():
            self.api_keys = DEEPSEEK_API_KEY
            self.provider = "deepseek"
            self._setup_client_with_next_key()
        elif "qwen" in self.model_name.lower() or "llama" in self.model_name.lower():
            self.api_keys = ZHI_API_KEY
            self.provider = "zhizengzeng"
            self._setup_client_with_next_key()
        elif "gemini" in self.model_name.lower():
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
            self.provider = "google"
            self.api_keys = [GOOGLE_API_KEY]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
    def _setup_client_with_next_key(self) -> None:
        """Setup client with the next available API key."""
        if not self.api_keys:
            raise ValueError("No API keys available")
            
        self._api_key_index = (self._api_key_index + 1) % len(self.api_keys)
        selected_api_key = self.api_keys[self._api_key_index]
        
        if self.provider == "deepseek":
            self.client = OpenAI(api_key=selected_api_key, base_url="https://api.deepseek.com")
        elif self.provider == "zhizengzeng":
            self.client = OpenAI(api_key=selected_api_key, base_url="https://api.zhizengzeng.com/v1/chat/completions")
            
        logger.info(f"Using {self.provider} API key: {selected_api_key[:8]}...")
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _make_api_call(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Make API call with retry mechanism and rate limiting."""
        max_retries = 5
        retry_delay = 2  # seconds
        
        # Validate and fix message format for different providers
        if self.provider == "deepseek":
            # Make sure each message has the correct format for DeepSeek
            validated_messages = []
            for msg in messages:
                # DeepSeek requires specific role values
                if msg["role"] not in ["system", "user", "assistant", "tool"]:
                    # Convert unknown roles to user
                    msg = {"role": "user", "content": msg.get("content", "")}
                # Make sure content is a string and not None
                if "content" not in msg or msg["content"] is None:
                    msg["content"] = ""
                validated_messages.append(msg)
            messages = validated_messages
        elif self.provider == "zhizengzeng":
            # Make sure each message has the correct format for Zhizengzeng
            validated_messages = []
            for msg in messages:
                # Most APIs require specific role values
                if msg["role"] not in ["system", "user", "assistant"]:
                    # Convert unknown roles to user
                    msg = {"role": "user", "content": msg.get("content", "")}
                # Make sure content is a string and not None
                if "content" not in msg or msg["content"] is None:
                    msg["content"] = ""
                validated_messages.append(msg)
            messages = validated_messages
        elif self.provider == "google":
            # Convert messages to Gemini format
            validated_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Add system message as context
                    validated_messages.append({"role": "user", "content": f"System: {msg['content']}"})
                else:
                    validated_messages.append(msg)
            messages = validated_messages
            
        # Log messages for debugging
        logger.debug(f"Provider: {self.provider}, Model: {self.model_name}")
        logger.debug(f"Request messages: {json.dumps(messages, indent=2)}")
        
        for attempt in range(max_retries):
            try:
                self._enforce_rate_limit()
                
                if self.provider == "google":
                    # Handle Gemini API call
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[msg["content"] for msg in messages],
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens
                        }
                    )
                    return response.text
                else:
                    # Handle other providers
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    except Exception as e:
                        # If max_tokens fails, try with max_completion_tokens
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_completion_tokens=max_tokens
                        )
                    
                    # More robust error handling
                    if response is None:
                        raise ValueError("API returned None response")
                    if not hasattr(response, 'choices') or not response.choices:
                        raise ValueError("API response missing 'choices' attribute or has empty choices")
                    if not hasattr(response.choices[0], 'message'):
                        raise ValueError("API response first choice missing 'message' attribute")
                    if not hasattr(response.choices[0].message, 'content'):
                        raise ValueError("API response message missing 'content' attribute")
                    
                    return response.choices[0].message.content
                    
            except (OpenAIError, json.JSONDecodeError, Exception) as e:
                logger.error(f"Attempt {attempt + 1} failed with error: {str(e)}")
                
                # Try with a different API key on failure
                if self.provider in ["deepseek", "zhizengzeng"] and attempt < max_retries - 1:
                    self._setup_client_with_next_key()
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Returning default error message.")
                    return None
    
    def get_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Get a response from the language model."""
        if self.provider in ["openai", "deepseek", "zhizengzeng", "google"]:
            messages = [{"role": "user", "content": prompt}]
            return self._make_api_call(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_chat_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Get a response from the language model using chat format."""
        if self.provider in ["openai", "deepseek", "zhizengzeng", "google"]:
            return self._make_api_call(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
