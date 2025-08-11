# rl/rl_conversation.py
from typing import Any, Dict, List
from Conversation import Conversation
from rl.prompt_space import build_buyer_system_prompt, PromptAction

class RLConversation(Conversation):
    def __init__(self, *args, prompt_action: PromptAction, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_action = prompt_action

    def format_buyer_prompt(self) -> List[Dict[str, str]]:
        product = self.product_data
        system_text = build_buyer_system_prompt(
            product=product,
            budget_value=self.budget or 0.0,
            action=self.prompt_action,
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
        for turn in self.conversation_history[1:]:
            if turn["speaker"] == "Seller":
                messages.append({"role": "user", "content": turn["message"] if turn["message"] else ""})
            else:
                messages.append({"role": "assistant", "content": turn["message"] if turn["message"] else ""})
        return messages