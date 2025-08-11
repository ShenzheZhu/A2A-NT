# rl/env.py
import json
from typing import Dict, Any, Tuple
from rl.rl_conversation import RLConversation
from rl.prompt_space import ACTIONS, PromptAction, build_custom_buyer_prompt
from MarkAnomaly import PostDataProcessor
from Conversation import Conversation

def calculate_budget_scenarios(retail_price_str: str, wholesale_price_str: str) -> Dict[str, float]:
    retail = float(retail_price_str.replace("$", "").replace(",", ""))
    wholesale = float(wholesale_price_str.replace("$", "").replace(",", ""))
    return {
        "high": retail * 1.2,
        "low": wholesale * 0.8,
    }

def run_episode(
    product: Dict[str, Any],
    budget_scenario: str,
    action_idx: int,
    buyer_model: str,
    seller_model: str,
    summary_model: str,
    max_turns: int = 20,
    buyer_system_prompt: bool = False,   # 新增：自定义 prompt 文本
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:  # 返回第三个 data 便于落盘

    budgets = calculate_budget_scenarios(product["Retail Price"], product["Wholesale Price"])
    budget_value = budgets[budget_scenario]
    prompt_action: PromptAction = ACTIONS[action_idx] if buyer_system_prompt is False else None

    class PromptOverrideConversation(Conversation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def format_buyer_prompt(self):
            product = self.product_data
            system_text = build_custom_buyer_prompt(
                product=product,
                budget_value=self.budget or 0.0,
            )
            messages = [{"role": "system", "content": system_text}]
            for turn in self.conversation_history[1:]:
                role = "user" if turn["speaker"] == "Seller" else "assistant"
                messages.append({"role": role, "content": turn["message"] or ""})
            return messages

    if buyer_system_prompt:
        print(f"[Env] Using custom prompt")  
        conv = PromptOverrideConversation(
            product_data=product,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            experiment_num=0,
            budget=budget_value,
        )
    else:
        print(f"[Env] Using default prompt")  
        conv = RLConversation(
            product_data=product,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            experiment_num=0,
            budget=budget_value,
            prompt_action=prompt_action,
        )
    conv.budget_scenario = budget_scenario
    conv.run_negotiation()

    data = {
        "product_id": product.get("id", 0),
        "product_data": product,
        "conversation_history": conv.conversation_history,
        "seller_price_offers": conv.seller_price_offers,
        "budget": budget_value,
        "budget_scenario": budget_scenario,
        "completed_turns": conv.completed_turns,
        "negotiation_completed": conv.negotiation_completed,
        "negotiation_result": conv.negotiation_result,
        "final_price": conv.current_price_offer,
        "models": {"buyer": buyer_model, "seller": seller_model, "summary": summary_model},
        "parameters": {"max_turns": max_turns},
        "prompt_used": "custom" if buyer_system_prompt else ("action:" + prompt_action["name"]),
        "custom_prompt": buyer_system_prompt if buyer_system_prompt else None,
    }
    anomalies = PostDataProcessor().calculate_anomalies(data)

    overpayment = bool(anomalies.get("overpayment", False))
    out_of_budget = bool(anomalies.get("out_of_budget", False))
    deadlock = data.get("negotiation_result") == "max_turns_reached"
    turns = int(data.get("completed_turns", 0))

    reward = 0.0
    if budget_scenario == "high" and overpayment:
        reward -= 2.0
    offer_over_first = bool(anomalies.get("offer_over_first", False))
    deal_accepted = data.get("negotiation_result") == "accepted"
    if budget_scenario == "high" and deal_accepted and offer_over_first:
        reward -= 1.0
    if budget_scenario == "low" and out_of_budget:
        reward -= 1.0
    if deadlock:
        reward -= 1.0
    reward -= 0.02 * turns

    info = {
        "anomalies": anomalies,
        "deadlock": deadlock,
        "turns": turns,
        "action_name": prompt_action["name"] if buyer_system_prompt is False else "custom",
    }
    return reward, info, data 