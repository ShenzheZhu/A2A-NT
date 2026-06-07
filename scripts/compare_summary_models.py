import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation import normalize_judge_label
from LanguageModel import LanguageModel, ModelCallError
from experiment_utils import extract_price_from_text, looks_like_no_price, parse_csv

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


PRICE_FIXTURES = [
    {
        "id": "single_offer_with_addon",
        "seller_message": "I can offer the laptop for $1,099. The setup package is $49 if you want it.",
        "expected": 1099.0,
    },
    {
        "id": "no_price",
        "seller_message": "Thanks for your interest. Tell me which features matter most to you.",
        "expected": None,
    },
    {
        "id": "usd_prefix",
        "seller_message": "I can meet you at USD 850 for the tablet today.",
        "expected": 850.0,
    },
    {
        "id": "warranty_not_main_price",
        "seller_message": "The phone would be $720, and an optional protection plan is $90.",
        "expected": 720.0,
    },
]


JUDGE_FIXTURES = [
    {
        "id": "clear_acceptance",
        "seller_message": "I can do $225.",
        "buyer_message": "Deal at $225. I am ready to pay now.",
        "expected": "ACCEPTANCE",
    },
    {
        "id": "counter_offer",
        "seller_message": "I can do $225.",
        "buyer_message": "Could you do $205? If so, I can buy today.",
        "expected": "CONTINUE",
    },
    {
        "id": "clear_rejection",
        "seller_message": "The best I can do is $225.",
        "buyer_message": "I cannot proceed at that price.",
        "expected": "REJECTION",
    },
    {
        "id": "conditional_acceptance",
        "seller_message": "I can do $225.",
        "buyer_message": "That works only if delivery and tax are included.",
        "expected": "CONTINUE",
    },
    {
        "id": "explicit_same_price_acceptance",
        "seller_message": "I can sell it for $850.",
        "buyer_message": "Yes, I can do $850. Let's move forward.",
        "expected": "ACCEPTANCE",
    },
    {
        "id": "accept_offer_without_price_repeat",
        "seller_message": "My best offer is $1,099.",
        "buyer_message": "That price works for me. I accept.",
        "expected": "ACCEPTANCE",
    },
    {
        "id": "okay_price_it_is",
        "seller_message": "I can come down to $640.",
        "buyer_message": "Okay, $640 it is.",
        "expected": "ACCEPTANCE",
    },
    {
        "id": "accept_with_payment_intent",
        "seller_message": "I can do $720 today.",
        "buyer_message": "I accept your $720 offer and can pay today.",
        "expected": "ACCEPTANCE",
    },
    {
        "id": "price_mismatch_counter_offer",
        "seller_message": "I can do $900.",
        "buyer_message": "I can buy it today for $800.",
        "expected": "CONTINUE",
    },
    {
        "id": "if_you_can_counter_offer",
        "seller_message": "The lowest I can go is $500.",
        "buyer_message": "If you can do $450, we have a deal.",
        "expected": "CONTINUE",
    },
    {
        "id": "conditional_no_extra_fees",
        "seller_message": "I can do $1,250.",
        "buyer_message": "I agree as long as there are no extra fees.",
        "expected": "CONTINUE",
    },
    {
        "id": "conditional_shipping",
        "seller_message": "I can do $300.",
        "buyer_message": "Sounds good, but only if shipping is included.",
        "expected": "CONTINUE",
    },
    {
        "id": "agree_word_then_counter",
        "seller_message": "I can do $225.",
        "buyer_message": "I agree $225 is fair, but could you do $210?",
        "expected": "CONTINUE",
    },
    {
        "id": "ask_warranty",
        "seller_message": "I can sell it for $700.",
        "buyer_message": "Can you tell me what warranty is included?",
        "expected": "CONTINUE",
    },
    {
        "id": "ask_best_price",
        "seller_message": "The listed price is $999.",
        "buyer_message": "What's your best price on this?",
        "expected": "CONTINUE",
    },
    {
        "id": "thinking_not_done",
        "seller_message": "I can do $620.",
        "buyer_message": "I need to think about it. Can you do any better?",
        "expected": "CONTINUE",
    },
    {
        "id": "too_expensive_counter",
        "seller_message": "I can do $225.",
        "buyer_message": "Thanks, but $225 is too much. If you can reach $205, I can buy.",
        "expected": "CONTINUE",
    },
    {
        "id": "explicit_pass",
        "seller_message": "The best I can do is $780.",
        "buyer_message": "No thanks, I will pass.",
        "expected": "REJECTION",
    },
    {
        "id": "above_budget_rejection",
        "seller_message": "I can do $1,050.",
        "buyer_message": "This is above my budget, so I cannot proceed.",
        "expected": "REJECTION",
    },
    {
        "id": "walk_away",
        "seller_message": "I cannot go below $450.",
        "buyer_message": "Unfortunately I need to walk away from this deal.",
        "expected": "REJECTION",
    },
    {
        "id": "cannot_afford",
        "seller_message": "The final price is $1,100.",
        "buyer_message": "I cannot afford that price.",
        "expected": "REJECTION",
    },
    {
        "id": "stop_negotiating",
        "seller_message": "I can do $375.",
        "buyer_message": "I don't want to continue negotiating.",
        "expected": "REJECTION",
    },
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_price_prompt(seller_message):
    return f"""Extract the price offered by the seller in the following message.
Return only the numerical price (with currency symbol) if there is a clear price offer.
If there is no clear price offer, return 'None'.

IMPORTANT: Only focus on the price of the product itself. Ignore any prices for add-ons like insurance, warranty, gifts, or accessories. Only extract the current offer price for the main product.

Now for the current message, please STRICTLY ONLY return the price with $ symbol, no other text:
Seller's message:
{seller_message}
Price:"""


def build_judge_prompt(buyer_message, seller_message):
    return f"""
You are evaluating if the buyer's latest response indicates agreement to a deal.

Buyer's latest message: "{buyer_message}"
Seller's latest message: "{seller_message}"

Determine if the buyer's response indicates:
A. ACCEPTANCE - The buyer has clearly agreed to the deal
B. REJECTION - The buyer has clearly rejected the deal or indicated they cannot proceed
C. CONTINUE - The buyer wants to continue negotiating

Please strictly output a single line containing just one of: ACCEPTANCE, REJECTION, or CONTINUE.
"""


def score_price_response(response, expected):
    text = "" if response is None else str(response).strip()
    if expected is None:
        return looks_like_no_price(text) or extract_price_from_text(text, allow_bare_number=True) is None
    parsed = extract_price_from_text(text, allow_bare_number=True)
    return parsed is not None and abs(parsed - expected) <= 0.01


def timed_model_call(model, prompt, temperature, max_tokens):
    started = perf_counter()
    response = model.get_response(prompt, temperature=temperature, max_tokens=max_tokens)
    elapsed = perf_counter() - started
    return response, elapsed, model.last_call_attempts, list(model.last_error_categories)


def compare_model(model_name, task="all"):
    model = LanguageModel(model_name=model_name)
    price_results = []
    judge_results = []

    if task in {"all", "price"}:
        for fixture in PRICE_FIXTURES:
            try:
                response, elapsed, attempts, error_categories = timed_model_call(
                    model,
                    build_price_prompt(fixture["seller_message"]),
                    temperature=0,
                    max_tokens=40,
                )
                error = None
            except ModelCallError as exc:
                response = None
                elapsed = None
                attempts = exc.attempts
                error_categories = list(model.last_error_categories)
                error = exc.to_dict()
            correct = False if error else score_price_response(response, fixture["expected"])
            price_results.append(
                {
                    "id": fixture["id"],
                    "expected": fixture["expected"],
                    "response": response,
                    "error": error,
                    "correct": correct,
                    "elapsed_seconds": elapsed,
                    "attempts": attempts,
                    "error_categories": error_categories,
                }
            )

    if task in {"all", "judge"}:
        for fixture in JUDGE_FIXTURES:
            try:
                response, elapsed, attempts, error_categories = timed_model_call(
                    model,
                    build_judge_prompt(fixture["buyer_message"], fixture["seller_message"]),
                    temperature=0,
                    max_tokens=20,
                )
                error = None
            except ModelCallError as exc:
                response = None
                elapsed = None
                attempts = exc.attempts
                error_categories = list(model.last_error_categories)
                error = exc.to_dict()
            label, warning = normalize_judge_label(response)
            correct = False if error else label == fixture["expected"]
            judge_results.append(
                {
                    "id": fixture["id"],
                    "expected": fixture["expected"],
                    "response": response,
                    "label": label,
                    "warning": warning,
                    "error": error,
                    "correct": correct,
                    "elapsed_seconds": elapsed,
                    "attempts": attempts,
                    "error_categories": error_categories,
                }
            )

    total = len(price_results) + len(judge_results)
    correct = sum(item["correct"] for item in price_results + judge_results)
    judge_elapsed = [
        item["elapsed_seconds"]
        for item in judge_results
        if item["elapsed_seconds"] is not None
    ]
    return {
        "model": model_name,
        "price_correct": sum(item["correct"] for item in price_results),
        "price_total": len(price_results),
        "judge_correct": sum(item["correct"] for item in judge_results),
        "judge_total": len(judge_results),
        "judge_invalid": sum(1 for item in judge_results if item.get("warning")),
        "judge_retried": sum(1 for item in judge_results if item.get("attempts", 0) > 1),
        "judge_avg_latency_seconds": sum(judge_elapsed) / len(judge_elapsed) if judge_elapsed else None,
        "overall_correct": correct,
        "overall_total": total,
        "overall_accuracy": correct / total if total else None,
        "price_results": price_results,
        "judge_results": judge_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare candidate summary/judge models on fixed A2A-NT fixtures.")
    parser.add_argument(
        "--models",
        default="openai/gpt-5.4-mini,deepseek/deepseek-v4-flash",
        help="Comma-separated model IDs",
    )
    parser.add_argument("--task", choices=["all", "price", "judge"], default="all")
    parser.add_argument("--live", action="store_true", help="Call model APIs. Omit for fixture-only dry run.")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    models = parse_csv(args.models)
    print(f"Models: {', '.join(models)}")
    print(f"Task: {args.task}")
    print(f"Fixtures: {len(PRICE_FIXTURES)} price + {len(JUDGE_FIXTURES)} judge")

    if not args.live:
        print("[dry-run] add --live to call providers")
        return

    payload = {
        "created_at": utc_now(),
        "models": models,
        "task": args.task,
        "price_fixture_count": len(PRICE_FIXTURES),
        "judge_fixture_count": len(JUDGE_FIXTURES),
        "results": [compare_model(model_name, task=args.task) for model_name in models],
    }

    for result in payload["results"]:
        print(
            f"{result['model']}: "
            f"{result['overall_correct']}/{result['overall_total']} "
            f"({result['overall_accuracy']:.2%}) | "
            f"price {result['price_correct']}/{result['price_total']} | "
            f"judge {result['judge_correct']}/{result['judge_total']} | "
            f"judge invalid {result['judge_invalid']} | "
            f"judge retried {result['judge_retried']}"
        )

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"artifacts/summary_model_compare_{stamp}.json"
    output_path = ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
