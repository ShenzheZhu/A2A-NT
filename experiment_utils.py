import json
import os
import re
from pathlib import Path


DEFAULT_BUDGET_SCENARIOS = ["wholesale", "mid", "retail"]
SUPPORTED_BUDGET_SCENARIOS = ["low", "wholesale", "mid", "retail", "high"]
NO_PRICE_PATTERN = re.compile(r"\b(none|no price|no clear price|not specified|not found|n/a)\b", re.IGNORECASE)
CURRENCY_PRICE_PATTERN = re.compile(r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
BARE_PRICE_PATTERN = re.compile(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\b")
RESULT_FILENAME_PATTERN = re.compile(r"^product_(?P<product_id>\d+)_exp_(?P<experiment_num>\d+)\.json$")
TERMINAL_REJECTION_PATTERN = re.compile(
    r"\b("
    r"(?:i|we)(?:'|’| wi)?ll have to pass|"
    r"(?:i|we)(?:'|’| wi)?ll pass|"
    r"(?:i|we) have to pass|"
    r"have to walk away|"
    r"walk away|"
    r"move on|"
    r"look elsewhere|"
    r"keep looking|"
    r"can't move forward|"
    r"can’t move forward|"
    r"cannot move forward|"
    r"can't proceed|"
    r"can’t proceed|"
    r"cannot proceed|"
    r"thanks for your time|"
    r"thank you for your time|"
    r"take care|"
    r"no deal"
    r")\b",
    re.IGNORECASE,
)
TERMINAL_CLOSURE_PATTERN = re.compile(
    r"\b("
    r"talk soon|"
    r"be in touch|"
    r"have (?:a )?(?:great|wonderful|nice) day|"
    r"you too|"
    r"thank you|"
    r"thanks for your (?:time|help)|"
    r"my pleasure|"
    r"you're welcome|"
    r"you are very welcome|"
    r"good luck|"
    r"best of luck|"
    r"take care|"
    r"bye|"
    r"goodbye|"
    r"glad i could help|"
    r"enjoy your new|"
    r"let me know if you need anything else"
    r")\b",
    re.IGNORECASE,
)
COUNTER_OFFER_PATTERN = re.compile(
    r"\b("
    r"would you|could you|can you|"
    r"if you can|how about|meet me|come down|counter|"
    r"my offer|best offer|final offer|"
    r"i can do|i can pay|i can stretch to|"
    r"take it or"
    r")\b",
    re.IGNORECASE,
)
ACTIVE_COUNTER_OFFER_PATTERN = re.compile(
    r"("
    r"\b(?:would|could|can) you\b.{0,80}\b(?:do|meet|accept|match|come down|make it|make that work)\b|"
    r"\bif you can\b.{0,120}\b(?:buy|pay|deal|close|take|proceed|move forward|purchase)\b|"
    r"\b(?:any chance|any possibility|possibility of|negotiate further|bring it within|under \$|provide|confirm)\b|"
    r"\b(?:deal at|final offer|take it or|i can pay|i can do|i can stretch to)\b"
    r")",
    re.IGNORECASE,
)
NEGATED_PRICE_OFFER_PATTERN = re.compile(
    r"("
    r"(?:can't|can’t|cannot|couldn't|couldn’t|won't|won’t|unable to|not able to)\s+"
    r"(?:make|do|go down to|accept|meet|sell(?: it)? for|offer)?\s*"
    r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)|"
    r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)\s+(?:is|was)\s+(?:too low|below|outside|beyond)|"
    r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?).{0,80}"
    r"(?:can't|can’t|cannot|couldn't|couldn’t|won't|won’t|unable to|not able to)\s+"
    r"(?:make|do|make it work|work)"
    r")",
    re.IGNORECASE,
)
POSITIVE_OFFER_NEAR_PRICE_PATTERN = re.compile(
    r"\b("
    r"best|lowest|absolute lowest|floor|"
    r"can\s+(?:do|offer|make|meet|accept|hold)|"
    r"could\s+(?:do|offer|come down|meet|accept)|"
    r"start(?:ing)?\s+(?:at|point)|"
    r"how about|"
    r"set it at|"
    r"try|"
    r"finalize at|"
    r"does this work|"
    r"what do you think|"
    r"thoughts on|"
    r"sound|"
    r"i(?:'|’| wi)?ll accept|"
    r"agree to|deal at|it's yours|it is yours|"
    r"for the (?:phone|laptop|tv|product|item)|"
    r"before (?:tax|shipping|fees)"
    r")\b",
    re.IGNORECASE,
)
PARTIAL_PAYMENT_CONTEXT_PATTERN = re.compile(
    r"\b("
    r"first|initial|deposit|down payment|installment|instalment|"
    r"partial payment|payment now"
    r")\b",
    re.IGNORECASE,
)
REMAINING_PAYMENT_CONTEXT_PATTERN = re.compile(
    r"\b("
    r"second|remaining|balance|remainder|rest|other half|second half|"
    r"before shipping|later|next payment"
    r")\b",
    re.IGNORECASE,
)


def safe_path_name(value):
    """Return a filesystem-safe label while preserving readable model names."""
    return re.sub(r"[^A-Za-z0-9._=-]+", "__", value).strip("_") or "model"


def parse_price(price_str):
    """Parse a dollar price string into a float."""
    return float(str(price_str).replace("$", "").replace(",", ""))


def looks_like_no_price(text):
    return text is None or bool(NO_PRICE_PATTERN.search(str(text)))


def price_candidates_from_text(text, allow_bare_number=False):
    """Extract possible price values from model output or currency-marked text."""
    if looks_like_no_price(text):
        return []

    text = str(text)
    matches = CURRENCY_PRICE_PATTERN.findall(text)
    if not matches and allow_bare_number:
        matches = BARE_PRICE_PATTERN.findall(text)

    candidates = []
    for match in matches:
        try:
            candidates.append(parse_price(match))
        except ValueError:
            continue
    return candidates


def extract_price_from_text(text, allow_bare_number=False):
    candidates = price_candidates_from_text(text, allow_bare_number=allow_bare_number)
    return candidates[0] if candidates else None


def prices_match(left, right, tolerance=0.01):
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


def text_has_terminal_rejection(text):
    return bool(TERMINAL_REJECTION_PATTERN.search(str(text or "")))


def text_has_terminal_closure(text):
    return bool(TERMINAL_CLOSURE_PATTERN.search(str(text or "")))


def text_is_terminal_emoji_closure(text):
    value = str(text or "").strip()
    return bool(value) and not re.search(r"[A-Za-z0-9$?]", value)


def text_has_active_counter_offer(text):
    return bool(ACTIVE_COUNTER_OFFER_PATTERN.search(str(text or "")))


def buyer_message_is_terminal_rejection(text):
    return text_has_terminal_rejection(text) and not text_has_active_counter_offer(text)


def buyer_has_counter_offer(text):
    value = str(text or "")
    if buyer_message_is_terminal_rejection(value):
        return False
    return bool(COUNTER_OFFER_PATTERN.search(value))


def result_has_terminal_not_closed(data):
    """Return True when a max-turn result is actually a terminal/farewell closure."""
    if data.get("negotiation_result") != "max_turns_reached":
        return False
    history = data.get("conversation_history")
    if not isinstance(history, list) or not history:
        return False

    tail = []
    for turn in history[-6:]:
        if isinstance(turn, dict):
            tail.append(str(turn.get("message", "")))
    if not tail:
        return False

    last_message = tail[-1]
    if buyer_message_is_terminal_rejection(last_message):
        return True

    recent_messages = tail[-3:]
    closure_count = sum(1 for message in tail if text_has_terminal_closure(message))
    terminal_like_count = sum(
        1 for message in tail
        if text_has_terminal_closure(message) or text_is_terminal_emoji_closure(message)
    )
    has_recent_counter = any(text_has_active_counter_offer(message) for message in recent_messages)
    last_is_text_closure = text_has_terminal_closure(last_message)
    last_is_emoji_closure = text_is_terminal_emoji_closure(last_message)
    return (
        (last_is_text_closure or last_is_emoji_closure)
        and "?" not in last_message
        and closure_count >= 2
        and (last_is_text_closure or terminal_like_count >= 3)
        and not has_recent_counter
    )


def _price_pattern_matches_value(match, price):
    values = [group for group in match.groups()[1:] if group]
    return any(prices_match(parse_price(value), price) for value in values)


def message_has_positive_offer_for_price(text, price, window=80):
    value = str(text or "")
    for match in CURRENCY_PRICE_PATTERN.finditer(value):
        candidate = parse_price(match.group(1))
        if not prices_match(candidate, price):
            continue
        start = max(0, match.start() - window)
        end = min(len(value), match.end() + window)
        if POSITIVE_OFFER_NEAR_PRICE_PATTERN.search(value[start:end]):
            return True
    return False


def price_is_rejected_without_positive_offer(text, price):
    value = str(text or "")
    for match in CURRENCY_PRICE_PATTERN.finditer(value):
        candidate = parse_price(match.group(1))
        if not prices_match(candidate, price):
            continue
        after_price = value[match.end():match.end() + 120]
        if re.search(
            r"\b(?:can't|can’t|cannot|couldn't|couldn’t|won't|won’t|unable to|not able to)\s+"
            r"(?:make it work|make|work|accept|do)\b",
            after_price,
            re.IGNORECASE,
        ):
            return True
    for match in NEGATED_PRICE_OFFER_PATTERN.finditer(value):
        if _price_pattern_matches_value(match, price):
            return not message_has_positive_offer_for_price(value, price)
    return False


def price_is_boundary_without_positive_offer(text, price):
    """Return True when a price is mentioned as a boundary, not an offer."""
    value = str(text or "")

    for match in CURRENCY_PRICE_PATTERN.finditer(value):
        candidate = parse_price(match.group(1))
        if not prices_match(candidate, price):
            continue
        start = max(0, match.start() - 80)
        end = min(len(value), match.end() + 80)
        window = value[start:end]
        if re.search(r"\b(?:beyond|above|over|at least|minimum|threshold)\b.{0,30}\$", window, re.IGNORECASE):
            return True
        if message_has_positive_offer_for_price(value, price):
            return False
        if re.search(
            r"\b("
            r"below (?:my |our )?(?:cost|wholesale)|"
            r"under (?:my |our )?(?:cost|wholesale)|"
            r"anything lower|selling at a loss|take a loss|losing money"
            r")\b",
            window,
            re.IGNORECASE,
        ):
            return True
    return False


def positive_seller_offer_events(data, max_price=None):
    """Return parsed seller offer events that are genuine positive offers."""
    events = []
    for event in data.get("price_extraction_events", []) or []:
        if not isinstance(event, dict):
            continue
        price = event.get("price")
        if price is None:
            continue
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        if max_price is not None and price_value > float(max_price):
            continue
        seller_message = event.get("seller_message")
        if (
            not price_is_rejected_without_positive_offer(seller_message, price_value)
            and not price_is_boundary_without_positive_offer(seller_message, price_value)
            and message_has_positive_offer_for_price(seller_message, price_value)
        ):
            events.append({**event, "price": price_value})
    return events


def result_has_false_feasible_offer_extraction(data):
    """Return True when a rejected row only looks feasible due to price extraction."""
    if data.get("negotiation_result") != "rejected":
        return False
    offers = data.get("seller_price_offers")
    if not isinstance(offers, list) or not offers or "budget" not in data:
        return False
    try:
        budget = float(data["budget"])
        last_offer = float(offers[-1])
    except (TypeError, ValueError):
        return False
    if last_offer >= budget:
        return False

    matched_index = None
    matched_event = None
    price_events = data.get("price_extraction_events", []) or []
    for index in range(len(price_events) - 1, -1, -1):
        event = price_events[index]
        if not isinstance(event, dict):
            continue
        price = event.get("price")
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        if prices_match(price_value, last_offer):
            matched_index = index
            matched_event = event
            break

    if matched_event is None:
        return False

    seller_message = matched_event.get("seller_message")
    rejected_or_boundary = (
        price_is_rejected_without_positive_offer(seller_message, last_offer)
        or price_is_boundary_without_positive_offer(seller_message, last_offer)
    )
    if not rejected_or_boundary:
        return False

    earlier_events = price_events[:matched_index]
    for event in reversed(earlier_events):
        if not isinstance(event, dict):
            continue
        price = event.get("price")
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        if price_value <= budget and message_has_positive_offer_for_price(event.get("seller_message"), price_value):
            return False
    return True


def result_has_partial_payment_price_extraction(data):
    """Return True when the extracted deal price is only a partial payment."""
    if data.get("negotiation_result") != "accepted":
        return False
    offers = data.get("seller_price_offers")
    if not isinstance(offers, list) or not offers:
        return False
    try:
        last_offer = float(offers[-1])
    except (TypeError, ValueError):
        return False

    candidate_messages = []
    for event in reversed(data.get("price_extraction_events", []) or []):
        if not isinstance(event, dict):
            continue
        try:
            price_value = float(event.get("price"))
        except (TypeError, ValueError):
            continue
        if prices_match(price_value, last_offer):
            candidate_messages.append(str(event.get("seller_message", "")))
            break

    for turn in reversed(data.get("conversation_history", []) or []):
        if isinstance(turn, dict) and turn.get("speaker") == "Seller":
            candidate_messages.append(str(turn.get("message", "")))
            break

    for message in candidate_messages:
        if not any(prices_match(price, last_offer) for price in price_candidates_from_text(message)):
            continue
        if (
            PARTIAL_PAYMENT_CONTEXT_PATTERN.search(message)
            and REMAINING_PAYMENT_CONTEXT_PATTERN.search(message)
        ):
            return True
    return False


def parse_csv(value):
    if value is None:
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value):
    return [int(item) for item in parse_csv(value) or []]


def calculate_budget_scenarios(retail_price_str, wholesale_price_str):
    """Calculate the supported budget scenarios."""
    retail_price = parse_price(retail_price_str)
    wholesale_price = parse_price(wholesale_price_str)

    return {
        "high": retail_price * 1.2,  # Retail Price * 1.2
        "retail": retail_price,  # Retail Price
        "mid": (retail_price + wholesale_price) / 2,  # (Retail Price + Wholesale Price) / 2
        "wholesale": wholesale_price,  # Wholesale Price
        "low": wholesale_price * 0.8,  # Wholesale Price * 0.8
    }


def select_budget_scenarios(all_budgets, budget_scenarios=None):
    """Return requested budget scenarios in caller-provided order."""
    budget_scenarios = budget_scenarios or DEFAULT_BUDGET_SCENARIOS
    missing_budgets = [name for name in budget_scenarios if name not in all_budgets]
    if missing_budgets:
        raise ValueError(f"Unsupported budget scenarios: {', '.join(missing_budgets)}")
    return {name: all_budgets[name] for name in budget_scenarios}


def load_products(products_file):
    products = json.loads(Path(products_file).read_text(encoding="utf-8"))
    if not isinstance(products, list):
        raise ValueError(f"{products_file} must contain a list of products")
    return products


def select_products(products, start_index=0, product_limit=None, product_ids=None):
    """Select products while preserving their original dataset indices."""
    product_ids_set = set(product_ids) if product_ids else None
    indexed_products = list(enumerate(products[start_index:], start=start_index))

    if product_ids_set is not None:
        indexed_products = [
            (index, product)
            for index, product in indexed_products
            if int(product.get("id", -1)) in product_ids_set
        ]

    if product_limit is not None:
        indexed_products = indexed_products[:product_limit]

    return indexed_products


def result_base_dir(output_dir, seller_model, buyer_model, product_id):
    seller_dir = safe_path_name(seller_model)
    buyer_dir = safe_path_name(buyer_model)
    return os.path.join(output_dir, f"seller_{seller_dir}", buyer_dir, f"product_{product_id}")


def result_file_experiment_num(path):
    match = RESULT_FILENAME_PATTERN.match(Path(path).name)
    if not match:
        return None
    return int(match.group("experiment_num"))


def iter_result_files(result_dir, product_id=None):
    result_path = Path(result_dir)
    if not result_path.exists():
        return []

    files = []
    for path in result_path.rglob("product_*_exp_*.json"):
        match = RESULT_FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        if product_id is not None and int(match.group("product_id")) != int(product_id):
            continue
        files.append(path)
    return sorted(files)


def result_has_system_data_error(data):
    """Return True when a result should be excluded from formal metrics."""
    return bool(
        data.get("data_error", False)
        or data.get("system_data_error", False)
        or data.get("negotiation_result") == "model_error"
        or data.get("run_fatal_error", False)
        or data.get("price_scale_warning", False)
        or data.get("terminal_not_closed", False)
        or result_has_terminal_not_closed(data)
        or result_has_false_feasible_offer_extraction(data)
        or result_has_partial_payment_price_extraction(data)
    )


def inspect_result_file(path, include_error_files=False):
    """Return a compact validity record for a result JSON file."""
    info = {
        "path": str(path),
        "parseable": False,
        "valid": False,
        "data_error": None,
        "system_data_error": None,
        "error": None,
    }
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        info["error"] = str(exc)
        return info

    info["parseable"] = True
    info["data_error"] = bool(data.get("data_error", False))
    info["system_data_error"] = result_has_system_data_error(data)
    has_required_shape = (
        isinstance(data.get("conversation_history"), list)
        and "product_id" in data
        and "experiment_num" in data
    )
    info["valid"] = has_required_shape and (include_error_files or not info["system_data_error"])
    if not has_required_shape:
        info["error"] = "missing_required_result_fields"
    return info


def _numeric(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_usage_bucket():
    return {
        "calls": 0,
        "usage_available_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def _add_usage_to_bucket(bucket, event):
    bucket["calls"] += 1
    if event.get("usage_available"):
        bucket["usage_available_calls"] += 1

    prompt_tokens = _numeric(event.get("prompt_tokens"))
    completion_tokens = _numeric(event.get("completion_tokens"))
    total_tokens = _numeric(event.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    for field, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("total_tokens", total_tokens),
    ):
        if value is not None:
            bucket[field] += int(value)

    cost = _numeric(event.get("estimated_cost_usd"))
    if cost is not None:
        bucket["estimated_cost_usd"] += cost


def summarize_usage_events(events):
    """Aggregate non-sensitive provider usage events."""
    summary = _empty_usage_bucket()
    summary["by_model"] = {}
    summary["by_role"] = {}

    for raw_event in events or []:
        if not isinstance(raw_event, dict):
            continue
        _add_usage_to_bucket(summary, raw_event)

        model = raw_event.get("model") or raw_event.get("requested_model") or "unknown"
        role = raw_event.get("role") or "unknown"
        _add_usage_to_bucket(summary["by_model"].setdefault(model, _empty_usage_bucket()), raw_event)
        _add_usage_to_bucket(summary["by_role"].setdefault(role, _empty_usage_bucket()), raw_event)

    summary["estimated_cost_usd"] = round(summary["estimated_cost_usd"], 8)
    for group in (summary["by_model"], summary["by_role"]):
        for bucket in group.values():
            bucket["estimated_cost_usd"] = round(bucket["estimated_cost_usd"], 8)
    return summary


def aggregate_usage_from_results(result_dir, include_error_files=True):
    """Aggregate usage events from parseable result JSON files."""
    events = []
    result_files = 0
    files_with_usage = 0

    for path in iter_result_files(result_dir):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result_has_system_data_error(data) and not include_error_files:
            continue
        result_files += 1

        file_events = data.get("usage_events")
        if isinstance(file_events, list) and file_events:
            files_with_usage += 1
            events.extend(file_events)

    summary = summarize_usage_events(events)
    summary["result_files"] = result_files
    summary["files_with_usage"] = files_with_usage
    return summary


def count_valid_results(result_dir, product_id=None, include_error_files=False):
    return sum(
        1
        for path in iter_result_files(result_dir, product_id=product_id)
        if inspect_result_file(path, include_error_files=include_error_files)["valid"]
    )


def next_experiment_number(result_dir, product_id=None):
    exp_nums = [
        result_file_experiment_num(path)
        for path in iter_result_files(result_dir, product_id=product_id)
    ]
    exp_nums = [num for num in exp_nums if num is not None]
    return max(exp_nums) + 1 if exp_nums else 0
