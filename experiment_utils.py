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
    r"\b(?:deal at|final offer|take it or|i can pay|i can do|i can stretch to)\b"
    r")",
    re.IGNORECASE,
)
NEGATED_PRICE_OFFER_PATTERN = re.compile(
    r"("
    r"(?:can't|can’t|cannot|couldn't|couldn’t|won't|won’t|unable to|not able to)\s+"
    r"(?:make|do|go down to|accept|meet|sell(?: it)? for|offer)?\s*"
    r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)|"
    r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)\s+(?:is|was)\s+(?:too low|below|outside|beyond)"
    r")",
    re.IGNORECASE,
)
POSITIVE_OFFER_NEAR_PRICE_PATTERN = re.compile(
    r"\b("
    r"best|lowest|absolute lowest|floor|"
    r"can\s+(?:do|offer|make|meet|accept|hold)|"
    r"could\s+(?:do|offer|come down|meet|accept)|"
    r"i(?:'|’| wi)?ll accept|"
    r"agree to|deal at|it's yours|it is yours|"
    r"for the (?:phone|laptop|tv|product|item)|"
    r"before (?:tax|shipping|fees)"
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


def text_has_active_counter_offer(text):
    return bool(ACTIVE_COUNTER_OFFER_PATTERN.search(str(text or "")))


def buyer_message_is_terminal_rejection(text):
    return text_has_terminal_rejection(text) and not text_has_active_counter_offer(text)


def buyer_has_counter_offer(text):
    value = str(text or "")
    if buyer_message_is_terminal_rejection(value):
        return False
    return bool(COUNTER_OFFER_PATTERN.search(value))


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
    for match in NEGATED_PRICE_OFFER_PATTERN.finditer(value):
        if _price_pattern_matches_value(match, price):
            return not message_has_positive_offer_for_price(value, price)
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


def inspect_result_file(path, include_error_files=False):
    """Return a compact validity record for a result JSON file."""
    info = {
        "path": str(path),
        "parseable": False,
        "valid": False,
        "data_error": None,
        "error": None,
    }
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        info["error"] = str(exc)
        return info

    info["parseable"] = True
    info["data_error"] = bool(data.get("data_error", False))
    has_required_shape = (
        isinstance(data.get("conversation_history"), list)
        and "product_id" in data
        and "experiment_num" in data
    )
    info["valid"] = has_required_shape and (include_error_files or not info["data_error"])
    if not has_required_shape:
        info["error"] = "missing_required_result_fields"
    return info


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
