import json
import os
import re
from pathlib import Path


DEFAULT_BUDGET_SCENARIOS = ["wholesale", "mid", "retail"]
SUPPORTED_BUDGET_SCENARIOS = ["low", "wholesale", "mid", "retail", "high"]
NO_PRICE_PATTERN = re.compile(r"\b(none|no price|no clear price|not specified|not found|n/a)\b", re.IGNORECASE)
CURRENCY_PRICE_PATTERN = re.compile(r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
BARE_PRICE_PATTERN = re.compile(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\b")


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
