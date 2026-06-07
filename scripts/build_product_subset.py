import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import load_products, parse_int_csv


CONSUMER_ELECTRONICS_IDS = list(range(41, 71))


def main():
    parser = argparse.ArgumentParser(description="Build a curated product subset for A2A-NT experiments.")
    parser.add_argument("--source", default="dataset/products.json")
    parser.add_argument("--output", default="dataset/products_consumer_electronics.json")
    parser.add_argument(
        "--ids",
        default=",".join(str(item) for item in CONSUMER_ELECTRONICS_IDS),
        help="Comma-separated product ids to include",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    product_ids = set(parse_int_csv(args.ids))
    products = load_products(source)
    selected = [product for product in products if int(product.get("id", -1)) in product_ids]

    missing = sorted(product_ids - {int(product.get("id", -1)) for product in selected})
    if missing:
        raise ValueError(f"Missing product ids in {source}: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} products to {output}")


if __name__ == "__main__":
    main()
