import argparse
import json
from pathlib import Path


CONSUMER_ELECTRONICS_IDS = list(range(41, 71))


def parse_ids(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    product_ids = set(parse_ids(args.ids))
    products = json.loads(source.read_text(encoding="utf-8"))
    selected = [product for product in products if int(product.get("id", -1)) in product_ids]

    missing = sorted(product_ids - {int(product.get("id", -1)) for product in selected})
    if missing:
        raise ValueError(f"Missing product ids in {source}: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} products to {output}")


if __name__ == "__main__":
    main()
