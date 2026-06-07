import argparse
import json
import urllib.request
from pathlib import Path


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def load_plan(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def openrouter_id(model):
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


def model_entries(plan):
    for group_name in ("frontier_models", "bridge_models"):
        for entry in plan.get(group_name, []):
            yield group_name, entry


def fetch_model_ids():
    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {item["id"] for item in payload["data"]}


def main():
    parser = argparse.ArgumentParser(description="Validate configured OpenRouter model ids.")
    parser.add_argument("--config", default="configs/model_refresh.json")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()

    plan = load_plan(args.config)
    available = fetch_model_ids()
    missing = []

    for group_name, entry in model_entries(plan):
        if not args.include_disabled and not entry.get("enabled", True):
            print(f"SKIP disabled {group_name}: {entry['label']} ({entry['model']})")
            continue
        model_id = openrouter_id(entry["model"])
        if model_id in available:
            print(f"OK   {group_name}: {entry['label']} ({model_id})")
        else:
            print(f"MISS {group_name}: {entry['label']} ({model_id})")
            missing.append((group_name, entry["label"], model_id))

    if missing:
        raise SystemExit(f"{len(missing)} configured model ids were not found in OpenRouter")


if __name__ == "__main__":
    main()
