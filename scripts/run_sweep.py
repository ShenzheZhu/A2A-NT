import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import count_valid_results, load_products, parse_csv, safe_path_name


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_plan(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def enabled_models(entries):
    return [entry for entry in entries if entry.get("enabled", True)]


def disabled_models(entries):
    return [entry for entry in entries if not entry.get("enabled", True)]


def build_pairs(plan, mode):
    frontier = enabled_models(plan.get("frontier_models", []))
    bridge = enabled_models(plan.get("bridge_models", []))
    pairs = []

    if mode in {"frontier-grid", "all"}:
        for seller in frontier:
            for buyer in frontier:
                pairs.append((seller, buyer, "frontier-grid"))

    if mode in {"bridge", "all"}:
        for frontier_model in frontier:
            for bridge_model in bridge:
                pairs.append((frontier_model, bridge_model, "bridge-new-seller"))
                pairs.append((bridge_model, frontier_model, "bridge-new-buyer"))

    return pairs


def count_products(products_file):
    return len(load_products(products_file))


def build_pair_command(plan, seller, buyer, args):
    cmd = [
        sys.executable,
        "main.py",
        "--products-file",
        args.products_file or plan["products_file"],
        "--buyer-model",
        buyer["model"],
        "--seller-model",
        seller["model"],
        "--summary-model",
        args.summary_model or plan["summary_model"],
        "--max-turns",
        str(args.max_turns or plan["max_turns"]),
        "--num-experiments",
        str(args.num_experiments or plan["num_experiments"]),
        "--budget-scenarios",
        ",".join(args.budgets or plan["budgets"]),
        "--output-dir",
        args.output_dir or plan["output_dir"],
        "--append",
    ]
    judge_confirmation_model = args.judge_confirmation_model or plan.get("judge_confirmation_model")
    if judge_confirmation_model:
        cmd.extend(["--judge-confirmation-model", judge_confirmation_model])
    if args.product_limit is not None:
        cmd.extend(["--product-limit", str(args.product_limit)])
    if args.include_error_files:
        cmd.append("--include-error-files")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def pair_id(seller, buyer, kind):
    return (
        f"{kind}__seller_{safe_path_name(seller['model'])}"
        f"__buyer_{safe_path_name(buyer['model'])}"
    )


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


class SweepManifest:
    def __init__(self, path, initial_payload):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.payload = dict(initial_payload)
        self.payload.setdefault("running_pairs", {})
        self.payload.setdefault("completed_pairs", [])
        self.payload.setdefault("failed_pairs", [])
        self.payload.setdefault("events", [])
        self.write()

    def write(self):
        self.payload["updated_at"] = utc_now()
        write_json_atomic(self.path, self.payload)

    def update(self, **kwargs):
        with self.lock:
            self.payload.update(kwargs)
            self.write()

    def event(self, event_type, **kwargs):
        with self.lock:
            event = {"type": event_type, "time": utc_now()}
            event.update(kwargs)
            self.payload["events"].append(event)
            self.write()

    def pair_started(self, identifier, seller, buyer, kind, log_path):
        with self.lock:
            self.payload["running_pairs"][identifier] = {
                "seller": seller["model"],
                "buyer": buyer["model"],
                "kind": kind,
                "log_path": str(log_path),
                "started_at": utc_now(),
            }
            self.write()

    def pair_finished(self, identifier, result):
        with self.lock:
            self.payload["running_pairs"].pop(identifier, None)
            if result["returncode"] == 0:
                self.payload["completed_pairs"].append(result)
            else:
                self.payload["failed_pairs"].append(result)
            self.payload["completed_result_json_count"] = count_valid_results(
                self.payload["output_dir"],
                include_error_files=False,
            )
            self.write()


def run_pair(plan, seller, buyer, kind, args, manifest=None, session_dir=None):
    cmd = build_pair_command(plan, seller, buyer, args)
    identifier = pair_id(seller, buyer, kind)
    log_path = None

    print(f"\n=== {seller['label']} seller vs {buyer['label']} buyer ({kind}) ===")
    print(" ".join(cmd))
    if args.dry_run:
        return {"pair_id": identifier, "returncode": 0, "dry_run": True}

    if session_dir is not None:
        log_dir = Path(session_dir) / "pairs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{identifier}.log"

    if manifest is not None:
        manifest.pair_started(identifier, seller, buyer, kind, log_path)

    started_at = utc_now()
    if log_path is None:
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
    else:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )

    result = {
        "pair_id": identifier,
        "seller": seller["model"],
        "buyer": buyer["model"],
        "kind": kind,
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": utc_now(),
        "log_path": str(log_path) if log_path else None,
    }
    if manifest is not None:
        manifest.pair_finished(identifier, result)

    if completed.returncode != 0 and not args.continue_on_error:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return result


def make_session(plan, args, pairs, products_file, product_count, budgets, num_experiments):
    run_id = args.run_id or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    session_dir = ROOT / "logs" / "run_sessions" / run_id
    manifest_path = Path(args.manifest_path) if args.manifest_path else session_dir / "manifest.json"
    output_dir = args.output_dir or plan["output_dir"]
    payload = {
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "plan_name": plan["name"],
        "mode": args.mode,
        "products_file": products_file,
        "product_count": product_count,
        "budgets": budgets,
        "num_experiments": num_experiments,
        "pair_count": len(pairs),
        "expected_conversation_cells": len(pairs) * product_count * len(budgets) * num_experiments,
        "completed_result_json_count": count_valid_results(output_dir, include_error_files=False),
        "output_dir": output_dir,
        "parallel_workers": args.parallel_workers,
        "continue_on_error": args.continue_on_error,
        "manifest_path": str(manifest_path),
        "session_dir": str(session_dir),
    }
    return run_id, session_dir, SweepManifest(manifest_path, payload)


def run_pairs(plan, pairs, args, manifest=None, session_dir=None):
    if args.parallel_workers <= 1 or args.dry_run:
        results = []
        for seller, buyer, kind in pairs:
            results.append(run_pair(plan, seller, buyer, kind, args, manifest=manifest, session_dir=session_dir))
        return results

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_pair = {
            executor.submit(run_pair, plan, seller, buyer, kind, args, manifest, session_dir): (seller, buyer, kind)
            for seller, buyer, kind in pairs
        }
        for future in concurrent.futures.as_completed(future_to_pair):
            try:
                results.append(future.result())
            except Exception as exc:
                seller, buyer, kind = future_to_pair[future]
                failure = {
                    "pair_id": pair_id(seller, buyer, kind),
                    "seller": seller["model"],
                    "buyer": buyer["model"],
                    "kind": kind,
                    "returncode": None,
                    "exception": str(exc),
                    "finished_at": utc_now(),
                }
                results.append(failure)
                if manifest is not None:
                    manifest.event("pair_exception", **failure)
                if not args.continue_on_error:
                    raise
    return results


def main():
    parser = argparse.ArgumentParser(description="Run an A2A-NT experiment sweep.")
    parser.add_argument("--config", default="configs/sweep_example.json")
    parser.add_argument("--mode", choices=["frontier-grid", "bridge", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--product-limit", type=int, default=None)
    parser.add_argument("--products-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--judge-confirmation-model", default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--budgets", default=None, help="Comma-separated budget scenario override")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Number of buyer-seller pairs to run concurrently")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed pairs and continue the sweep")
    parser.add_argument("--include-error-files", action="store_true", help="Count data_error result JSON files as completed when resuming")
    parser.add_argument("--run-id", default=None, help="Optional run session identifier for logs/run_sessions")
    parser.add_argument("--manifest-path", default=None, help="Optional manifest JSON path")
    parser.add_argument("--skip-postprocess", action="store_true", help="Do not run result postprocessing after the sweep")
    parser.add_argument("--postprocess-move-errors", action="store_true", help="Move error files during postprocessing")
    parser.add_argument("--repair-price-scale", action="store_true", help="Apply price-scale repair suggestions during postprocessing")
    args = parser.parse_args()

    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1")
    if args.budgets:
        args.budgets = parse_csv(args.budgets)

    plan = load_plan(args.config)
    products_file = args.products_file or plan["products_file"]
    product_count = args.product_limit or count_products(products_file)
    budgets = args.budgets or plan["budgets"]
    num_experiments = args.num_experiments or plan["num_experiments"]
    pairs = build_pairs(plan, args.mode)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    for entry in disabled_models(plan.get("bridge_models", [])):
        print(f"[skip disabled bridge] {entry['label']}: {entry.get('note', 'disabled')}")

    cells = len(pairs) * product_count * len(budgets) * num_experiments
    print(f"Plan: {plan['name']}")
    print(f"Mode: {args.mode}")
    print(f"Pairs: {len(pairs)} | products: {product_count} | budgets: {len(budgets)} | experiments: {num_experiments}")
    print(f"Conversation cells: {cells}")
    print(f"Parallel workers: {args.parallel_workers}")

    if args.dry_run:
        run_pairs(plan, pairs, args)
        return

    run_id, session_dir, manifest = make_session(
        plan=plan,
        args=args,
        pairs=pairs,
        products_file=products_file,
        product_count=product_count,
        budgets=budgets,
        num_experiments=num_experiments,
    )
    print(f"Run ID: {run_id}")
    print(f"Manifest: {manifest.path}")

    try:
        run_pairs(plan, pairs, args, manifest=manifest, session_dir=session_dir)
        failed_pairs = manifest.payload.get("failed_pairs", [])
        status = "completed_with_failures" if failed_pairs else "completed"
        manifest.update(status=status)
    except Exception:
        manifest.update(status="failed")
        raise

    if not args.skip_postprocess:
        manifest.update(status="postprocessing")
        from MarkAnomaly import run_postprocess

        run_postprocess(
            base_dir=args.output_dir or plan["output_dir"],
            move_error_files=args.postprocess_move_errors,
            repair_price_scale=args.repair_price_scale,
        )
        manifest.update(
            status="completed",
            completed_result_json_count=count_valid_results(
                args.output_dir or plan["output_dir"],
                include_error_files=False,
            ),
        )


if __name__ == "__main__":
    main()
