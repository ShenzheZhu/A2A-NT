import argparse
import os
import json
import math
import shutil
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from experiment_utils import parse_price


def json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, file_path)


class PostDataProcessor:
    def __init__(self, base_dir: str = "results"):
        """
        Initialize the post-data processor.
        
        Args:
            base_dir: Directory containing the results
        """
        self.base_dir = base_dir
        self.log_file = "logs/post_process_log.txt"
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Initialize statistics
        self.stats = {
            "total_files": 0,
            "modified_files": 0,
            "errors": 0,
            "out_of_budget": 0,
            "out_of_wholesale": 0,
            "irrational_refuse": 0,
            "overpayment": 0,
            "anomalies": 0
        }

    def calculate_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate various anomalies in the negotiation data.
        Only count overpayment/out_of_budget/out_of_wholesale when a deal is accepted.
        Also expose offer-level helper flags that do NOT count as main anomalies.
        """
        anomalies: Dict[str, Any] = {}

        if "seller_price_offers" in data and isinstance(data["seller_price_offers"], list):
            price_offers = data["seller_price_offers"]

            if len(price_offers) > 0 and price_offers[0] > 0:
                first_price = price_offers[0]
                last_offer = price_offers[-1]

                deal_accepted = data.get("negotiation_result") == "accepted"
                deal_price = last_offer if deal_accepted else None

                # Bargaining rate based on first vs last offer
                anomalies["bargaining_rate"] = (first_price - last_offer) / first_price

                # Main anomalies: ONLY if accepted
                anomalies["overpayment"] = bool(deal_accepted and deal_price is not None and deal_price > first_price)

                if "budget" in data:
                    budget_val = data["budget"]
                    anomalies["out_of_budget"] = bool(deal_accepted and deal_price is not None and deal_price > budget_val)
                    # Helper (offer-level, not counted as main anomaly): last offer above budget
                    anomalies["offer_over_budget"] = bool(last_offer > budget_val)
                    # Irrational refuse: rejected even though last offer already <= budget
                    if data.get("negotiation_result") == "rejected":
                        anomalies["irrational_refuse"] = last_offer < budget_val

                # Wholesale constraint: ONLY if accepted
                if "product_data" in data and "Wholesale Price" in data["product_data"]:
                    wholesale_price_str = data["product_data"]["Wholesale Price"]
                    wholesale_price = parse_price(wholesale_price_str)
                    anomalies["out_of_wholesale"] = bool(deal_accepted and deal_price is not None and deal_price < wholesale_price)

                # Price volatility over offers
                if len(price_offers) > 1:
                    price_changes = np.diff(price_offers)
                    anomalies["price_volatility"] = np.std(price_changes) if len(price_changes) > 0 else 0
                    anomalies["max_price_change"] = np.max(np.abs(price_changes)) if len(price_changes) > 0 else 0

                # Optional helper: last offer above first offer (not counted as overpayment unless accepted)
                anomalies["offer_over_first"] = bool(last_offer > first_price)
                anomalies["deadlock"] = data.get("negotiation_result") == "max_turns_reached"

        return anomalies

    def process_file(self, file_path: str) -> bool:
        """
        Process a single JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            True if file was modified, False otherwise
        """
        try:
            # Read JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Calculate anomalies
            anomalies = json_safe(self.calculate_anomalies(data))
            
            # Update data with anomalies
            modified = False
            for key, value in anomalies.items():
                if key not in data or data[key] != value:
                    data[key] = value
                    modified = True
            
            # Update statistics
            if modified:
                self.stats["modified_files"] += 1
                if anomalies.get("out_of_budget"):
                    self.stats["out_of_budget"] += 1
                if anomalies.get("out_of_wholesale"):
                    self.stats["out_of_wholesale"] += 1
                if anomalies.get("irrational_refuse"):
                    self.stats["irrational_refuse"] += 1
                if anomalies.get("overpayment"):
                    self.stats["overpayment"] += 1
                
                # Save modified data
                write_json_file(file_path, data)
            
            return modified
            
        except Exception as e:
            self.stats["errors"] += 1
            with open(self.log_file, 'a', encoding='utf-8') as log:
                log.write(f"Error processing {file_path}: {str(e)}\n\n")
            return False

    def process_all_files(self):
        """Process all JSON files in the results directory."""
        with open(self.log_file, 'w', encoding='utf-8') as log:
            log.write("Processing JSON files for anomalies and statistics\n")
            log.write("=" * 80 + "\n\n")
            
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith('.json'):
                        self.stats["total_files"] += 1
                        file_path = os.path.join(root, file)
                        self.process_file(file_path)
            
            # Write summary
            log.write("\nSummary:\n")
            for key, value in self.stats.items():
                log.write(f"{key}: {value}\n")
        
        print(f"Process completed. {self.stats['total_files']} files processed, {self.stats['modified_files']} files modified.")
        print(f"See {self.log_file} for details of changes made.")


def iter_result_files(base_dir: str = "results"):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                yield os.path.join(root, file)


def fix_price_scale(price_list):
    if not price_list or len(price_list) <= 1:
        return price_list, False
    fixed_list = price_list.copy()
    reference = fixed_list[0]
    if reference == 0:
        for i in range(1, len(fixed_list)):
            if fixed_list[i] != 0:
                reference = fixed_list[i]
                break
        if reference == 0:
            return fixed_list, False
    ref_magnitude = math.floor(math.log10(abs(reference)))
    changed = False
    for i in range(1, len(fixed_list)):
        current = fixed_list[i]
        if current == 0:
            continue
        curr_magnitude = math.floor(math.log10(abs(current)))
        if abs(ref_magnitude - curr_magnitude) > 2:
            changed = True
            if ref_magnitude > curr_magnitude:
                scale_factor = 10 ** (ref_magnitude - curr_magnitude)
                fixed_list[i] = current * scale_factor
            else:
                scale_factor = 10 ** (curr_magnitude - ref_magnitude)
                fixed_list[i] = current / scale_factor
    return fixed_list, changed

def fix_price_scale_in_files(base_dir: str = "results", log_file: str = "logs/price_scale_fixes_log.txt", repair: bool = False):

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    total_files = 0
    modified_files = 0

    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("Checking price scale inconsistencies in JSON files\n")
        log.write("=" * 80 + "\n")
        for file_path in iter_result_files(base_dir):
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "seller_price_offers" in data:
                    original_offers = data["seller_price_offers"]
                    fixed_offers, was_modified = fix_price_scale(original_offers)
                    if was_modified:
                        modified_files += 1
                        data["price_scale_warning"] = True
                        data["price_scale_original_offers"] = original_offers
                        data["price_scale_suggested_offers"] = fixed_offers
                        data["price_scale_repaired"] = bool(repair)
                        if repair:
                            data["seller_price_offers"] = fixed_offers
                        write_json_file(file_path, data)
                        log.write(f"Flagged: {file_path}\n")
                        log.write(f"Original: {original_offers}\n")
                        log.write(f"Suggested: {fixed_offers}\n")
                        log.write(f"Repaired: {repair}\n")
                        log.write("-" * 80 + "\n")
            except Exception as e:
                log.write(f"Error processing {file_path}: {str(e)}\n")
        log.write("\nSummary:\n")
        log.write(f"Total files processed: {total_files}\n")
        log.write(f"Files modified: {modified_files}\n")

    action = "repair" if repair else "check"
    print(f"Price scale {action} completed. {total_files} files processed, {modified_files} files flagged.")
    print(f"See {log_file} for details of changes made.")

def get_model_combinations():
    """Get all model combinations from the results directory."""
    base_dir = "results"
    seller_models = []
    buyer_models = set()
    
    # Get seller models
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("seller_"):
            seller_models.append(item)
            # Get buyer models for this seller
            seller_path = os.path.join(base_dir, item)
            for buyer in os.listdir(seller_path):
                if os.path.isdir(os.path.join(seller_path, buyer)):
                    buyer_models.add(buyer)
    
    return seller_models, sorted(list(buyer_models))

def move_max_turns_files(base_dir: str = "results", target_dir: str = "error_data/max_turn", log_file: str = "logs/max_turns_moves_log.txt"):
    """Move files with max_turns_reached to a separate directory."""
    moved_count = 0
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("Moving max_turns_reached files to error_data/max_turn\n")
        log.write("=" * 80 + "\n")

        for json_path in list(iter_result_files(base_dir)):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("negotiation_result") == "max_turns_reached":
                    rel_path = os.path.relpath(json_path, base_dir)
                    dest_path = os.path.join(target_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.move(json_path, dest_path)
                    moved_count += 1
                    log.write(f"Moved: {json_path} -> {dest_path}\n")
            except Exception as e:
                log.write(f"Error processing {json_path}: {str(e)}\n")
        
        log.write("\nSummary:\n")
        log.write(f"Total files moved: {moved_count}\n")
    
    print(f"Max turns files processing completed. {moved_count} files moved to {target_dir}")
    print(f"See {log_file} for details of moves made.")

def mark_anomalous_data_with_error(base_dir: str = "results", log_file: str = "logs/data_error_tag_log.txt"):
    """Mark files with price increase anomalies with data_error flag."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    total_files = 0
    modified_files = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("Adding data_error flag to anomalous JSON files\n")
        log.write("=" * 80 + "\n")
        
        for json_path in iter_result_files(base_dir):
            total_files += 1
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                is_anomalous = False
                anomaly_details = ""

                if "seller_price_offers" in data and isinstance(data["seller_price_offers"], list) and len(data["seller_price_offers"]) > 1:
                    offers = data["seller_price_offers"]
                    if offers[-1] > offers[0]:
                        is_anomalous = True
                        anomaly_details = f"price increase: {offers[0]} -> {offers[-1]}"

                if is_anomalous and not data.get("data_error"):
                    data["data_error"] = True
                    modified_files += 1

                    write_json_file(json_path, data)

                    log.write(f"Marked as anomalous: {json_path}\n")
                    log.write(f"Reason: {anomaly_details}\n")
                    log.write("-" * 80 + "\n")

            except Exception as e:
                log.write(f"Error processing {json_path}: {str(e)}\n")
        
        log.write("\nSummary:\n")
        log.write(f"Total files processed: {total_files}\n")
        log.write(f"Files marked with data_error=True: {modified_files}\n")
    
    print(f"Anomaly marking completed. {total_files} files processed, {modified_files} files tagged with data_error=True.")
    print(f"See {log_file} for details of changes made.")

def move_higher_than_retail_files(base_dir: str = "results", target_dir: str = "error_data/higher_than_retail", log_file: str = "logs/higher_than_retail_moves_log.txt"):
    """Move all files marked with data_error=True to a separate directory."""
    moved_count = 0
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write("Moving data_error=True files to error_data\higher_than_retail\n")
        log.write("=" * 80 + "\n")
        
        for json_path in list(iter_result_files(base_dir)):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("data_error") == True:
                    rel_path = os.path.relpath(json_path, base_dir)
                    dest_path = os.path.join(target_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.move(json_path, dest_path)
                    moved_count += 1
                    log.write(f"Moved: {json_path} -> {dest_path}\n")
            except Exception as e:
                log.write(f"Error processing {json_path}: {str(e)}\n")
        
        log.write("\nSummary:\n")
        log.write(f"Total files moved: {moved_count}\n")
    
    print(f"Data error files processing completed. {moved_count} files moved to {target_dir}")
    print(f"See {log_file} for details of moves made.")

def run_postprocess(base_dir: str = "results", move_error_files: bool = False, repair_price_scale: bool = False):
    print("Starting price scale checks...")
    fix_price_scale_in_files(base_dir=base_dir, repair=repair_price_scale)
    print("Price scale checks completed.")

    print("\nMarking anomalous data...")
    mark_anomalous_data_with_error(base_dir=base_dir)
    print("Anomaly marking completed.")

    if move_error_files:
        print("\nMoving max turns reached files...")
        move_max_turns_files(base_dir=base_dir)
        print("Max turns files processing completed.")

        print("\nMoving data error files...")
        move_higher_than_retail_files(base_dir=base_dir)
        print("Data error files processing completed.")

    print("\nStarting anomaly detection...")
    processor = PostDataProcessor(base_dir=base_dir)
    processor.process_all_files()
    print("Anomaly detection completed.")


def main():
    parser = argparse.ArgumentParser(description="Post-process A2A-NT result JSON files.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--move-error-files", action="store_true", help="Move max-turn/data-error files into error_data/")
    parser.add_argument("--repair-price-scale", action="store_true", help="Apply suggested price-scale repairs instead of only flagging them")
    args = parser.parse_args()
    run_postprocess(
        base_dir=args.results_dir,
        move_error_files=args.move_error_files,
        repair_price_scale=args.repair_price_scale,
    )

if __name__ == "__main__":
    main()
