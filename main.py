import argparse
import os
import sys
from Conversation import Conversation
from LanguageModel import RUN_FATAL_EXIT_CODE
from experiment_utils import (
    SUPPORTED_BUDGET_SCENARIOS,
    calculate_budget_scenarios,
    count_valid_results,
    iter_result_files,
    load_products,
    next_experiment_number,
    parse_csv,
    parse_int_csv,
    result_base_dir,
    select_budget_scenarios,
    select_products,
)


def _run_product_experiment(
    product_index,
    product,
    buyer_model,
    seller_model,
    summary_model,
    max_turns,
    num_experiments=3,
    output_dir="results",
    append=False,
    budget_scenarios=None,
    dry_run=False,
    include_error_files=False,
    judge_confirmation_model=None,
):
    """Run experiments for the specified product."""
    print(f"\nStarting experiments for product {product_index}...")

    product_id = product.get("id", product_index + 1)
    
    all_budgets = calculate_budget_scenarios(
        product["Retail Price"],
        product["Wholesale Price"]
    )
    budgets = select_budget_scenarios(all_budgets, budget_scenarios)

    # Create output directory structure
    # Format: seller_{seller_model}/{buyer_model}/product_{product_id}
    base_output_dir = result_base_dir(output_dir, seller_model, buyer_model, product_id)
    
    # Run experiments for each budget scenario
    for budget_name, budget_value in budgets.items():
        # Create budget-specific output directory
        full_output_dir = os.path.join(base_output_dir, f"budget_{budget_name}")
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f"\n{'-'*20}")
        print(f"Running experiments with budget scenario: {budget_name} (${budget_value:.2f})")
        print(f"{'-'*20}")

        if dry_run:
            print(
                "[dry-run] would run "
                f"seller={seller_model} buyer={buyer_model} product={product_id} "
                f"budget={budget_name} experiments={num_experiments}"
            )
            continue
        
        existing_files = iter_result_files(full_output_dir, product_id=product_id)
        existing_total = len(existing_files)
        existing_count = count_valid_results(
            full_output_dir,
            product_id=product_id,
            include_error_files=include_error_files,
        )
        
        # Check if we already have enough experiments
        if existing_count >= num_experiments:
            print(f"\nSkipping product {product_id} with budget {budget_name} - already has {existing_count} valid conversations (target: {num_experiments})")
            continue
            
        print(f"\nRunning experiments for product {product_id} with budget {budget_name}")
        print(
            f"Found {existing_count} valid conversations ({existing_total} JSON files), "
            f"need {num_experiments - existing_count} more"
        )
        
        # Calculate how many more experiments we need to run
        remaining_experiments = num_experiments - existing_count
        
        # Always avoid overwriting parseable, invalid, or partially failed evidence files.
        start_num = next_experiment_number(full_output_dir, product_id=product_id)
        
        # Run experiments
        for exp_num in range(remaining_experiments):
            experiment_num = start_num + exp_num
            print(f"\nRunning experiment {existing_count + exp_num + 1}/{num_experiments}")
            print(f"Experiment Number: {experiment_num}")
            print(f"Budget: ${budget_value:.2f}")
            
            # Create conversation instance with budget
            conversation = Conversation(
                product_data=product,
                buyer_model=buyer_model,
                seller_model=seller_model,
                summary_model=summary_model,
                max_turns=max_turns,  # Used as a safety mechanism
                experiment_num=experiment_num,
                budget=budget_value,  # Pass the budget value
                judge_confirmation_model=judge_confirmation_model,
            )
            
            # Add budget name to the conversation object
            conversation.budget_scenario = budget_name
            
            # Run the negotiation
            conversation.run_negotiation()
            
            # Save conversation
            conversation.save_conversation(full_output_dir)

            if conversation.run_fatal_error:
                print(
                    "Run-fatal model error encountered. "
                    f"Stopping subprocess with exit code {RUN_FATAL_EXIT_CODE}."
                )
                sys.exit(RUN_FATAL_EXIT_CODE)
            
            # Print summary of the completed negotiation
            print(f"\nExperiment {existing_count + exp_num + 1} completed and saved to {full_output_dir}")
            print(f"Completed turns: {conversation.completed_turns}")
            if conversation.negotiation_completed:
                print(f"Negotiation result: {conversation.negotiation_result}")
                if conversation.negotiation_result == "accepted":
                    if conversation.current_price_offer is None:
                        print("Deal accepted at price: None")
                    else:
                        print(f"Deal accepted at price: ${conversation.current_price_offer:.2f}")
                elif conversation.negotiation_result == "rejected":
                    if conversation.current_price_offer is None:
                        print("Deal rejected. Final price offered: None")
                    else:
                        print(f"Deal rejected. Final price offered: ${conversation.current_price_offer:.2f}")
            else:
                print("Negotiation reached maximum turns without natural conclusion")


def run_experiment(
    product_index,
    products_file,
    buyer_model,
    seller_model,
    summary_model,
    max_turns,
    num_experiments=3,
    output_dir="results",
    append=False,
    budget_scenarios=None,
    dry_run=False,
    include_error_files=False,
    judge_confirmation_model=None,
):
    """Run experiments for the specified product index."""
    try:
        products = load_products(products_file)
    except ValueError as exc:
        print(f"Error: {exc}")
        return None

    if product_index < 0 or product_index >= len(products):
        print(f"Error: Invalid product index {product_index}")
        return None

    return _run_product_experiment(
        product_index=product_index,
        product=products[product_index],
        buyer_model=buyer_model,
        seller_model=seller_model,
        summary_model=summary_model,
        max_turns=max_turns,
        num_experiments=num_experiments,
        output_dir=output_dir,
        append=append,
        budget_scenarios=budget_scenarios,
        dry_run=dry_run,
        include_error_files=include_error_files,
        judge_confirmation_model=judge_confirmation_model,
    )

def run_all_products(
    products_file,
    buyer_model,
    seller_model,
    summary_model,
    max_turns,
    num_experiments=5,
    output_dir="results",
    append=False,
    start_index=0,
    product_limit=None,
    product_ids=None,
    budget_scenarios=None,
    dry_run=False,
    include_error_files=False,
    judge_confirmation_model=None,
):
    """Run experiments for all products in the dataset."""
    try:
        products = load_products(products_file)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    
    print(f"Found {len(products)} products in the dataset.")
    print(f"Using products from: {products_file}")
    selected_products = select_products(
        products,
        start_index=start_index,
        product_limit=product_limit,
        product_ids=product_ids,
    )

    # Run experiments for each selected product
    for i, product in selected_products:
        print(f"\n{'='*50}")
        print(f"Processing product {i+1}/{len(products)}: {product['Product Name']}")
        print(f"{'='*50}")
        
        _run_product_experiment(
            product_index=i,
            product=product,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            num_experiments=num_experiments,
            output_dir=output_dir,
            append=append,
            budget_scenarios=budget_scenarios,
            dry_run=dry_run,
            include_error_files=include_error_files,
            judge_confirmation_model=judge_confirmation_model,
        )

def main():
    """Main entry point for the negotiation simulator."""
    parser = argparse.ArgumentParser(description="LLM-based Negotiation Simulator")
    parser.add_argument("--products-file", default="dataset/products.json", help="Path to the products JSON file")
    parser.add_argument("--buyer-model", default="gpt-3.5-turbo", help="Model to use for the buyer agent")
    parser.add_argument("--seller-model", default="gpt-3.5-turbo", help="Model to use for the seller agent")
    parser.add_argument("--summary-model", default="gpt-3.5-turbo", help="Model to use for extracting price offers from seller messages")
    parser.add_argument("--judge-confirmation-model", default=None, help="Optional second-pass judge model for boundary cases")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum number of conversation turns (safety limit)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--num-experiments", type=int, default=3, help="Number of experiments to run")
    parser.add_argument("--append", action="store_true", help="Append new runs to existing results instead of overwriting")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based product index to start from")
    parser.add_argument("--product-limit", type=int, default=None, help="Maximum number of products to run")
    parser.add_argument("--product-ids", default=None, help="Comma-separated product ids to run")
    parser.add_argument(
        "--budget-scenarios",
        default="wholesale,mid,retail",
        help=f"Comma-separated budget scenarios: {', '.join(SUPPORTED_BUDGET_SCENARIOS)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without calling model APIs")
    parser.add_argument("--include-error-files", action="store_true", help="Count data_error result JSON files as completed when resuming")
    parser.add_argument("--postprocess", action="store_true", help="Run non-destructive result postprocessing after this run")
    parser.add_argument("--postprocess-move-errors", action="store_true", help="When postprocessing, move error files into error_data/")
    parser.add_argument("--repair-price-scale", action="store_true", help="When postprocessing, apply price-scale repair suggestions")
    
    args = parser.parse_args()
    product_ids = parse_int_csv(args.product_ids)
    budget_scenarios = parse_csv(args.budget_scenarios)
    
    # Run experiments for all products
    run_all_products(
        products_file=args.products_file,
        buyer_model=args.buyer_model,
        seller_model=args.seller_model, 
        summary_model=args.summary_model,
        max_turns=args.max_turns,
        num_experiments=args.num_experiments,
        output_dir=args.output_dir,
        append=args.append,
        start_index=args.start_index,
        product_limit=args.product_limit,
        product_ids=product_ids,
        budget_scenarios=budget_scenarios,
        dry_run=args.dry_run,
        include_error_files=args.include_error_files,
        judge_confirmation_model=args.judge_confirmation_model,
    )

    if args.postprocess and not args.dry_run:
        from MarkAnomaly import run_postprocess

        run_postprocess(
            base_dir=args.output_dir,
            move_error_files=args.postprocess_move_errors,
            repair_price_scale=args.repair_price_scale,
        )
    
    print("\nExperiments for all products completed successfully!")

if __name__ == "__main__":
    main()
