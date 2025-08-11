import os
import json
import argparse
from Conversation import Conversation
from LanguageModel import LanguageModel
import random

def calculate_budget_scenarios(retail_price_str, wholesale_price_str):
    """Calculate the five budget scenarios."""
    # Remove $ and commas, convert to float
    retail_price = float(retail_price_str.replace("$", "").replace(",", ""))
    wholesale_price = float(wholesale_price_str.replace("$", "").replace(",", ""))
    
    # Calculate the five budget scenarios
    budgets = {
        "high": retail_price * 1.2,  # Retail Price * 1.2
        #"retail": retail_price,  # Retail Price
        #"mid": (retail_price + wholesale_price) / 2,  # (Retail Price + Wholesale Price) / 2
        #"wholesale": wholesale_price,  # Wholesale Price
        "low": wholesale_price * 0.8  # Wholesale Price * 0.8
    }
    
    return budgets

def run_experiment(product_index, products_file, buyer_model, seller_model, summary_model, max_turns, num_experiments=3, output_dir="results", append=False):
    """Run experiments for the specified product."""
    print(f"\nStarting experiments for product {product_index}...")
    
    # Load product data
    with open(products_file, "r") as f:
        products = json.load(f)
    
    if not isinstance(products, list):
        print(f"Error: {products_file} must contain a list of products")
        return None
        
    if product_index < 0 or product_index >= len(products):
        print(f"Error: Invalid product index {product_index}")
        return None
        
    product = products[product_index]
    product_id = product.get("id", product_index + 1)
    
    # Calculate the five budget scenarios
    budgets = calculate_budget_scenarios(
        product["Retail Price"], 
        product["Wholesale Price"]
    )
    
    # Create output directory structure
    # Format: seller_{seller_model}/{buyer_model}/product_{product_id}
    base_output_dir = os.path.join(output_dir, f"seller_{seller_model}/{buyer_model}/product_{product_id}")
    
    # Run experiments for each budget scenario
    for budget_name, budget_value in budgets.items():
        # Create budget-specific output directory
        full_output_dir = os.path.join(base_output_dir, f"budget_{budget_name}")
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f"\n{'-'*20}")
        print(f"Running experiments with budget scenario: {budget_name} (${budget_value:.2f})")
        print(f"{'-'*20}")
        
        # Count existing experiment files
        existing_files = [f for f in os.listdir(full_output_dir) if f.startswith(f'product_{product_id}_exp_') and f.endswith('.json')]
        existing_count = len(existing_files)
        
        # Check if we already have enough experiments
        if existing_count >= num_experiments and not append:
            print(f"\nSkipping product {product_id} with budget {budget_name} - already has {existing_count} conversations (target: {num_experiments})")
            continue
            
        print(f"\nRunning experiments for product {product_id} with budget {budget_name}")
        print(f"Found {existing_count} existing conversations, need {num_experiments - existing_count} more")
        
        # Calculate how many more experiments we need to run
        remaining_experiments = num_experiments - existing_count
        
        # Find the next available experiment number
        if append and existing_files:
            # Extract experiment numbers from filenames like "product_1_exp_0.json"
            exp_nums = [int(f.split('_exp_')[1].split('.')[0]) for f in existing_files]
            start_num = max(exp_nums) + 1
        else:
            start_num = existing_count
        
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
                budget=budget_value  # Pass the budget value
            )
            
            # Add budget name to the conversation object
            conversation.budget_scenario = budget_name
            
            # Run the negotiation
            conversation.run_negotiation()
            
            # Save conversation
            conversation.save_conversation(full_output_dir)
            
            # Print summary of the completed negotiation
            print(f"\nExperiment {existing_count + exp_num + 1} completed and saved to {full_output_dir}")
            print(f"Completed turns: {conversation.completed_turns}")
            if conversation.negotiation_completed:
                print(f"Negotiation result: {conversation.negotiation_result}")
                if conversation.negotiation_result == "accepted":
                    print(f"Deal accepted at price: ${conversation.current_price_offer:.2f}")
                elif conversation.negotiation_result == "rejected":
                    print(f"Deal rejected. Final price offered: ${conversation.current_price_offer:.2f}")
            else:
                print("Negotiation reached maximum turns without natural conclusion")

def run_all_products(products_file, buyer_model, seller_model, summary_model, max_turns, num_experiments=5, output_dir="results", append=False):
    """Run experiments for all products in the dataset."""
    # Load all products
    with open(products_file, "r") as f:
        products = json.load(f)
    
    if not isinstance(products, list):
        print(f"Error: {products_file} must contain a list of products")
        return
    
    print(f"Found {len(products)} products in the dataset.")
    print(f"Using products from: {products_file}")
    
    # Run experiments for each product
    for i, product in enumerate(products):
        print(f"\n{'='*50}")
        print(f"Processing product {i+1}/{len(products)}: {product['Product Name']}")
        print(f"{'='*50}")
        
        run_experiment(
            product_index=i,
            products_file=products_file,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            num_experiments=num_experiments,
            output_dir=output_dir,
            append=append
        )

def main():
    """Main entry point for the negotiation simulator."""
    parser = argparse.ArgumentParser(description="LLM-based Negotiation Simulator")
    parser.add_argument("--products-file", default="dataset/products.json", help="Path to the products JSON file")
    parser.add_argument("--buyer-model", default="gpt-3.5-turbo", help="Model to use for the buyer agent")
    parser.add_argument("--seller-model", default="gpt-3.5-turbo", help="Model to use for the seller agent")
    parser.add_argument("--summary-model", default="gpt-3.5-turbo", help="Model to use for extracting price offers from seller messages")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum number of conversation turns (safety limit)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--num-experiments", type=int, default=3, help="Number of experiments to run")
    parser.add_argument("--append", action="store_true", help="Append new runs to existing results instead of overwriting")
    
    args = parser.parse_args()
    
    # Run experiments for all products
    run_all_products(
        products_file=args.products_file,
        buyer_model=args.buyer_model,
        seller_model=args.seller_model, 
        summary_model=args.summary_model,
        max_turns=args.max_turns,
        num_experiments=args.num_experiments,
        output_dir=args.output_dir,
        append=args.append
    )
    
    print("\nExperiments for all products completed successfully!")

if __name__ == "__main__":
    main()
