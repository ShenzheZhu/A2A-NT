#!/bin/bash

# Define model lists
BUYER_MODELS=("deepseek-chat" "deepseek-reasoner" "gpt-4o-mini" "gpt-3.5-turbo" "qwen2.5-7b-instruct" "qwen2.5-14b-instruct" "gpt-4.1" "o4-mini" "o3")
SELLER_MODELS=("deepseek-chat" "deepseek-reasoner" "gpt-4o-mini" "gpt-3.5-turbo" "qwen2.5-7b-instruct" "qwen2.5-14b-instruct" "gpt-4.1" "o4-mini" "o3")
SUMMARY_MODEL="gpt-4o-mini"  # Using deepseek-chat for all summary tasks

# Set common parameters
PRODUCTS_FILE="dataset/products_mini.json"
OUTPUT_DIR="results"
LOGS_DIR="logs"
MAX_TURNS=30 
NUM_EXPERIMENTS=1
MAX_PARALLEL_JOBS=64  # Control how many jobs can run in parallel

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Function to wait until we have fewer than MAX_PARALLEL_JOBS running
wait_for_job_slot() {
  while [[ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]]; do
    sleep 5
  done
  return 0
}

# Track running processes
declare -a PIDS=()

# Log start time
echo "Starting all experiments at $(date)"
echo "Running 4x4 model combination experiments (16 combinations total) with 5 budget scenarios per product"
echo "Running max $MAX_PARALLEL_JOBS experiments in parallel at a time"
echo "Summary model: $SUMMARY_MODEL"
echo "Safety max turns: $MAX_TURNS"
echo "==============================================================="

# Loop through all model combinations and run them in controlled parallel
for SELLER in "${SELLER_MODELS[@]}"; do
  for BUYER in "${BUYER_MODELS[@]}"; do
    # Wait for a job slot to become available
    wait_for_job_slot
    
    LOG_FILE="$LOGS_DIR/${SELLER}_vs_${BUYER}.log"
    
    echo "Starting experiments with Seller: $SELLER, Buyer: $BUYER"
    
    # Run the experiment in the background
    (
      echo "===============================================================" > "$LOG_FILE"
      echo "Running experiments with:" >> "$LOG_FILE"
      echo "Seller: $SELLER" >> "$LOG_FILE"
      echo "Buyer: $BUYER" >> "$LOG_FILE"
      echo "Start time: $(date)" >> "$LOG_FILE"
      echo "Budget scenarios: Retail Price*1.2, Retail Price, (Retail+Wholesale)/2, Wholesale Price, Wholesale Price*0.8" >> "$LOG_FILE"
      echo "===============================================================" >> "$LOG_FILE"
      
      # Run the experiment for this model combination
      python main.py \
        --products-file "$PRODUCTS_FILE" \
        --buyer-model "$BUYER" \
        --seller-model "$SELLER" \
        --summary-model "$SUMMARY_MODEL" \
        --max-turns "$MAX_TURNS" \
        --num-experiments "$NUM_EXPERIMENTS" \
        --output-dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1
      
      echo "===============================================================" >> "$LOG_FILE"
      echo "Completed $SELLER vs $BUYER at $(date)" >> "$LOG_FILE"
      echo "Completed experiments with Seller: $SELLER, Buyer: $BUYER"
    ) &
    
    # Record the process ID
    PIDS+=($!)
    
    echo "Started job for $SELLER vs $BUYER with PID $!"
  done
done

echo "All jobs have been queued. Waiting for completion..."
echo "Running with max $MAX_PARALLEL_JOBS parallel jobs"

# Wait for all background processes to complete
while [[ $(jobs -r | wc -l) -gt 0 ]]; do
  sleep 5
done

echo ""
echo "==============================================================="
echo "Experiments completed at $(date)"
echo "Results saved to $OUTPUT_DIR"
echo "Logs saved to $LOGS_DIR"
echo "Budget scenarios run for each product:"
echo "- High: Retail Price * 1.2"
echo "- Retail: Retail Price"
echo "- Mid: (Retail Price + Wholesale Price) / 2"
echo "- Wholesale: Wholesale Price"
echo "- Low: Wholesale Price * 0.8"

# Run MarkAnomaly.py after all experiments are completed
echo ""
echo "==============================================================="
echo "Starting anomaly detection at $(date)"
python MarkAnomaly.py
echo "Anomaly detection completed at $(date)"