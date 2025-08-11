#!/usr/bin/env bash
# run_rl_eval.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Sellers for evaluation (can add/remove)
SELLER_MODELS_EVAL=("deepseek-chat" "deepseek-reasoner" "gpt-4o-mini" "gpt-3.5-turbo" "qwen2.5-7b-instruct" "qwen2.5-14b-instruct" "gpt-4.1" "o4-mini" "o3")

BUYER_MODEL="qwen2.5-7b-instruct"
SUMMARY_MODEL="gpt-4o-mini"
TEST_FILE="dataset/products_mini.json"
BEST_JSON="artifacts/rl_best_action.json"
MAX_TURNS=10
SEED=123
CUSTOM_PROMPT=True
# Concurrent sellers = number of sellers
MAX_WORKERS=${#SELLER_MODELS_EVAL[@]}

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
ART_DIR="artifacts"
mkdir -p "$LOG_DIR" "$ART_DIR"

LOG_FILE="$LOG_DIR/rl_eval_${TS}.log"
PROGRESS_LOG="$LOG_DIR/rl_eval_progress_${TS}.log"
RES_JSON="$ART_DIR/rl_eval_${TS}.json"

EP_DIR="artifacts/eval_episodes_${TS}"
SUM_JSONL="artifacts/eval_summaries_${TS}.jsonl"
echo "Episodes dir: ${EP_DIR}"
echo "Summaries jsonl: ${SUM_JSONL}"


to_py_list() {
  local -n arr=$1; local s="["
  for x in "${arr[@]}"; do s+="\"$x\","; done
  echo "${s%,}]"
}
SELLERS_EVAL_PY=$(to_py_list SELLER_MODELS_EVAL)

echo "=== RL Eval (multi-sellers) ==="
echo "Eval sellers: ${SELLER_MODELS_EVAL[*]}"
echo "Buyer: ${BUYER_MODEL} | Summary: ${SUMMARY_MODEL}"
echo "Test file: ${TEST_FILE}"
echo "Best JSON: ${BEST_JSON}"
echo "Logs: ${LOG_FILE}"
echo "Progress log: ${PROGRESS_LOG}"
echo "================================"

# Metrics → stdout → LOG_FILE; Progress bar → stderr → PROGRESS_LOG

python -u - >> "$LOG_FILE" 2>> "$PROGRESS_LOG" <<PY
from rl.eval_bandit import evaluate_all_sellers_parallel_models_only
from rl.prompt_space import ACTIONS
import json, pathlib

with open("${BEST_JSON}","r",encoding="utf-8") as f:
    best=json.load(f)
idx=int(best.get("best_action_idx", 0))

evaluate_all_sellers_parallel_models_only(
    best_action_idx=idx,
    seller_models=${SELLERS_EVAL_PY},
    products_file="${TEST_FILE}",
    buyer_model="${BUYER_MODEL}",
    summary_model="${SUMMARY_MODEL}",
    max_turns=${MAX_TURNS},
    seed=${SEED},
    seller_workers=${MAX_WORKERS},
    show_progress=True,
    save_path="${RES_JSON}",
    custom_prompt=${CUSTOM_PROMPT},   # Key: pass custom prompt
    summary_jsonl_path="${SUM_JSONL}",
    episodes_dir="${EP_DIR}",
)
PY