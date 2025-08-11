#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Recommend starting with one strong seller for convergence; add others as needed
SELLER_MODELS_TRAIN=("gpt-3.5-turbo" "qwen2.5-7b-instruct")

BUYER_MODEL="qwen2.5-7b-instruct"
SUMMARY_MODEL="gpt-4o-mini"
PRODUCTS_FILE="dataset/products.json"
TEST_FILE="dataset/products_mini.json"

# Each seller: at least >= 192 steps (covers 96 actions * 2 budgets)
STEPS_PER_SELLER=400
MAX_TURNS=8
SEED=42
WARMUP_ENABLED=1
EPSILON=0.10
ENFORCE_MIN_COUNT_EVERY=20

LOG_DIR="logs"
ART_DIR="artifacts"
mkdir -p "$LOG_DIR" "$ART_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/rl_train_${TS}.log"
PROGRESS_LOG="$LOG_DIR/rl_progress_${TS}.log"
BEST_JSON="$ART_DIR/rl_best_action.json"

to_py_list() {
  local -n arr=$1; local s="["
  for x in "${arr[@]}"; do s+="\"$x\","; done
  echo "${s%,}]"
}
SELLERS_TRAIN_PY=$(to_py_list SELLER_MODELS_TRAIN)
WARMUP_PY=$([[ "$WARMUP_ENABLED" -eq 1 ]] && echo True || echo False)

echo "=== RL Train (multi-sellers) ==="
echo "Train sellers: ${SELLER_MODELS_TRAIN[*]}"
echo "Buyer: ${BUYER_MODEL} | Summary: ${SUMMARY_MODEL}"
echo "Steps per seller: ${STEPS_PER_SELLER} | max_turns: ${MAX_TURNS} | seed: ${SEED}"
echo "Warmup: ${WARMUP_PY} | epsilon(init): ${EPSILON} | enforce_min_count_every: ${ENFORCE_MIN_COUNT_EVERY}"
echo "Logs: ${LOG_FILE}"
echo "Progress: ${PROGRESS_LOG}"
echo "Best action JSON: ${BEST_JSON}"
echo "================================"

# Metrics → stdout → LOG_FILE; Progress bar → stderr → PROGRESS_LOG
python -u - 1> >(tee -a "$LOG_FILE") 2> >(tee -a "$PROGRESS_LOG" >&2) <<PY
from rl.train_bandit import train_multi_sellers
from rl.prompt_space import ACTIONS
import json, sys

print(f"[ACTIONS] total={len(ACTIONS)} warmup_episodes={2*len(ACTIONS)}")

sellers = ${SELLERS_TRAIN_PY}
idx = train_multi_sellers(
    seller_models=sellers,
    products_file="${PRODUCTS_FILE}",
    test_file="${TEST_FILE}",
    buyer_model="${BUYER_MODEL}",
    summary_model="${SUMMARY_MODEL}",
    steps_per_seller=${STEPS_PER_SELLER},
    max_turns=${MAX_TURNS},
    seed=${SEED},
    warmup_enabled=${WARMUP_PY},
    epsilon=${EPSILON},
    enforce_min_count_every=${ENFORCE_MIN_COUNT_EVERY},
    mute_dialogue=True,   # Mute dialogue, keep real-time metrics and progress bar
)
best = {"best_action_idx": int(idx), "best_action_name": ACTIONS[idx]["name"]}
print("\\n[Train finished] best:", best)
with open("${BEST_JSON}", "w", encoding="utf-8") as f:
    json.dump(best, f, ensure_ascii=False, indent=2)
PY