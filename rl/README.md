### RL Prompt Optimization (Softmax Bandit)

- Goal: Select the best buyer system prompt from an auditable action space via online softmax bandit, minimizing negotiation anomalies and unnecessary turns.
- Method: REINFORCE-style softmax bandit with a moving baseline (not UCB/Thompson).

### Core math
- Policy over K actions: $\pi_i(\theta)=\frac{e^{\theta_i}}{\sum_{j=1}^K e^{\theta_j}}$
- Sample: $a_t\sim \pi(\theta)$, receive reward $r_t$
- Baseline and advantage: $b_t=0.9\,b_{t-1}+0.1\,r_t,\; A_t=r_t-b_t$
- Update (only chosen action): $\theta_{a_t}\leftarrow\theta_{a_t}+\eta\,A_t(1-\pi_{a_t})$

### Reward shaping (rl/env.py)
- High budget + overpayment: −2.0
- High budget + accepted and final > first quote: −1.0
- Low budget + out_of_budget: −1.0
- Deadlock: −1.0
- Turn penalty: −0.02 × turns

### Action space (rl/prompt_space.py)
- ~96 combinations (`SPACE` → `ACTIONS`): budget emphasis, price-increase handling, no-progress exit turns, progress threshold, concession style, non‑price ask, tone, brevity, self‑check, etc.
- Prompts are rendered by `build_buyer_system_prompt(...)`.

### Training workflow (rl/train_bandit.py, see train.sh)
- Warmup coverage: each action once at high and low; online updates.
- Main training:
  - Budget sampling: first half low; second half picks high with prob 0.7.
  - Annealed exploration: $\varepsilon: 0.10 \to 0.02$ by global progress.
  - Top‑K active set: start K=24; shrink to K=12 at 2/3 progress; sample only within active set (normalized).
  - Every N steps, force the least‑sampled action in active set; otherwise ε‑random vs softmax sampling.
- Output: best single action is argmax $\theta$. Saves distribution CSV and best index JSON.

### Evaluation (rl/eval_bandit.py, see eval.sh)
- Two modes:
  - Bandit best action: `CUSTOM_PROMPT=False`
  - Side experiment (custom single prompt): `CUSTOM_PROMPT=True` (bypasses `ACTIONS` via `buyer_system_prompt=True`)
- Metrics: overpayment@high, out_of_budget@low, deadlock, avg_turns.
- Outputs:
  - Per‑seller episodes JSONL: `artifacts/eval_episodes_*/episodes__*.jsonl`
  - Optional overall JSON and summaries JSONL.

### Key files
- `rl/prompt_space.py`: action space and prompt rendering
- `rl/policy.py`: softmax bandit with baseline update
- `rl/env.py`: episode run, reward calculation, anomaly flags
- `rl/rl_conversation.py`: maps action → buyer system prompt
- `rl/train_bandit.py`: warmup, annealed ε, Top‑K, multi‑seller loop
- `rl/eval_bandit.py`: per‑seller and multi‑seller evaluation
- `train.sh` / `eval.sh`: entry scripts
- Artifacts: `artifacts/rl_best_action.json`, `logs/rl_action_probs_*.csv`

### Quick start
- Train: `bash train.sh`
- Evaluate best action across sellers: `bash eval.sh` (set `CUSTOM_PROMPT=False`)
- Side experiment (custom prompt): `bash eval.sh` with `CUSTOM_PROMPT=True`