# rl/train_bandit.py
import json
import random
import sys
import math
import collections
import time
from typing import List, Dict, Any
from typing import Optional, Callable
from rl.env import run_episode
from rl.prompt_space import ACTIONS
from rl.policy import SoftmaxPolicy
import os, csv
from datetime import datetime
import contextlib, os, sys, logging

@contextlib.contextmanager
def quiet(mute: bool = True):
    if not mute:
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    try:
        devnull_out = open(os.devnull, "w")
        devnull_err = open(os.devnull, "w")
        sys.stdout, sys.stderr = devnull_out, devnull_err
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        try:
            devnull_out.close(); devnull_err.close()
        except Exception:
            pass


def load_products(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_products_ids(path: str):
    data = load_products(path)
    ids, names = set(), set()
    for p in data:
        if "id" in p:
            ids.add(p["id"])
        if "Product Name" in p:
            names.add(p["Product Name"])
    return ids, names


def load_train_set(train_path="dataset/products.json", test_path="dataset/products_mini.json"):
    test_ids, test_names = load_products_ids(test_path)
    raw = load_products(train_path)
    filtered = []
    for p in raw:
        pid = p.get("id")
        pname = p.get("Product Name")
        if (pid is not None and pid in test_ids) or (pname is not None and pname in test_names):
            continue
        filtered.append(p)
    return filtered

def _dump_action_probs(policy: SoftmaxPolicy, csv_path: str = None, top_k: int = None) -> None:
    probs = policy._softmax()
    rows = []
    for i in range(len(ACTIONS)):
        rows.append({
            "rank": i,  # 暂存，排序后再改
            "idx": i,
            "name": ACTIONS[i]["name"],
            "prob": probs[i],
            "theta": policy.theta[i],
        })
    rows.sort(key=lambda x: x["prob"], reverse=True)
    for r, row in enumerate(rows):
        row["rank"] = r + 1
    # 打印 Top-N
    for row in rows[: (top_k or len(rows))]:
        print(f"{row['rank']:02d} | p={row['prob']:.4f} | theta={row['theta']:+.3f} | idx={row['idx']:03d} | {row['name']}")
    # 落盘 CSV（可选）
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank","idx","name","prob","theta"])
            w.writeheader()
            w.writerows(rows)

class RealtimeMeter:
    def __init__(self, window: int = 50, print_every: int = 1, total: int = None,
                 metrics_stream=sys.stdout, progress_stream=sys.stderr, min_samples: int = None):
        self.w = window
        self.buf = collections.deque(maxlen=window)
        self.t0 = time.time()
        self.print_every = print_every
        self.last_print = 0
        self.total = total
        self.metrics_stream = metrics_stream
        self.progress_stream = progress_stream
        self.min_samples = min_samples if min_samples is not None else max(10, window // 2)

    @staticmethod
    def _entropy(probs):
        return -sum(p*math.log(p+1e-12) for p in probs)

    def update(self, step, budget, reward, info, probs, top_name, top_prob):
        self.buf.append({
            "r": reward,
            "overpay_h": bool(info["anomalies"].get("overpayment", False)) and budget=="high",
            "oob_l": bool(info["anomalies"].get("out_of_budget", False)) and budget=="low",
            "dead": bool(info["deadlock"]),
            "turns": int(info["turns"]),
        })
        if len(self.buf) < self.min_samples:
            return
        N = len(self.buf)
        avg_r = sum(b["r"] for b in self.buf)/N
        std_r = (sum((b["r"]-avg_r)**2 for b in self.buf)/N)**0.5
        overpay = sum(b["overpay_h"] for b in self.buf)/N
        oob = sum(b["oob_l"] for b in self.buf)/N
        dead = sum(b["dead"] for b in self.buf)/N
        avg_t = sum(b["turns"] for b in self.buf)/N
        ent = self._entropy(probs)
        elapsed = time.time()-self.t0

        if step - self.last_print >= self.print_every:
            line = (f"step {step:6d} | bud={budget:4s} | R={reward:+.3f} | "
                    f"avgR_w={avg_r:+.3f}±{std_r:.3f} | "
                    f"overpay@high={overpay:.3f} oob@low={oob:.3f} dead={dead:.3f} | "
                    f"avgT={avg_t:.2f} | H={ent:.2f} | top={top_name[:22]} p={top_prob:.2f} | "
                    f"{elapsed:6.1f}s")
            print(line, file=self.metrics_stream, flush=True)
            self.last_print = step

    def progress(self, done: int):
        if not self.total:
            return
        pct = min(1.0, done / max(1, self.total))
        elapsed = time.time() - self.t0
        eta = elapsed * (1 - pct) / max(pct, 1e-6)
        bar_len = 30
        filled = int(pct * bar_len)
        bar = "█" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {done}/{self.total} {pct*100:5.1f}% | ETA {eta:6.1f}s",
              end="", file=self.progress_stream, flush=True)

    def newline(self):
        print("", file=self.metrics_stream)


def _warmup_coverage(
    policy: SoftmaxPolicy,
    products: List[Dict[str, Any]],
    buyer_model: str,
    seller_model: str,
    summary_model: str,
    max_turns: int,
    seed: int = 12345,
    rt: Optional[RealtimeMeter] = None,
    on_episode: Optional[Callable[[], None]] = None,
    mute_dialogue: bool = True,
):
    """
    覆盖性热身：每个动作在 high/low 各评估 1 次，共 2 * |ACTIONS| 次；在线更新。
    """
    rng = random.Random(seed)
    action_indices = list(range(len(ACTIONS)))
    rng.shuffle(action_indices)

    ptr = 0  # 轮转产品
    for budget in ["high", "low"]:
        for a in action_indices:
            product = products[ptr % len(products)]
            ptr += 1
            with quiet(mute_dialogue):
                reward, info, data = run_episode(
                    product=product,
                    budget_scenario=budget,
                    action_idx=a,
                    buyer_model=buyer_model,
                    seller_model=seller_model,
                    summary_model=summary_model,
                    max_turns=max_turns,
                )

            # 使用当前策略概率做一致的更新
            probs = policy._softmax()
            pa = probs[a] if a < len(probs) else 1.0 / len(ACTIONS)
            policy.update(a, pa, reward)

            if rt is not None:
                top_idx = max(range(len(probs)), key=lambda i: probs[i])
                rt.update(step=-1, budget=budget, reward=reward, info=info,
                          probs=probs, top_name=ACTIONS[top_idx]["name"], top_prob=probs[top_idx])
            if on_episode:
                on_episode()

            if rt is not None:
                top_idx = max(range(len(probs)), key=lambda i: probs[i])
                rt.update(
                    step=-1,  # 热身不计入主 step，可传 -1
                    budget=budget,
                    reward=reward,
                    info=info,
                    probs=probs,
                    top_name=ACTIONS[top_idx]["name"],
                    top_prob=probs[top_idx],
                )
    print("\n[Warmup] coverage done.")


def train_multi_sellers(
    seller_models: List[str],
    products_file: str = "dataset/products.json",
    test_file: str = "dataset/products_mini.json",
    buyer_model: str = "qwen2.5-7b-instruct",
    summary_model: str = "gpt-4o-mini",
    steps_per_seller: int = 200,
    max_turns: int = 20,
    seed: int = 42,
    warmup_enabled: bool = True,
    epsilon: float = 0.1,                 # 保留参数但实际用退火 dynamic_epsilon
    enforce_min_count_every: int = 20,
    mute_dialogue: bool = True,
) -> int:
    random.seed(seed)
    products = load_train_set(train_path=products_file, test_path=test_file)
    assert len(products) > 0, "Training set is empty after excluding test set."

    policy = SoftmaxPolicy(len(ACTIONS), learning_rate=0.1)

    # 进度条总步数（每个 seller ：热身 + 主训练）
    warmup_eps = 2 * len(ACTIONS) if warmup_enabled else 0
    total_eps = len(seller_models) * (warmup_eps + steps_per_seller)
    rt = RealtimeMeter(window=10, print_every=1, total=total_eps,
                    metrics_stream=sys.stdout, progress_stream=sys.stderr, min_samples=1)
    done = 0
    if mute_dialogue:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("LanguageModel").setLevel(logging.WARNING)

    counts = [0 for _ in range(len(ACTIONS))]
    global_step = 0
    total_steps_all = steps_per_seller * len(seller_models)

    for seller_model in seller_models:
        # 热身
        if warmup_enabled:
            def _tick():
                nonlocal done
                done += 1
                rt.progress(done)
            _warmup_coverage(policy, products, buyer_model, seller_model, summary_model, max_turns, rt=rt, on_episode=_tick, mute_dialogue=mute_dialogue)
            print(f"\n[Warmup] seller={seller_model} done.")

        # 构建 Top‑K 激活集合
        K = 24
        active = sorted(range(len(ACTIONS)), key=lambda i: policy.theta[i], reverse=True)[:K]
        active_set = set(active)
        print(f"[Active@start] seller={seller_model} K={K} active={active}")

        for local_step in range(1, steps_per_seller + 1):
            global_step += 1

            # 两阶段预算采样：先稳 low，再偏高权重 high
            half_steps = steps_per_seller // 2
            if local_step <= half_steps:
                budget_scenario = "low"
            else:
                budget_scenario = "high" if (random.random() < 0.7) else "low"

            # 退火 ε：0.10 → 0.02（按全局进度）
            progress = (global_step - 1) / max(1, total_steps_all - 1)
            eps0, eps1 = 0.10, 0.02
            dynamic_epsilon = eps0 + (eps1 - eps0) * progress

            # 2/3 进度再压缩 Top‑K → 12（只做一次）
            if local_step == int(steps_per_seller * 2 / 3):
                K2 = min(12, len(ACTIONS))
                active = sorted(range(len(ACTIONS)), key=lambda i: policy.theta[i], reverse=True)[:K2]
                active_set = set(active)
                print(f"[Active@shrink] seller={seller_model} K={K2} active={active}")

            probs = policy._softmax()

            # 仅在 active_set 内采样
            def sample_from_active(active_set, probs_list):
                # 规范化 active 概率
                ap = [probs_list[i] if i in active_set else 0.0 for i in range(len(ACTIONS))]
                s = sum(ap) or 1.0
                ap = [p / s for p in ap]
                r = random.random(); acc = 0.0; idx = 0
                for i, p in enumerate(ap):
                    acc += p
                    if r <= acc:
                        idx = i; break
                return idx

            if enforce_min_count_every and (local_step % enforce_min_count_every == 0):
                # 在 active_set 中挑“最少采样”的动作
                action_idx = min(active_set, key=lambda i: counts[i])
                pa = probs[action_idx]
            elif random.random() < dynamic_epsilon:
                # 在 active_set 中随机探索
                action_idx = random.choice(list(active_set))
                pa = probs[action_idx]
            else:
                # 在 active_set 内按概率采样
                action_idx = sample_from_active(active_set, probs)
                pa = probs[action_idx]

            product = random.choice(products)
            with quiet(mute_dialogue):
                reward, info, data = run_episode(
                    product=product,
                    budget_scenario=budget_scenario,
                    action_idx=action_idx,
                    buyer_model=buyer_model,
                    seller_model=seller_model,  # 当前阶段卖家
                    summary_model=summary_model,
                    max_turns=max_turns,
                )
            policy.update(action_idx, pa, reward)
            counts[action_idx] += 1

            # 实时轨迹与进度
            probs = policy._softmax()
            top_idx = max(range(len(probs)), key=lambda i: probs[i])
            rt.update(step=global_step, budget=budget_scenario, reward=reward, info=info,
                      probs=probs, top_name=ACTIONS[top_idx]["name"], top_prob=probs[top_idx])
            done += 1
            rt.progress(done)
            if global_step % 10 == 0:
                rt.newline()
                print(f"[{seller_model}] [{local_step}/{steps_per_seller}] bud={budget_scenario} "
                      f"action={ACTIONS[action_idx]['name']} R={reward:.3f} deadlock={info['deadlock']} anomalies={info['anomalies']}")

    rt.newline()

    best_idx = int(max(range(len(ACTIONS)), key=lambda i: policy.theta[i]))
    print("\n[Multi-seller] Best single action:", ACTIONS[best_idx]["name"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("logs", f"rl_action_probs_{ts}.csv")
    _dump_action_probs(policy, csv_path=csv_path, top_k=None)
    print(f"[action-probs] saved to {csv_path}")
    return best_idx