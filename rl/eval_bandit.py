# rl/eval_bandit.py
import json
import random
from typing import Dict, Any, List, Optional, Callable
from rl.env import run_episode
from concurrent.futures import ThreadPoolExecutor, as_completed
from rl.prompt_space import ACTIONS, build_custom_buyer_prompt
import sys, time, contextlib, os, logging, threading


@contextlib.contextmanager
def quiet(mute: bool = True):
    """
    Thread-safe-ish stdout/stderr silencer:
    - Redirects to /dev/null within the context
    - DOES NOT close the devnull handles to avoid races across threads
    """
    if not mute:
        yield
        return

    devnull_out = open(os.devnull, "w")
    devnull_err = open(os.devnull, "w")

    # Use redirectors; they restore sys.stdout/err automatically on exit
    with contextlib.redirect_stdout(devnull_out), contextlib.redirect_stderr(devnull_err):
        try:
            yield
        finally:
            # Don't close devnull here to avoid "I/O operation on closed file" in other threads
            pass


class ProgressBar:
    def __init__(self, total: int, stream=sys.stderr, bar_len: int = 30):
        self.total = max(1, int(total))
        self.stream = stream
        self.bar_len = bar_len
        self.start = time.time()
        self.done = 0

    def update(self, n: int = 1):
        self.done += n
        pct = min(1.0, self.done / self.total)
        elapsed = time.time() - self.start
        eta = elapsed * (1 - pct) / max(pct, 1e-6)
        filled = int(pct * self.bar_len)
        bar = "█" * filled + "-" * (self.bar_len - filled)
        print(f"\r[{bar}] {self.done}/{self.total} {pct*100:5.1f}% | ETA {eta:6.1f}s",
              end="", file=self.stream, flush=True)

    def finish(self):
        self.update(0)
        print("", file=self.stream, flush=True)


class SellersDashboard:
    def __init__(self, sellers: List[str], total_per_seller: int, stream=sys.stderr, bar_len: int = 30):
        self.sellers = sellers
        self.total = {s: max(1, int(total_per_seller)) for s in sellers}
        self.done = {s: 0 for s in sellers}
        self.start = {s: time.time() for s in sellers}
        self.stream = stream
        self.bar_len = bar_len
        self._lock = threading.Lock()
        self.use_clear = hasattr(stream, "isatty") and stream.isatty()

    def _bar(self, s: str) -> str:
        d = self.done[s]
        t = self.total[s]
        pct = min(1.0, d / t)
        elapsed = time.time() - self.start[s]
        eta = elapsed * (1 - pct) / max(pct, 1e-6)
        filled = int(pct * self.bar_len)
        bar = "█" * filled + "-" * (self.bar_len - filled)
        return f"{s[:18]:<18} [{bar}] {d}/{t} {pct*100:5.1f}% | ETA {eta:6.1f}s"
    
    def tick(self, seller: str, n: int = 1):
        
        with self._lock:
            self.done[seller] = min(self.total[seller], self.done[seller] + n)
            # 清屏并重绘所有 seller 的条
            if self.use_clear:
                print("\x1b[2J\x1b[H", end="", file=self.stream)
                for s in self.sellers:
                    print(self._bar(s), file=self.stream)
                self.stream.flush()
            else:
                print(self._bar(seller), file=self.stream, flush=True)


def load_products(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_aggregate(
    best_action_idx: int,
    products_file: str = "dataset/products_mini.json",
    buyer_model: str = "qwen2.5-7b-instruct",
    seller_model: str = "qwen2.5-14b-instruct",
    summary_model: str = "gpt-4o-mini",
    max_turns: int = 20,
    seed: int = 123,
    show_progress: bool = True,
    mute_dialogue: bool = True,
    on_progress: Optional[Callable[[int], None]] = None,
    save_path: Optional[str] = None,          
    episode_jsonl_path: Optional[str] = None, 
    custom_prompt: bool = False):

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LanguageModel").setLevel(logging.WARNING)

    random.seed(seed)
    products = load_products(products_file)
    num_products = len(products)
    total_episodes = num_products * 2

    pb = ProgressBar(total_episodes) if show_progress else None

    # 打开 JSONL 追加（每个 seller 各一个文件，线程安全：每个 seller 独占）
    episode_f = None
    if episode_jsonl_path:
        os.makedirs(os.path.dirname(episode_jsonl_path), exist_ok=True)
        episode_f = open(episode_jsonl_path, "a", encoding="utf-8")

    overpay_cnt = 0
    oob_cnt = 0
    deadlock_cnt = 0
    turns_sum = 0

    for budget in ["high", "low"]:
        for p in products:
            with quiet(mute_dialogue):
                # 需要 rl/env.py::run_episode 返回 (reward, info, data)
                _, info, data = run_episode(
                    product=p,
                    budget_scenario=budget,
                    action_idx=best_action_idx,
                    buyer_model=buyer_model,
                    seller_model=seller_model,
                    summary_model=summary_model,
                    max_turns=max_turns,
                    buyer_system_prompt=custom_prompt,
                )

            a = info["anomalies"]
            if budget == "high" and a.get("overpayment", False): overpay_cnt += 1
            if budget == "low"  and a.get("out_of_budget", False): oob_cnt += 1
            if info.get("deadlock", False): deadlock_cnt += 1
            turns_sum += int(info.get("turns", 0))

            # 逐行写入 episode 明细
            if episode_f:
                episode_record = {
                    "seller": seller_model,
                    "buyer": buyer_model,
                    "summary": summary_model,
                    "budget_scenario": budget,
                    "record": data,      # 含 conversation_history, offers, budget, result, models, parameters
                    "anomalies": a,
                }
                episode_f.write(json.dumps(episode_record, ensure_ascii=False) + "\n")
                episode_f.flush()

            if pb: pb.update(1)
            if on_progress: on_progress(1)

    if episode_f:
        episode_f.close()
    if pb: pb.finish()

    overpay_rate  = overpay_cnt / max(1, num_products)
    oob_rate      = oob_cnt     / max(1, num_products)
    deadlock_rate = deadlock_cnt/ max(1, total_episodes)
    avg_turns     = turns_sum   / max(1, total_episodes)

    summary = {
        "seller": seller_model,
        "num_products": num_products,
        "total_episodes": total_episodes,
        "overpayment_count": overpay_cnt,
        "oob_count": oob_cnt,
        "deadlock_count": deadlock_cnt,
        "overpayment_rate": overpay_rate,
        "oob_rate": oob_rate,
        "deadlock_rate": deadlock_rate,
        "avg_turns": avg_turns,
    }

    # 可选：同时输出一个完整JSON（包含全部 episodes），仅在需要时使用
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 大文件风险：如数据量很大建议仅用 JSONL
        with open(save_path, "w", encoding="utf-8") as f:
            out = {
                "meta": {
                    "products_file": products_file,
                    "best_action_idx": best_action_idx,
                    "best_action_name": ACTIONS[best_action_idx]["name"],
                    "buyer_model": buyer_model,
                    "seller_model": seller_model,
                    "summary_model": summary_model,
                    "max_turns": max_turns,
                },
                "summary": summary,
            }
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {save_path}")

    print(f"[aggregate] seller={seller_model} products={num_products} episodes={total_episodes}")
    print(f"[aggregate] overpayment(high): {overpay_cnt}/{num_products} = {overpay_rate:.3f}")
    print(f"[aggregate] out_of_budget(low): {oob_cnt}/{num_products} = {oob_rate:.3f}")
    print(f"[aggregate] deadlock(all): {deadlock_cnt}/{total_episodes} = {deadlock_rate:.3f}")
    print(f"[aggregate] avg_turns(all): {avg_turns:.2f}")
    return summary


def evaluate_all_sellers_parallel_models_only(
    best_action_idx: int,
    seller_models: List[str],
    products_file: str = "dataset/products_mini.json",
    buyer_model: str = "qwen2.5-7b-instruct",
    summary_model: str = "gpt-4o-mini",
    max_turns: int = 20,
    seed: int = 123,
    seller_workers: int = 4,
    show_progress: bool = True,
    save_path: Optional[str] = None,          
    summary_jsonl_path: Optional[str] = None, 
    episodes_dir: Optional[str] = None,   
    custom_prompt: bool = False, 
):
    products = load_products(products_file)
    total_per_seller = len(products) * 2
    dashboard = SellersDashboard(seller_models, total_per_seller) if show_progress else None

    results: List[Dict[str, Any]] = []

    # 打开 seller 汇总 JSONL
    summary_f = None
    if summary_jsonl_path:
        os.makedirs(os.path.dirname(summary_jsonl_path), exist_ok=True)
        summary_f = open(summary_jsonl_path, "a", encoding="utf-8")

    def _run_one(seller_model: str):
        # 每个 seller 的 episodes.jsonl
        ep_jsonl = None
        if episodes_dir:
            os.makedirs(episodes_dir, exist_ok=True)
            safe_name = seller_model.replace("/", "_").replace(" ", "_")
            ep_jsonl = os.path.join(episodes_dir, f"episodes__{safe_name}.jsonl")

        summary = evaluate_aggregate(
            best_action_idx=best_action_idx,
            products_file=products_file,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            seed=seed,
            show_progress=False,      # 内层不画条
            mute_dialogue=False,
            on_progress=(lambda n: dashboard.tick(seller_model, n)) if dashboard else None,
            save_path=None,           # 不写完整大JSON
            episode_jsonl_path=ep_jsonl,  # 逐行写 episode
            custom_prompt=custom_prompt,  # 新增：自定义 prompt 文本
        )

        # 逐 seller 追加一条汇总到 JSONL
        if summary_f:
            summary_rec = {
                "best_action_idx": best_action_idx,
                "best_action_name": ACTIONS[best_action_idx]["name"] if custom_prompt is False else "custom",
                "buyer_model": buyer_model,
                "summary_model": summary_model,
                "products_file": products_file,
                "max_turns": max_turns,
                **summary,  # 含 seller 与各率
            }
            summary_f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")
            summary_f.flush()

        return summary

    with ThreadPoolExecutor(max_workers=seller_workers) as ex:
        futs = {ex.submit(_run_one, s): s for s in seller_models}
        for fut in as_completed(futs):
            results.append(fut.result())

    if summary_f:
        summary_f.close()

    # === overall summary across all sellers ===
    num_products = len(products)
    num_sellers = len(seller_models)
    total_high = num_products * num_sellers
    total_low  = num_products * num_sellers
    total_all  = 2 * num_products * num_sellers

    sum_overpay  = sum(r["overpayment_count"] for r in results)
    sum_oob      = sum(r["oob_count"] for r in results)
    sum_deadlock = sum(r["deadlock_count"] for r in results)

    # 加权平均回合数（每个 seller 恰好有 2*num_products 个 episode）
    turns_sum_total = sum(r["avg_turns"] * (2 * num_products) for r in results)
    overall_avg_turns = turns_sum_total / max(1, total_all)

    overall_summary = {
        "num_products": num_products,
        "num_sellers": num_sellers,
        "total_high_episodes": total_high,
        "total_low_episodes": total_low,
        "total_episodes": total_all,
        "overpayment_count": sum_overpay,
        "oob_count": sum_oob,
        "deadlock_count": sum_deadlock,
        "overpayment_rate": sum_overpay / max(1, total_high),
        "oob_rate":         sum_oob     / max(1, total_low),
        "deadlock_rate":    sum_deadlock/ max(1, total_all),
        "avg_turns": overall_avg_turns,
    }

    print("[overall] overpay={:.3f} oob={:.3f} dead={:.3f} avgT={:.2f}".format(
        overall_summary["overpayment_rate"],
        overall_summary["oob_rate"],
        overall_summary["deadlock_rate"],
        overall_summary["avg_turns"],
    ))

    # 可选：一个总汇总JSON（不含 episodes）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = {
            "meta": {
                "products_file": products_file,
                "best_action_idx": best_action_idx,
                "best_action_name": ACTIONS[best_action_idx]["name"] if custom_prompt is False else "custom",
                "buyer_model": buyer_model,
                "summary_model": summary_model,
                "max_turns": max_turns,
                "seller_models": seller_models,
                "summary_jsonl": summary_jsonl_path,
                "episodes_dir": episodes_dir,
            },
            "seller_results": results,
            "overall_summary": overall_summary,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {save_path}")

    if results:
        worst_overpay = max(r["overpayment_rate"] for r in results)
        worst_oob     = max(r["oob_rate"]         for r in results)
        worst_dead    = max(r["deadlock_rate"]    for r in results)
        print(f"[aggregate-all(models-only)] worst_overpay={worst_overpay:.3f} "
              f"worst_oob={worst_oob:.3f} worst_dead={worst_dead:.3f}")
    return results