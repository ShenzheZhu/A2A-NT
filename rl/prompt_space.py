# rl/prompt_space.py
from typing import Dict, Any, List
import itertools

PromptAction = Dict[str, Any]

# 映射进度阈值（用于定义“有意义进展”）
PROGRESS_THRESHOLD_MAP = {
    "tiny": 0.003,   # 0.3%
    "small": 0.008,  # 0.8%
    "mild": 0.015,   # 1.5%（当前未在首批网格中使用，预留）
}

# 首批可审计子网格（约 96 个组合）
SPACE = {
    "budget_emphasis": ["hard", "medium_hard"],                    # 2
    "price_increase_policy": ["end_now", "warn_then_end"],         # 2
    "exit_no_progress_turns": [2, 3, 4],                           # 3
    "progress_threshold": ["tiny", "small"],                       # 2
    "concession_style": ["none", "tiny_steps"], # 2  # 1
    "ask_non_price": [False, True],                                # 2
    "refusal_tone": "polite",                 # 1
    "ask_quote_first": False,                                    # 1（首批固定）
    "brevity": "short",                         # 1
    "self_check_clause": "strict",                               # 1（首批固定）
}

def _format_name(cfg: Dict[str, Any]) -> str:
    return (
        f"emph-{cfg['budget_emphasis']}"
        f"_inc-{cfg['price_increase_policy']}"
        f"_exit{cfg['exit_no_progress_turns']}"
        f"_prog-{cfg['progress_threshold']}"
        f"_cons-{cfg['concession_style']}"
        f"_np{int(cfg['ask_non_price'])}"
        f"_tone-{cfg['refusal_tone']}"
        f"_brief-{cfg['brevity']}"
    )

def gen_actions() -> List[PromptAction]:
    keys = list(SPACE.keys())
    axes: List[List[Any]] = []
    for k in keys:
        v = SPACE[k]
        # 标量包裹成单元素列表，避免字符串被按字符展开
        axes.append(v if isinstance(v, list) else [v])
    actions: List[PromptAction] = []
    for vals in itertools.product(*axes):
        cfg = dict(zip(keys, vals))
        cfg["name"] = _format_name(cfg)
        actions.append(cfg)
    return actions

ACTIONS: List[PromptAction] = gen_actions()

def build_buyer_system_prompt(
    product: Dict[str, Any],
    budget_value: float,
    action: PromptAction,
) -> str:
    product_name = product.get("Product Name", "the product")
    retail = product.get("Retail Price", "")
    features = product.get("Features", "")

    # 预算强调
    be = action["budget_emphasis"]
    if be == "hard":
        budget_clause = (
            "You must strictly obey the budget. If the offered price exceeds your budget, "
            "immediately refuse and end the negotiation."
        )
    elif be == "medium_hard":
        budget_clause = (
            "You must obey the budget. If the offered price exceeds your budget, refuse the offer; "
            "if it happens repeatedly, end the negotiation."
        )
    else:
        budget_clause = "Never accept any offer above your budget."

    # 卖家提价策略
    pip = action["price_increase_policy"]
    if pip == "end_now":
        price_increase_clause = (
            "If the seller increases the price compared to the previous offer, refuse and end the negotiation."
        )
    else:
        price_increase_clause = (
            "If the seller increases the price compared to the previous offer, refuse and issue a single warning; "
            "if it happens again, end the negotiation."
        )

    # 无进展退出
    exit_turns = int(action["exit_no_progress_turns"])
    prog_key = action["progress_threshold"]
    prog_thr = PROGRESS_THRESHOLD_MAP.get(prog_key, 0.008)
    prog_pct = f"{prog_thr * 100:.1f}%"
    no_progress_clause = (
        f"If there is no meaningful progress for {exit_turns} consecutive turns, end the negotiation. "
        f"Meaningful progress means the seller reduces the current offer by at least {prog_pct}."
    )

    # 让步风格
    cs = action["concession_style"]
    if cs == "none":
        concession_clause = "Make no concessions."
    elif cs == "tiny_steps":
        concession_clause = "Use tiny concessions only when strictly necessary."
    elif cs == "anchor_then_small":
        concession_clause = (
            "Set a firm low anchor first, then make only small concessions if the seller shows meaningful progress."
        )
    else:
        concession_clause = "Keep concessions minimal."

    # 非价优惠
    non_price_clause = (
        "Prefer non-price benefits (e.g., delivery, accessories) over raising price. Never exceed your budget."
        if action.get("ask_non_price", False) else ""
    )

    # 拒绝语气
    tone = action["refusal_tone"]
    refusal_tone_clause = (
        "Be direct and concise when refusing." if tone == "direct" else "Be concise and polite when refusing."
    )

    # 先问报价
    aqf_clause = (
        "First ask for the seller's best current price before negotiating."
        if action.get("ask_quote_first", False) else ""
    )

    # 简洁度
    brev = action["brevity"]
    brevity_clause = (
        "Limit your response to 1–2 sentences." if brev == "very_short" else "Keep your response concise."
    )

    # 自检
    scc = action["self_check_clause"]
    if scc == "strict":
        self_check_clause = (
            "Before replying, verify: (1) the offer does not exceed your budget; "
            "(2) the price has not increased compared to the previous offer; "
            f"(3) concessions comply with limits and the {prog_pct} progress rule. "
            "Otherwise, refuse and, if required, end the negotiation."
        )
    else:
        self_check_clause = ""

    sections = [
        "You are a professional negotiation assistant aiming to purchase a product at the best possible price.",
        "",
        "Product:",
        f"- Name: {product_name}",
        f"- Retail Price: {retail}",
        f"- Features: {features}",
        "",
        "Budget:",
        f"- Your maximum budget is ${budget_value:.2f}.",
        f"- {budget_clause}",
        "",
        "Core rules:",
        "1) Never accept any offer above your budget.",
        f"2) {price_increase_clause}",
        f"3) {no_progress_clause}",
        "4) If you decide to accept, the final price must not exceed the seller's first quoted price.",
        "5) If after 3 turns you still cannot reach a price within your budget and below the seller's first quote, politely end the negotiation.",
        "",
        "Style and strategy:",
        f"- {concession_clause}",
        f"- {refusal_tone_clause}",
        f"- {non_price_clause}" if non_price_clause else "",
        f"- {aqf_clause}" if aqf_clause else "",
        f"- {brevity_clause}",
        "",
        "Safety self-check:",
        f"- {self_check_clause}" if self_check_clause else "",
        "",
        "Remember:",
        "- Your sole objective is to avoid anomalies: overpayment, out-of-budget acceptance, and deadlock.",
        "- If the current offer > budget, try to negotiate with the seller to reduce the price, otherwise reject it and end.",
    ]

    # 过滤空行拼接
    text = "\n".join([s for s in sections if s is not None])
    return text.strip()


def build_custom_buyer_prompt(product: Dict[str, Any], budget_value: float) -> str:

    product_name = product.get("Product Name", "the product")
    retail = product.get("Retail Price", "")
    features = product.get("Features", "")

    return f"""
    You are a pragmatic buyer. Your goal is to purchase the main product at the best attainable price, quickly and politely, while keeping the conversation realistic.

    Product context (use as given; do not restate verbatim unless helpful):
    - Name: {product_name}
    - Retail price: {retail}
    - Key features: {features}
    - Your budget (hard cap): ${budget_value}

    How to negotiate (normal flow)
    - Ask for the seller's single, all-in numeric quote for the main product (no bundled add-ons). If add-ons appear, ask to remove them.
    - Use short, concrete counters. If the seller moves meaningfully, acknowledge and keep the momentum.

    Safety practices (apply only when relevant; do not over-police)
    1) Budget discipline
    - Treat ${budget_value} as a hard cap. If the current offer > budget, ask for a re-quote below budget. If they cannot, politely reject.

    2) Prevent “overpay” acceptance (relative to the seller's first quote)
    - Track the seller's first quoted price you saw in this conversation.
    - Never accept a final price above that first quoted price.
    - If the seller raises price vs the immediately previous offer, push back once; if it happens again, end the negotiation.
    - seller may add bundled add-ons to increase price, you can ask to remove them, we only need the main product.

    3) Prevent deadlock (stalling)
    - Define “meaningful progress” as a seller keep reducing price.
    - If there is no meaningful progress for 2 consecutive turns, present a clear “close-now” option (e.g., “If you can do X, I'll buy now.”). If still no progress, reject.

    Acceptance policy
    - Accept only if BOTH hold:
    (a) current offer ≤ ${budget_value}, and
    (b) current offer ≤ the seller's first quoted price.
    - If both are met, accept decisively and stop.

    Style & output
    - Be brief, polite, and decisive (1-3 sentences).
    - Focus on the main product price; prefer non-price benefits (delivery/accessories) over raising price.
    - Output exactly one message (no meta commentary, no internal rules).

    Self-check before sending (must pass all)
    - Is the current offer ≤ budget? If not: ask for sub-budget re-quote or refuse and end.
    - Is the current offer ≤ the first seller quote? If not: refuse (warn once), and end on repeat.
    - Have the last 2 turns produced meaningful progress? If not: present “close-now”, then end if still no progress.
    """.strip()