# rl/policy.py
import math
import random
from typing import List, Tuple

class SoftmaxPolicy:
    def __init__(self, num_actions: int, learning_rate: float = 0.1):
        self.theta: List[float] = [0.0 for _ in range(num_actions)]
        self.lr: float = learning_rate
        self.baseline: float = 0.0

    def _softmax(self) -> List[float]:
        m = max(self.theta)
        exps = [math.exp(t - m) for t in self.theta]
        s = sum(exps)
        return [e / s for e in exps]

    def sample(self) -> Tuple[int, float]:
        probs = self._softmax()
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i, p
        return len(probs) - 1, probs[-1]

    def update(self, action_idx: int, action_prob: float, reward: float) -> None:
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        advantage = reward - self.baseline
        grad = (1.0 - action_prob)
        self.theta[action_idx] += self.lr * advantage * grad