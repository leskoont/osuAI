"""Метрики: консольная строка статуса, CSV-лог, опционально TensorBoard.

Вместо бесконечно растущих списков rewards/fps (утечка памяти в долгих сессиях) — скользящие окна.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np


class Metrics:
    def __init__(self, log_dir: str, tensorboard: bool):
        os.makedirs(log_dir, exist_ok=True)
        self.step_times: deque[float] = deque(maxlen=600)
        self.infer_times: deque[float] = deque(maxlen=600)
        self.rewards: deque[float] = deque(maxlen=600)
        self.kinds: deque[str] = deque(maxlen=600)
        self.episode_log = open(os.path.join(log_dir, "episodes.csv"), "a", newline="")
        self.episode_csv = csv.writer(self.episode_log)
        if self.episode_log.tell() == 0:
            self.episode_csv.writerow(["time", "env_steps", "updates", "steps", "reward", "h300", "h100",
                                       "h50", "miss", "accuracy", "max_combo", "eps"])
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir)
            except Exception as e:
                print(f"[metrics] TensorBoard недоступен: {e}")
        self.last_print = time.perf_counter()
        self.steps_at_print = 0

    def step(self, dt: float, infer: float, reward: float, kind: str) -> None:
        self.step_times.append(dt)
        self.infer_times.append(infer)
        self.rewards.append(reward)
        self.kinds.append(kind)

    def summary(self) -> dict[str, float]:
        st = np.array(self.step_times) * 1000 if self.step_times else np.zeros(1)
        it = np.array(self.infer_times) * 1000 if self.infer_times else np.zeros(1)
        return {
            "step_ms_p50": float(np.percentile(st, 50)), "step_ms_p99": float(np.percentile(st, 99)),
            "infer_ms_p50": float(np.percentile(it, 50)), "infer_ms_p99": float(np.percentile(it, 99)),
            "reward_avg": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "greedy_frac": (sum(k == "greedy" for k in self.kinds) / len(self.kinds)) if self.kinds else 0.0,
        }

    def episode(self, env_steps: int, updates: int, ep: dict, eps: float) -> None:
        total = ep["h300"] + ep["h100"] + ep["h50"] + ep["miss"]
        acc = (300 * ep["h300"] + 100 * ep["h100"] + 50 * ep["h50"]) / (3 * total) if total else 0.0
        ep["accuracy"] = acc
        self.episode_csv.writerow([f"{time.time():.0f}", env_steps, updates, ep["steps"],
                                   f"{ep['reward']:.3f}", ep["h300"], ep["h100"], ep["h50"], ep["miss"],
                                   f"{acc:.2f}", ep["max_combo"], f"{eps:.3f}"])
        self.episode_log.flush()
        if self.tb is not None:
            for k in ("reward", "accuracy", "miss", "max_combo"):
                self.tb.add_scalar(f"episode/{k}", ep[k], env_steps)

    def scalars(self, values: dict[str, float], step: int) -> None:
        if self.tb is not None:
            for k, v in values.items():
                self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        self.episode_log.close()
        if self.tb is not None:
            self.tb.close()
