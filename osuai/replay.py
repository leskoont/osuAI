"""Replay-буфер с предвыделенной памятью uint8.

Хранится только новейший кадр каждого шага (а не весь стек) — в `stack` раз меньше памяти:
300k шагов 96x128 ≈ 3.7 GB вместо 14.7 GB (и вместо ~39 GB float64 в исходной версии).
Стек кадров и n-step возвраты собираются векторно при сэмплировании.

Один писатель (поток актора) и сколько угодно читателей без блокировок: писатель сначала
пишет данные, потом увеличивает `total`; читатели не трогают `gap` самых старых слотов,
которые писатель может перезаписать во время сэмплирования.
"""

from __future__ import annotations

import os

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, int], stack: int, n_step: int,
                 gamma: float, cells: int, gap: int = 256):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.stack = stack
        self.n_step = n_step
        self.gamma = gamma
        self.cells = cells
        self.gap = gap
        self.frames = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.first = np.zeros(self.capacity, dtype=bool)
        self.terminal = np.zeros(self.capacity, dtype=bool)
        self.total = 0  # глобальный счётчик записанных шагов
        self._disc = (gamma ** np.arange(n_step)).astype(np.float32)

    def __len__(self) -> int:
        return min(self.total, self.capacity)

    def add(self, frame: np.ndarray, action: int, reward: float, first: bool, terminal: bool) -> None:
        i = self.total % self.capacity
        self.frames[i] = frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.first[i] = first
        self.terminal[i] = terminal
        self.total += 1  # публикуем шаг только после записи данных

    def mark_terminal_last(self) -> None:
        if self.total:
            self.terminal[(self.total - 1) % self.capacity] = True

    # ------------------------------------------------------------------ sampling
    def valid_range(self, total: int | None = None) -> tuple[int, int]:
        total = self.total if total is None else total
        size = min(total, self.capacity)
        oldest = total - size
        lo = oldest + max(self.stack - 1, 1) + (self.gap if total > self.capacity else 0)
        hi = total - self.n_step  # c = i + n <= total - 1
        return lo, hi

    def can_sample(self) -> bool:
        lo, hi = self.valid_range()
        return hi > lo

    def batch_spec(self, batch: int) -> dict[str, tuple[tuple[int, ...], type]]:
        st = (batch, self.stack, *self.obs_shape)
        return {"obs": (st, np.uint8), "press": ((batch,), np.float32), "action": ((batch,), np.int64),
                "ret": ((batch,), np.float32), "done": ((batch,), np.float32),
                "next_obs": (st, np.uint8), "next_press": ((batch,), np.float32)}

    def _stack_src(self, c: np.ndarray) -> np.ndarray:
        """Глобальные индексы кадров стека состояния c: (B, stack).
        Кадры до начала эпизода заменяются первым кадром эпизода (как делает актор)."""
        src = np.empty((len(c), self.stack), dtype=np.int64)
        cur = c.astype(np.int64, copy=True)
        src[:, -1] = cur
        for o in range(1, self.stack):
            cur = np.where(self.first[cur % self.capacity], cur, cur - 1)
            src[:, -1 - o] = cur
        return src

    def stack_at(self, c: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty((len(c), self.stack, *self.obs_shape), dtype=np.uint8)
        # одна векторная выборка прямо в выходной (pinned) буфер; mode="wrap" = индекс % capacity
        np.take(self.frames, self._stack_src(c), axis=0, out=out, mode="wrap")
        return out

    def prev_press_at(self, c: np.ndarray) -> np.ndarray:
        cap = self.capacity
        press = (self.actions[(c - 1) % cap] // self.cells).astype(np.float32)
        return np.where(self.first[c % cap], 0.0, press).astype(np.float32)

    def sample(self, batch: int, rng: np.random.Generator,
               out: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray] | None:
        total = self.total
        lo, hi = self.valid_range(total)
        if hi <= lo:
            return None
        idx = rng.integers(lo, hi, size=batch)
        return self.gather(idx, out)

    def gather(self, idx: np.ndarray, out: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Собирает батч. `out` — заранее выделенные (например, pinned) массивы: без аллокаций
        и page faults на каждом шаге (иначе сборка батча в 10 раз медленнее)."""
        if out is None:
            out = {k: np.empty(shape, dtype) for k, (shape, dtype) in self.batch_spec(len(idx)).items()}
        cap = self.capacity
        ret = np.zeros(len(idx), dtype=np.float32)
        alive = np.ones(len(idx), dtype=bool)
        for k in range(self.n_step):
            j = (idx + k) % cap
            ret += np.where(alive, self._disc[k] * self.rewards[j], 0.0).astype(np.float32)
            # эпизод закончился после шага j (терминально) или следующий шаг — начало нового эпизода.
            # Обрыв эпизода (пауза) тоже считаем концом: при gamma=0.95 смещение пренебрежимо.
            stop = alive & (self.terminal[j] | self.first[(idx + k + 1) % cap])
            alive &= ~stop
        nxt = idx + self.n_step
        self.stack_at(idx, out["obs"])
        self.stack_at(nxt, out["next_obs"])
        out["press"][:] = self.prev_press_at(idx)
        out["next_press"][:] = self.prev_press_at(nxt)
        out["action"][:] = self.actions[idx % cap]
        out["ret"][:] = ret
        out["done"][:] = ~alive
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        total = self.total
        size = min(total, self.capacity)
        order = np.arange(total - size, total) % self.capacity
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, frames=self.frames[order], actions=self.actions[order],
                 rewards=self.rewards[order], first=self.first[order], terminal=self.terminal[order],
                 obs_shape=np.array(self.obs_shape), cells=np.array(self.cells))
        os.replace(tmp, path)

    def load(self, path: str) -> int:
        data = np.load(path)
        if tuple(data["obs_shape"]) != self.obs_shape or int(data["cells"]) != self.cells:
            raise ValueError(f"{path}: несовместимая форма наблюдений/действий")
        n = len(data["actions"])
        start = max(0, n - self.capacity)
        m = n - start
        if m == 0:
            return 0
        slots = (self.total + np.arange(m)) % self.capacity
        self.frames[slots] = data["frames"][start:]
        self.actions[slots] = data["actions"][start:]
        self.rewards[slots] = data["rewards"][start:]
        self.first[slots] = data["first"][start:]
        self.terminal[slots] = data["terminal"][start:]
        self.first[slots[0]] = True  # начало загруженного куска — всегда граница эпизода
        self.total += m
        return m
