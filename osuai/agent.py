"""Инференс актора: fp16 + channels_last + CUDA Graph на высокоприоритетном CUDA-потоке.

Исходная версия вызывала `model.predict()` на каждом кадре — это ~20-50 мс оверхеда Keras
(и на Windows TF >= 2.11 вообще работает только на CPU). Здесь один шаг — это:
копия 4x96x128 uint8 (48 KiB) из pinned-памяти → replay готового CUDA Graph (весь forward
+ argmax одной командой) → копия одного int64 обратно. На RTX 2060 это ~0.3-0.6 мс.
"""

from __future__ import annotations

import numpy as np
import torch

from .actions import ActionSpace
from .config import Config
from .model import build_model


class InferenceEngine:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.cuda = device.type == "cuda"
        self.space = ActionSpace.for_obs(cfg.obs.height, cfg.obs.width)
        self.model = build_model(cfg.obs, cfg.model, device, half=self.cuda,
                                 channels_last=cfg.train.channels_last).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = list(self.model.parameters())
        self._graph = None
        shape = (1, cfg.obs.stack, cfg.obs.height, cfg.obs.width)
        if self.cuda:
            # priority=-1 — высокий приоритет: ядра актора обгоняют ядра обучения в очереди GPU
            self.stream = torch.cuda.Stream(device=device, priority=-1)
            self.host_frames = torch.zeros(shape, dtype=torch.uint8).pin_memory()
            self.host_press = torch.zeros(1, dtype=torch.float32).pin_memory()
            self.host_action = torch.zeros(1, dtype=torch.int64).pin_memory()
            self.host_q = torch.zeros(self.space.n, dtype=torch.float16).pin_memory()
            self.host_frames_np = self.host_frames.numpy()
            self.host_press_np = self.host_press.numpy()
            self.host_action_np = self.host_action.numpy()
            self.host_q_np = self.host_q.numpy()
            self.dev_frames = torch.zeros(shape, dtype=torch.uint8, device=device)
            self.dev_press = torch.zeros(1, dtype=torch.float32, device=device)
            self._prepare(cfg.train.cuda_graphs)

    # ------------------------------------------------------------------ setup
    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.model(self.dev_frames, self.dev_press)
        return q, q.argmax(dim=1)

    def _prepare(self, use_graph: bool) -> None:
        s = self.stream
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):  # прогрев: cudnn.benchmark выбирает алгоритмы до захвата графа
                self.out_q, self.out_a = self._forward()
        s.synchronize()
        if not use_graph:
            return
        try:
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g, stream=s):
                self.out_q, self.out_a = self._forward()
            self._graph = g
        except Exception as e:  # pragma: no cover - зависит от драйвера
            print(f"[agent] CUDA Graph недоступен ({e}), работаем без него")
            self._graph = None
        s.synchronize()

    # ------------------------------------------------------------------ inference
    def act(self, frames: np.ndarray, prev_press: float, want_q: bool = False
            ) -> tuple[int, np.ndarray | None]:
        """frames: (stack, H, W) uint8. Возвращает жадное действие и (опционально) Q-карту."""
        if not self.cuda:
            with torch.inference_mode():
                q = self.model(torch.from_numpy(frames)[None], torch.tensor([float(prev_press)]))
                a = int(q.argmax(dim=1).item())
                return a, (q[0].numpy() if want_q else None)
        self.host_frames_np[0] = frames
        self.host_press_np[0] = prev_press
        with torch.cuda.stream(self.stream):
            self.dev_frames.copy_(self.host_frames, non_blocking=True)
            self.dev_press.copy_(self.host_press, non_blocking=True)
            if self._graph is not None:
                self._graph.replay()
            else:
                with torch.no_grad():
                    self.out_q, self.out_a = self._forward()
            self.host_action.copy_(self.out_a, non_blocking=True)
            if want_q:
                self.host_q.copy_(self.out_q[0], non_blocking=True)
            self.stream.synchronize()
        return int(self.host_action_np[0]), (self.host_q_np.astype(np.float32) if want_q else None)

    # ------------------------------------------------------------------ weights
    @torch.no_grad()
    def load_params(self, src: list[torch.Tensor]) -> None:
        """Копирует веса на месте (адреса тензоров не меняются → CUDA Graph остаётся валидным)."""
        if self.cuda:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))  # src мог писаться на default
            with torch.cuda.stream(self.stream):
                torch._foreach_copy_(self.params, src)
            self.stream.synchronize()
        else:
            torch._foreach_copy_(self.params, src)

    def load_state_dict(self, sd: dict) -> None:
        self.load_params([sd[k] for k, _ in self.model.named_parameters()])


class EpsilonSchedule:
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start, self.end, self.decay = start, end, max(1, decay_steps)

    def __call__(self, step: int) -> float:
        frac = min(1.0, step / self.decay)
        return self.start + (self.end - self.start) * frac


def explore(greedy: int, eps: float, space: ActionSpace, rng: np.random.Generator,
            guide: int | None, guided_prob: float, press_prob: float) -> tuple[int, str]:
    """ε-greedy с подсказками: при исследовании с вероятностью guided_prob берём действие
    эвристического «учителя» (детектор кругов), иначе — случайная ячейка/нажатие."""
    if rng.random() >= eps:
        return greedy, "greedy"
    if guide is not None and rng.random() < guided_prob:
        return guide, "guided"
    cell = int(rng.integers(space.cells))
    press = int(rng.random() < press_prob)
    return press * space.cells + cell, "random"
