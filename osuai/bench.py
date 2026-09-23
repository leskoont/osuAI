"""Бенчмарк конвейера без игры: предобработка, детектор, инференс, сэмплирование, обучение.

    python main.py benchmark [--config configs/rtx2060.toml]

Помогает подобрать batch_size / replay_ratio / agent.hz под конкретное железо.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import system
from .actions import ActionSpace
from .agent import InferenceEngine
from .capture import Preprocessor
from .config import Config
from .learner import Learner
from .model import resolve_device
from .replay import ReplayBuffer
from .reward import CircleDetector
from .sim import SimEnv


def _timeit(fn, n: int, warmup: int = 5) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        fn()
        out[i] = time.perf_counter() - t
    return out * 1000


def _fmt(name: str, ms: np.ndarray) -> str:
    return (f"  {name:<34} p50 {np.percentile(ms, 50):7.3f} мс   p99 {np.percentile(ms, 99):7.3f} мс"
            f"   ({1000 / max(np.mean(ms), 1e-9):8.0f}/с)")


def run(cfg: Config, updates: int = 200) -> None:
    system.tune(cfg.system)
    device = resolve_device(cfg.train.device)
    space = ActionSpace.for_obs(cfg.obs.height, cfg.obs.width)
    print(f"Устройство: {system.describe_device(device)}; torch {torch.__version__}")
    print(f"Вход {cfg.obs.stack}x{cfg.obs.height}x{cfg.obs.width}, действий {space.n}, "
          f"батч {cfg.train.batch_size}, AMP {cfg.train.amp and device.type == 'cuda'}")
    rng = np.random.default_rng(0)

    # --- предобработка кадра playfield при 1440p (1728x1296 BGRA)
    pre = Preprocessor(cfg.obs)
    raw = rng.integers(0, 255, (1296, 1728, 4), dtype=np.uint8)
    print("Захват/предобработка:")
    print(_fmt("resize 1728x1296 BGRA → obs+det", _timeit(lambda: pre(raw), 100)))

    sim = SimEnv(cfg, seed=0)
    for _ in range(90):
        sim.act(0.5, 0.5, 0)
    frame = sim.observe()
    det = CircleDetector(cfg.detector)
    print(_fmt("детектор кругов", _timeit(lambda: det.detect(frame.color), 300)))

    # --- инференс
    engine = InferenceEngine(cfg, device)
    stack = rng.integers(0, 255, (cfg.obs.stack, cfg.obs.height, cfg.obs.width), dtype=np.uint8)
    print("Инференс актора (batch 1):")
    print(_fmt("act() без Q-карты", _timeit(lambda: engine.act(stack, 0.0), 500, 20)))
    print(_fmt("act() с Q-картой (визуализация)", _timeit(lambda: engine.act(stack, 0.0, True), 200)))

    # --- replay + обучение
    n = 20_000
    buf = ReplayBuffer(n, (cfg.obs.height, cfg.obs.width), cfg.obs.stack, cfg.train.n_step,
                       cfg.train.gamma, space.cells)
    frames = rng.integers(0, 255, (64, cfg.obs.height, cfg.obs.width), dtype=np.uint8)
    for i in range(n):
        buf.add(frames[i % 64], int(rng.integers(space.n)), float(rng.normal()), i % 1000 == 0, False)
    print("Обучение:")
    out = {k: np.empty(shape, dt) for k, (shape, dt) in buf.batch_spec(cfg.train.batch_size).items()}
    print(_fmt(f"сборка батча {cfg.train.batch_size} (фоновый поток)",
               _timeit(lambda: buf.sample(cfg.train.batch_size, rng, out), 100)))
    learner = Learner(cfg, device, buf)
    learner.make_prefetcher()
    for _ in range(10):
        learner.step_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(updates):
        learner.step_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    learner.prefetcher.stop()
    ups = updates / dt
    print(f"  апдейтов в секунду                  {ups:8.1f}  ({dt / updates * 1000:.2f} мс/апдейт)")
    need = cfg.agent.hz * cfg.train.replay_ratio
    print(f"  нужно при hz={cfg.agent.hz:g}, replay_ratio={cfg.train.replay_ratio:g}: {need:.0f}/с → "
          f"{'OK' if ups >= need * 1.3 else 'МАЛО — уменьшите batch_size/replay_ratio или model.channels'}")
    if device.type == "cuda":
        print(f"  пик памяти GPU: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    mem = cfg.train.replay_capacity * cfg.obs.height * cfg.obs.width / 2**30
    print(f"Replay на {cfg.train.replay_capacity} шагов займёт ~{mem:.1f} GiB RAM")
