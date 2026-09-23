"""Системные настройки под реальное время: потоки, GIL, таймер Windows, приоритет, cuDNN."""

from __future__ import annotations

import sys
import time

import cv2
import torch

from . import winapi
from .config import SystemConfig


def tune(cfg: SystemConfig) -> None:
    winapi.set_dpi_aware()
    winapi.begin_timer_resolution(cfg.timer_resolution_ms)
    winapi.set_process_priority(cfg.process_priority)
    # i5-10400: 6C/12T. Ограничиваем внутрипоточный параллелизм torch/OpenCV, чтобы
    # не отбирать ядра у osu! и у собственных потоков захвата/актора.
    torch.set_num_threads(max(1, cfg.torch_threads))
    cv2.setNumThreads(max(1, cfg.cv2_threads))
    cv2.setUseOptimized(True)
    # По умолчанию поток держит GIL до 5 мс — это треть кадра при 60 Гц.
    sys.setswitchinterval(cfg.switch_interval)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # форма входа фиксирована → выбираем лучшие ядра


def restore(cfg: SystemConfig) -> None:
    winapi.end_timer_resolution(cfg.timer_resolution_ms)


def sleep_until(deadline: float) -> None:
    """Точный сон: обычный sleep до ~1 мс до дедлайна, дальше короткий спин."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.0015:
            time.sleep(remaining - 0.001)
        else:
            time.sleep(0)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return f"CPU ({torch.get_num_threads()} потоков torch)"
    p = torch.cuda.get_device_properties(device)
    arch = f"sm_{p.major}{p.minor}"
    s = f"{p.name}, {p.total_memory / 2**30:.1f} GiB, {arch}"
    if arch not in torch.cuda.get_arch_list():
        s += f"  [ВНИМАНИЕ: сборка torch не содержит ядер для {arch} — поставьте wheel cu126/cu128]"
    return s
