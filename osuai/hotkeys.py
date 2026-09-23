"""Горячие клавиши через GetAsyncKeyState с детекцией фронта.

Исходная версия использовала `keyboard.is_pressed()` + `time.sleep(0.3)` прямо в игровом
цикле — каждое нажатие хоткея замораживало агента на 300 мс.
"""

from __future__ import annotations

from . import winapi as w
from .config import HotkeyConfig


class Hotkeys:
    def __init__(self, cfg: HotkeyConfig):
        self.codes = {name: w.vk_code(getattr(cfg, name)) for name in ("pause", "training", "save",
                                                                        "focus", "quit")}
        self.state = {name: w.key_down(vk) for name, vk in self.codes.items()}

    def poll(self) -> list[str]:
        fired = []
        for name, vk in self.codes.items():
            down = w.key_down(vk)
            if down and not self.state[name]:
                fired.append(name)
            self.state[name] = down
        return fired
