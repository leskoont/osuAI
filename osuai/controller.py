"""Ввод в игру через один вызов SendInput на шаг.

Исходная версия двигала мышь тремя способами подряд (mouse_event + SendInput + pydirectinput),
спала 20 мс после движения и по 10 мс на каждое нажатие, а клавиши только «тапала» —
слайдеры держать было невозможно. Здесь: одно атомарное событие движения (с учётом смещения
монитора в виртуальном рабочем столе) + состояние клавиши «зажата/отпущена» без задержек.
"""

from __future__ import annotations

import atexit

from . import winapi as w
from .config import InputConfig


class InputController:
    def __init__(self, cfg: InputConfig):
        self.keys = [w.vk_code(k) for k in cfg.keys]
        self.alternate = cfg.alternate and len(self.keys) > 1
        self.held: int | None = None
        self.next_key = 0
        self.region = (0, 0, 1, 1)
        self.virtual = w.virtual_screen()
        atexit.register(self.release)  # никогда не оставляем зажатую клавишу

    def set_region(self, region: tuple[int, int, int, int]) -> None:
        self.region = region
        self.virtual = w.virtual_screen()

    def to_screen(self, nx: float, ny: float) -> tuple[int, int]:
        l, t, r, b = self.region
        return int(l + nx * (r - l)), int(t + ny * (b - t))

    def _move(self, x: int, y: int) -> w.INPUT:
        vx, vy, vw, vh = self.virtual
        ax = int(round((x - vx) * 65535 / max(vw - 1, 1)))
        ay = int(round((y - vy) * 65535 / max(vh - 1, 1)))
        inp = w.INPUT(type=w.INPUT_MOUSE)
        inp.u.mi = w.MOUSEINPUT(ax, ay, 0, w.MOUSEEVENTF_MOVE | w.MOUSEEVENTF_ABSOLUTE
                                | w.MOUSEEVENTF_VIRTUALDESK, 0, 0)
        return inp

    def _key(self, vk: int, up: bool) -> w.INPUT:
        if vk in (0x01, 0x02):  # кнопки мыши
            flags = {(0x01, False): w.MOUSEEVENTF_LEFTDOWN, (0x01, True): w.MOUSEEVENTF_LEFTUP,
                     (0x02, False): w.MOUSEEVENTF_RIGHTDOWN, (0x02, True): w.MOUSEEVENTF_RIGHTUP}
            inp = w.INPUT(type=w.INPUT_MOUSE)
            inp.u.mi = w.MOUSEINPUT(0, 0, 0, flags[(vk, up)], 0, 0)
            return inp
        inp = w.INPUT(type=w.INPUT_KEYBOARD)
        flags = w.KEYEVENTF_SCANCODE | (w.KEYEVENTF_KEYUP if up else 0)
        inp.u.ki = w.KEYBDINPUT(0, w.scan_code(vk), flags, 0, 0)
        return inp

    def apply(self, nx: float, ny: float, press: int) -> bool:
        """Двигает курсор и обновляет состояние клавиши. Возвращает True, если это новое нажатие."""
        inputs = [self._move(*self.to_screen(nx, ny))]
        onset = False
        if press and self.held is None:
            vk = self.keys[self.next_key]
            if self.alternate:
                self.next_key = (self.next_key + 1) % len(self.keys)
            inputs.append(self._key(vk, False))
            self.held = vk
            onset = True
        elif not press and self.held is not None:
            inputs.append(self._key(self.held, True))
            self.held = None
        w.send_inputs(inputs)
        return onset

    def release(self) -> None:
        if self.held is not None:
            w.send_inputs([self._key(self.held, True)])
            self.held = None
