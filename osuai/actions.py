"""Пространство действий и геометрия.

Действие — это индекс в плоской Q-карте формы (2, gh, gw):
    press * gh * gw + cy * gw + cx
где (cy, cx) — ячейка сетки поверх области захвата, press ∈ {0, 1} — держать ли клавишу.
Курсор ставится в центр ячейки. Сетка в 4 раза грубее входа сети
(96x128 → 24x32; при захвате playfield+поля одна ячейка = 20 osu!px, радиус круга CS4 ≈ 36 osu!px).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionSpace:
    grid_h: int
    grid_w: int

    @classmethod
    def for_obs(cls, obs_h: int, obs_w: int) -> "ActionSpace":
        return cls(obs_h // 4, obs_w // 4)

    @property
    def cells(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def n(self) -> int:
        return 2 * self.cells

    def encode(self, cy: int, cx: int, press: bool | int) -> int:
        return int(press) * self.cells + cy * self.grid_w + cx

    def decode(self, a: int) -> tuple[int, int, int]:
        press, cell = divmod(int(a), self.cells)
        cy, cx = divmod(cell, self.grid_w)
        return cy, cx, press

    def press_of(self, a: np.ndarray | int) -> np.ndarray | int:
        return a // self.cells

    def to_norm(self, a: int) -> tuple[float, float, int]:
        """Действие → (x, y) в долях области захвата [0, 1] + press."""
        cy, cx, press = self.decode(a)
        return (cx + 0.5) / self.grid_w, (cy + 0.5) / self.grid_h, press

    def from_norm(self, x: float, y: float, press: bool | int) -> int:
        cx = min(max(int(x * self.grid_w), 0), self.grid_w - 1)
        cy = min(max(int(y * self.grid_h), 0), self.grid_h - 1)
        return self.encode(cy, cx, press)


def playfield_region(client: tuple[int, int, int, int], margin_x: float, margin_y: float,
                     y_offset: float) -> tuple[int, int, int, int]:
    """Область playfield osu!stable (512x384 osu!px) с полями, в экранных координатах.

    osu!stable масштабирует playfield по min(w/640, h/480), центрирует и сдвигает вниз на y_offset.
    """
    left, top, right, bottom = client
    w, h = right - left, bottom - top
    scale = min(w / 640.0, h / 480.0)
    pw, ph = 512.0 * scale, 384.0 * scale
    px = left + (w - pw) / 2.0
    py = top + (h - ph) / 2.0 + y_offset * scale
    l = int(round(px - margin_x * scale))
    t = int(round(py - margin_y * scale))
    r = int(round(px + pw + margin_x * scale))
    b = int(round(py + ph + margin_y * scale))
    return max(l, left), max(t, top), min(r, right), min(b, bottom)
