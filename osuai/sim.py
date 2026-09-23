"""Мини-симулятор osu! (только hit circles) для отладки всего конвейера без игры.

Рисует круги с approach circles на чёрном фоне, судит нажатия по правилам osu!
(note lock, окна 300/100/50, промах после окна) и отдаёт те же счётчики, что tosu.
Позволяет за минуты убедиться, что агент действительно учится попадать, прежде чем
тратить часы на реальную игру. Есть «автопилот» для записи демонстраций.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .capture import Frame, Preprocessor
from .config import Config
from .reward import GameCounters

_COLORS = [(255, 128, 64), (64, 200, 255), (128, 255, 128), (200, 100, 255)]


@dataclass
class Note:
    t: float     # время удара, мс
    x: float     # центр в долях области
    y: float
    color: int
    judged: bool = False
    hit: bool = False


class SimEnv:
    has_events = True

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.sc = cfg.sim
        self.realtime = cfg.sim.realtime
        self.rng = np.random.default_rng(seed)
        self.w = cfg.obs.width * cfg.obs.detector_scale
        self.h = cfg.obs.height * cfg.obs.detector_scale
        self.aspect = self.h / self.w
        self.pre = Preprocessor(cfg.obs)
        self.dt = 1000.0 / cfg.agent.hz
        # playfield внутри области захвата с полями (как в реальном режиме)
        cc = cfg.capture
        tw, th = 512 + 2 * cc.margin_x, 384 + 2 * cc.margin_y
        self.pf = (cc.margin_x / tw, cc.margin_y / th, (cc.margin_x + 512) / tw, (cc.margin_y + 384) / th)
        self.fid = 0
        self.cursor = (0.5, 0.5)
        self.held = False
        self._press_t = -1e9
        self._ended = False
        self.new_map()

    # ------------------------------------------------------------------ map
    def new_map(self) -> None:
        sc, rng = self.sc, self.rng
        bpm = rng.uniform(sc.bpm_min, sc.bpm_max)
        beat = 60000.0 / bpm
        t = 1000.0
        x0, y0, x1, y1 = self.pf
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        self.notes: list[Note] = []
        i = 0
        while t < sc.map_seconds * 1000:
            self.notes.append(Note(t, x, y, (i // 4) % len(_COLORS)))
            t += beat * rng.choice([1.0, 0.5, 1.0, 2.0])
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.1, 0.35)
            x = float(np.clip(x + dist * np.cos(ang), x0, x1))
            y = float(np.clip(y + dist * np.sin(ang) / self.aspect, y0, y1))
            i += 1
        self.t = -500.0
        self.next_idx = 0  # первая несуженная нота (note lock)
        self.counters = GameCounters()
        self._ended = False

    # ------------------------------------------------------------------ env API
    def status(self) -> tuple[bool, str]:
        if self._ended:
            self.new_map()
            return False, "карта завершена"
        return True, ""

    def observe(self, after: int = 0) -> Frame:
        img = np.zeros((self.h, self.w, 3), np.uint8)
        r = int(round(self.sc.circle_radius * self.w))
        ap = self.sc.approach_ms
        visible = [n for n in self.notes[self.next_idx:self.next_idx + 8]
                   if not n.judged and n.t - ap <= self.t]
        for n in reversed(visible):  # ранние ноты рисуются поверх поздних, как в osu!
            c = (int(n.x * self.w), int(n.y * self.h))
            cv2.circle(img, c, r, _COLORS[n.color], -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (255, 255, 255), 2, cv2.LINE_AA)
            if n.t > self.t:
                ar = int(r * (1.0 + 3.0 * (n.t - self.t) / ap))
                cv2.circle(img, c, ar, (255, 255, 255), 2, cv2.LINE_AA)
        cx, cy = int(self.cursor[0] * self.w), int(self.cursor[1] * self.h)
        cv2.circle(img, (cx, cy), 3, (0, 220, 255), -1)
        gray, color = self.pre(img)
        self.fid += 1
        return Frame(gray, color, time.perf_counter(), self.fid)

    def act(self, nx: float, ny: float, press: int) -> bool:
        onset = bool(press) and not self.held
        self.held = bool(press)
        self.cursor = (nx, ny)
        if onset:
            self._judge_click()
        self._advance()
        return onset

    def _judge_click(self) -> None:
        win = self.sc.hit_window_ms
        r = self.sc.circle_radius
        for n in self.notes[self.next_idx:self.next_idx + 4]:
            if n.judged:
                continue
            if abs(self.t - n.t) > win:
                break  # note lock: раньше окна — клик игнорируется
            d = np.hypot(self.cursor[0] - n.x, (self.cursor[1] - n.y) * self.aspect)
            if d > r:
                break
            err = abs(self.t - n.t)
            if err <= win / 3:
                self.counters.h300 += 1
            elif err <= 2 * win / 3:
                self.counters.h100 += 1
            else:
                self.counters.h50 += 1
            n.judged = n.hit = True
            self.counters.combo += 1
            break

    def _advance(self) -> None:
        self.t += self.dt
        win = self.sc.hit_window_ms
        while self.next_idx < len(self.notes):
            n = self.notes[self.next_idx]
            if n.judged:
                self.next_idx += 1
            elif self.t > n.t + win:
                n.judged = True
                self.counters.miss += 1
                self.counters.combo = 0
                self.next_idx += 1
            else:
                break
        if self.next_idx >= len(self.notes):
            self._ended = True

    def game_counters(self) -> GameCounters:
        c = self.counters
        return GameCounters(c.h300, c.h100, c.h50, c.miss, c.slider_breaks, c.combo)

    def human_step(self) -> tuple[float, float, int, bool]:
        """Режим записи: вместо человека играет автопилот (идеальный игрок)."""
        nx, ny, press = self.cursor[0], self.cursor[1], 0
        for n in self.notes[self.next_idx:]:
            if not n.judged:
                nx, ny = n.x, n.y
                press = int(abs(self.t - n.t) <= self.dt * 0.6 and not self.held)
                break
        # как человек: держим клавишу ~50 мс после нажатия, затем отпускаем
        if not press and self.held and self.t - self._press_t < 50.0:
            press = 1
            nx, ny = self.cursor
        onset = self.act(nx, ny, press)
        if onset:
            self._press_t = self.t
        return nx, ny, press, onset

    def focus(self) -> bool:
        return True

    def release(self) -> None:
        self.held = False

    def close(self) -> None:
        pass
