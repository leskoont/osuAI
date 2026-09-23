"""Награды: точные события игры (tosu) + дешёвая эвристика по кадру (детектор).

Что было не так в исходной версии и почему переписано:
* 6 вызовов Tesseract OCR на кадр (parse_game_stats вызывался дважды) — каждый 50-200 мс,
  FPS падал до ~1. Области OCR к тому же не совпадали с интерфейсом osu! (счёт справа сверху).
  Теперь точные попадания/промахи/комбо читаются из памяти игры через tosu (gosumemory API).
* HoughCircles и 4 прохода по цветам на полном кадре 2560x1440 — дважды на кадр.
  Теперь один порог яркости + connectedComponentsWithStats на кадре 256x192 (~0.2 мс).
* кадр был RGB, а детектор конвертировал его как BGR — цвета путались.
* награда всегда ≥ 0 (clip в [0, 1]) — промахи не наказывались.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig, RewardConfig


@dataclass
class Detection:
    x: float        # центр в долях ширины/высоты области захвата
    y: float
    r: float        # радиус круга в долях ширины
    urgency: float  # 0 — только появился, 1 — пора нажимать


class CircleDetector:
    """Находит hit circles и оценивает срочность нажатия по approach circle.

    1. Маска ярких пикселей (V = max(B,G,R) > порога). Лучше всего с Background dim 100%.
    2. Distance transform: центр круга — локальный максимум расстояния до фона, значение
       максимума ≈ радиус. Слипшиеся круги (стримы, стэки) дают отдельные максимумы,
       а тонкие кольца (approach circles) и курсор — маленькие значения и отсекаются.
    3. Радиус approach circle — лучами из центра: внешний край первого яркого отрезка за краем
       круга, кольцо = группа из ≥4 лучей с одинаковой дистанцией (соседи дают разброс).
    4. Радиус круга одинаков для всей карты (CS): по «чистым» детекциям держим его оценку и
       измеряем кольцо, уже слившееся с кругом в одно пятно (последние ~50 мс до удара).
    Срочность = 1 - (R_кольца / R_круга - 1) / 3: 0 при появлении (кольцо в 4 раза больше), 1 — жать.
    """

    N_RAYS = 16

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.r_est: float | None = None
        self.prev: list[Detection] = []  # детекции прошлого кадра (для объектов без видимого кольца)
        ang = np.linspace(0, 2 * np.pi, self.N_RAYS, endpoint=False)
        self._dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def reset(self) -> None:
        self.prev = []

    def _ring_ratio(self, mask: np.ndarray, cx: float, cy: float, r: float) -> float | None:
        h, w = mask.shape
        radii = np.arange(r + 1.5, r * 4.3 + 2.0, 1.0, dtype=np.float32)  # сразу за краем круга
        xs = np.rint(cx + self._dirs[:, :1] * radii).astype(np.int32)
        ys = np.rint(cy + self._dirs[:, 1:] * radii).astype(np.int32)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        vals = mask[ys.clip(0, h - 1), xs.clip(0, w - 1)].astype(bool) & inside
        hit = vals.any(axis=1)
        if hit.sum() < 4:
            return None
        # внешний край кольца: конец первого яркого отрезка вдоль луча. В скинах osu! approach
        # circle при ударе совпадает с кругом именно по внешнему краю (масштаб 4x → 1x).
        start = vals.argmax(axis=1)
        after = ~vals & (np.arange(len(radii))[None, :] > start[:, None])
        end = np.where(after.any(axis=1), after.argmax(axis=1), len(radii))
        first = np.sort(radii[end[hit] - 1] + 0.5)
        tol = max(1.5, 0.06 * r)
        # самая большая группа лучей с почти одинаковой дистанцией — это кольцо
        j = np.searchsorted(first, first + 2 * tol, side="right")
        k = int(np.argmax(j - np.arange(len(first))))
        if j[k] - k < 4:
            return None
        return float(np.median(first[k:j[k]])) / r

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        v = cv2.max(cv2.max(bgr[..., 0], bgr[..., 1]), bgr[..., 2])  # HSV V без полной конверсии
        _, mask = cv2.threshold(v, self.cfg.brightness_threshold, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        min_r = self.cfg.min_radius * w
        peaks = (dist >= cv2.dilate(dist, self._kernel)) & (dist >= min_r) & (dist <= self.cfg.max_radius * w)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return []
        vals = dist[ys, xs]
        order = np.argsort(-vals)[:64]
        centers: list[tuple[float, float, float]] = []
        for i in order:  # NMS: плато и соседние пиксели одного круга
            x, y, r = float(xs[i]), float(ys[i]), float(vals[i])
            if all((x - cx) ** 2 + (y - cy) ** 2 > (0.8 * max(r, cr)) ** 2 for cx, cy, cr in centers):
                centers.append((x, y, r))
            if len(centers) >= 12:
                break
        out: list[Detection] = []
        for cx, cy, rp in centers:
            rp += 0.5  # distance transform меряет до первого фонового пикселя
            ratio = self._ring_ratio(mask, cx, cy, rp)
            if ratio is not None and ratio > 1.4:  # кольцо далеко — максимум DT = чистый радиус круга
                self.r_est = rp if self.r_est is None else 0.95 * self.r_est + 0.05 * rp
            if ratio is None and self.r_est is not None and rp > self.r_est * 1.06:
                ratio = rp / self.r_est  # кольцо слилось с кругом: пятно раздуто
            if self.r_est is not None and rp < 0.6 * self.r_est:
                continue  # что-то мелкое (курсор, цифра, хвост слайдера)
            if ratio is not None:
                urgency = float(np.clip(1.0 - (ratio - 1.0) / 3.0, 0.0, 1.0))
            else:
                # кольца не видно: либо оно только что сомкнулось (на прошлом кадре было почти
                # сомкнуто), либо не распознано (новый объект, наложение) — тогда не торопимся
                x, y, r = cx / w, cy / h, rp / w
                before = [p for p in self.prev if (p.x - x) ** 2 + ((p.y - y) * h / w) ** 2 < (0.5 * r) ** 2]
                if not before:
                    urgency = 0.5
                else:
                    urgency = 1.0 if before[0].urgency >= 0.85 else before[0].urgency
            out.append(Detection(cx / w, cy / h, rp / w, urgency))
        self.prev = out
        return out


def most_urgent(dets: list[Detection]) -> Detection | None:
    return max(dets, key=lambda d: d.urgency) if dets else None


class RewardShaper:
    """Плотная эвристическая награда по детекциям (помогает на старте обучения)."""

    def __init__(self, cfg: RewardConfig, aspect: float):
        self.cfg = cfg
        self.aspect = aspect  # высота/ширина области захвата — расстояния считаем в долях ширины

    def __call__(self, dets: list[Detection], x: float, y: float, onset: bool, scale: float) -> float:
        if scale <= 0:
            return 0.0
        c = self.cfg
        target = most_urgent(dets)
        reward = 0.0
        if target is not None:
            d = np.hypot(x - target.x, (y - target.y) * self.aspect)
            reward += c.proximity * target.urgency * float(np.exp(-0.5 * (d / target.r) ** 2))
            if onset:
                # бонус только в последние ~90 мс (при AR8) и максимален в момент удара:
                # иначе эвристика поощряла бы ранние нажатия на 50/100
                timing = (target.urgency - 0.85) / 0.15
                if d <= target.r * 1.1 and timing > 0:
                    reward += c.click_good * timing
                else:
                    reward -= c.click_spam
        elif onset:
            reward -= c.click_spam
        return reward * scale


def teacher_action(dets: list[Detection], space, prev_press: int) -> int | None:
    """Эвристический «учитель» для guided exploration: курсор на самый срочный круг,
    нажатие — когда approach circle сомкнулся с кругом (с отпусканием между нажатиями).
    Кольцо с отношением радиусов 1.15 — это ~5% времени появления (~30 мс при AR8) до удара."""
    target = most_urgent(dets)
    if target is None:
        return None
    press = int(target.urgency >= 0.95 and not prev_press)
    return space.from_norm(target.x, target.y, press)


@dataclass
class GameCounters:
    h300: int = 0
    h100: int = 0
    h50: int = 0
    miss: int = 0
    slider_breaks: int = 0
    combo: int = 0


class HitTracker:
    """Превращает абсолютные счётчики игры в дельты (с обработкой рестарта карты)."""

    def __init__(self):
        self.prev: GameCounters | None = None

    def reset(self) -> None:
        self.prev = None

    def delta(self, cur: GameCounters) -> GameCounters:
        prev, self.prev = self.prev, cur
        if prev is None:
            return GameCounters()
        d = GameCounters(cur.h300 - prev.h300, cur.h100 - prev.h100, cur.h50 - prev.h50,
                         cur.miss - prev.miss, cur.slider_breaks - prev.slider_breaks,
                         cur.combo - prev.combo)
        if min(d.h300, d.h100, d.h50, d.miss, d.slider_breaks) < 0:  # рестарт: счётчики обнулились
            return GameCounters()
        return d


@dataclass
class PendingStep:
    frame: np.ndarray
    action: int
    reward: float
    first: bool
    onset: bool


class CreditAssigner:
    """Задерживает запись шагов в replay на `delay` шагов, чтобы события игры, пришедшие
    с опозданием (tosu опрашивает память раз в 10-100 мс), можно было приписать тому
    нажатию, которое их вызвало, а не случайному кадру позже."""

    def __init__(self, cfg: RewardConfig, sink):
        self.cfg = cfg
        self.sink = sink  # sink(frame, action, reward, first, terminal)
        self.q: deque[PendingStep] = deque()

    def push(self, step: PendingStep) -> None:
        self.q.append(step)
        while len(self.q) > self.cfg.credit_window_steps:
            s = self.q.popleft()
            self.sink(s.frame, s.action, s.reward, s.first, False)

    def apply_events(self, d: GameCounters) -> float:
        """Раздаёт награды за события игры по ожидающим шагам. Возвращает сумму наград."""
        c = self.cfg
        if not self.q:
            return 0.0
        hit_reward = d.h300 * c.hit300 + d.h100 * c.hit100 + d.h50 * c.hit50
        n_hits = d.h300 + d.h100 + d.h50
        if hit_reward:
            onsets = [s for s in self.q if s.onset]
            # по одному событию на последние нажатия (при нескольких попаданиях за опрос)
            targets = onsets[-n_hits:] if onsets else [self.q[-1]]
            for s in targets:
                s.reward += hit_reward / len(targets)
        other = d.miss * c.miss + d.slider_breaks * c.slider_break
        ticks = max(0, d.combo - n_hits) if d.combo > 0 else 0
        other += ticks * c.combo_tick
        if other:
            lag = min(c.miss_lag_steps, len(self.q) - 1)
            self.q[-1 - lag].reward += other
        return hit_reward + other

    def flush(self, terminal: bool = True) -> None:
        while self.q:
            s = self.q.popleft()
            self.sink(s.frame, s.action, s.reward, s.first, terminal and not self.q)
