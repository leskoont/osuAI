"""Окно визуализации в отдельном потоке (не тормозит агента).

Показывает то, что реально видит сеть (кадр 96x128 ×scale), тепловую карту Q-значений,
найденные детектором круги и позицию курсора. Надписи латиницей: шрифты Hershey в OpenCV
не поддерживают кириллицу (в исходной версии вместо «ПАУЗА» рисовались «?????»).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VisSnapshot:
    gray: np.ndarray | None = None
    qmap: np.ndarray | None = None  # (2, gh, gw)
    cursor: tuple[float, float] | None = None
    press: int = 0
    dets: list = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


class Visualizer:
    def __init__(self, hz: float, scale: int, show_qmap: bool):
        self.period = 1.0 / max(hz, 1.0)
        self.scale = scale
        self.show_qmap = show_qmap
        self._snap = VisSnapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.last_push = 0.0
        self.thread = threading.Thread(target=self._run, name="vis", daemon=True)
        self.thread.start()

    def due(self) -> bool:
        return time.perf_counter() - self.last_push >= self.period

    def push(self, snap: VisSnapshot) -> None:
        self.last_push = time.perf_counter()
        with self._lock:
            self._snap = snap

    def _render(self, s: VisSnapshot) -> np.ndarray:
        if s.gray is None:
            img = np.zeros((96 * self.scale, 128 * self.scale, 3), np.uint8)
        else:
            h, w = s.gray.shape
            img = cv2.cvtColor(cv2.resize(s.gray, (w * self.scale, h * self.scale),
                                          interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        H, W = img.shape[:2]
        if self.show_qmap and s.qmap is not None:
            best = s.qmap.max(axis=0)  # лучшее Q в ячейке (с нажатием или без)
            m = (best - best.min()) / max(float(np.ptp(best)), 1e-6)
            heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
            img = cv2.addWeighted(img, 0.65, heat, 0.35, 0)
        for d in s.dets:
            c = (int(d.x * W), int(d.y * H))
            cv2.circle(img, c, max(2, int(d.r * W)), (0, int(255 * d.urgency), 255), 1)
        if s.cursor is not None:
            c = (int(s.cursor[0] * W), int(s.cursor[1] * H))
            cv2.circle(img, c, 6, (0, 0, 255) if s.press else (0, 255, 0), -1)
        for i, line in enumerate(s.lines):
            cv2.putText(img, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return img

    def _run(self) -> None:
        cv2.namedWindow("OsuAI Vision", cv2.WINDOW_AUTOSIZE)
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                with self._lock:
                    snap = self._snap
                cv2.imshow("OsuAI Vision", self._render(snap))
                cv2.waitKey(1)
                time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))
        finally:
            cv2.destroyAllWindows()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)
