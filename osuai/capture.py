"""Захват экрана в отдельном потоке + предобработка там же (конвейер).

Исходная версия на каждом шаге: снимала весь монитор 2560x1440 дважды (capture() и
capture_preprocessed()), конвертировала через PIL, делала float64 /255 и всё в главном потоке.
Здесь:
* dxcam (Desktop Duplication API, до 240+ FPS, BGRA без конверсии) с запасным mss;
* снимается только область playfield (≈1728x1296 при 1440p), а не весь экран;
* уменьшение INTER_AREA сразу в 4-канальном виде, затем «яркость» = max(B,G,R) → uint8 96x128;
  цветные круги остаются яркими (в оттенках серого синий круг почти чёрный);
* актор просто забирает последний готовый кадр — захват и предобработка не на критическом пути.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CaptureConfig, ObsConfig


@dataclass
class Frame:
    gray: np.ndarray    # (H, W) uint8 — вход сети
    color: np.ndarray   # (H*s, W*s, 3) uint8 BGR — для детектора и визуализации
    t: float
    fid: int


class Preprocessor:
    def __init__(self, obs: ObsConfig):
        self.size = (obs.width, obs.height)
        self.det_size = (obs.width * obs.detector_scale, obs.height * obs.detector_scale)

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dw, dh = self.det_size
        if img.shape[1] > 2 * dw and img.shape[0] > 2 * dh:
            # INTER_AREA с дробным коэффициентом (1728→256 = 6.75x) медленный: ~7 мс.
            # Билинейно до 2x (шаг < 2 px, тонкие кольца не теряются) + AREA ровно 2x: ~0.4 мс.
            img = cv2.resize(img, (2 * dw, 2 * dh), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(img, self.det_size, interpolation=cv2.INTER_AREA)
        color = np.ascontiguousarray(small[..., :3]) if small.shape[2] == 4 else small
        v = cv2.max(cv2.max(color[..., 0], color[..., 1]), color[..., 2])
        gray = cv2.resize(v, self.size, interpolation=cv2.INTER_AREA)
        return gray, color


class _MssBackend:
    def __init__(self, region: tuple[int, int, int, int], fps: int):
        import mss  # создаётся в потоке захвата: mss на Windows держит thread-local DC

        self.sct = mss.mss()
        l, t, r, b = region
        self.mon = {"left": l, "top": t, "width": r - l, "height": b - t}
        self.period = 1.0 / max(fps, 1)
        self.next_t = time.perf_counter()

    def grab(self) -> np.ndarray | None:
        now = time.perf_counter()
        if now < self.next_t:
            time.sleep(self.next_t - now)
        self.next_t = max(self.next_t + self.period, time.perf_counter() - self.period)
        shot = self.sct.grab(self.mon)
        return np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)

    def close(self) -> None:
        self.sct.close()


class _DxcamBackend:
    def __init__(self, region: tuple[int, int, int, int], fps: int, output: int,
                 monitor_origin: tuple[int, int]):
        import dxcam

        if output < 0:
            output = self._find_output(dxcam, region)
        self.cam = dxcam.create(output_idx=None if output < 0 else output, output_color="BGRA")
        ox, oy = monitor_origin
        try:  # точное смещение выхода на виртуальном рабочем столе (внутреннее поле dxcam)
            dc = self.cam._output.desc.DesktopCoordinates
            ox, oy = dc.left, dc.top
        except Exception:
            pass
        l, t, r, b = region
        self.cam.start(region=(l - ox, t - oy, r - ox, b - oy), target_fps=fps, video_mode=True)

    @staticmethod
    def _find_output(dxcam, region: tuple[int, int, int, int]) -> int:
        """Выход (монитор), на котором находится центр области; -1 — основной."""
        cx, cy = (region[0] + region[2]) // 2, (region[1] + region[3]) // 2
        for idx in range(8):
            try:
                cam = dxcam.create(output_idx=idx, output_color="BGRA")
            except Exception:
                break
            try:
                dc = cam._output.desc.DesktopCoordinates
                if dc.left <= cx < dc.right and dc.top <= cy < dc.bottom:
                    return idx
            except Exception:
                return -1
            finally:
                try:
                    cam.release()
                except Exception:
                    pass
                del cam
        return -1

    def grab(self) -> np.ndarray | None:
        return self.cam.get_latest_frame()

    def close(self) -> None:
        self.cam.stop()
        try:
            self.cam.release()
        except Exception:
            pass
        del self.cam


class ScreenSource:
    """Поток захвата. `latest(after)` возвращает кадр новее `after` (или ждёт его)."""

    def __init__(self, cfg: CaptureConfig, obs: ObsConfig, region: tuple[int, int, int, int],
                 monitor_origin: tuple[int, int] = (0, 0)):
        self.cfg = cfg
        self.pre = Preprocessor(obs)
        self.region = region
        self.monitor_origin = monitor_origin
        self.backend_name = ""
        self._frame: Frame | None = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._restart = threading.Event()
        self.capture_fps = 0.0
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="capture", daemon=True)

    def start(self) -> "ScreenSource":
        self.thread.start()
        return self

    def set_region(self, region: tuple[int, int, int, int], monitor_origin: tuple[int, int]) -> None:
        if region != self.region:
            self.region, self.monitor_origin = region, monitor_origin
            self._restart.set()

    def _make_backend(self):
        order = {"auto": ["dxcam", "mss"], "dxcam": ["dxcam"], "mss": ["mss"]}[self.cfg.backend]
        last: Exception | None = None
        for name in order:
            try:
                if name == "dxcam":
                    b = _DxcamBackend(self.region, self.cfg.target_fps, self.cfg.dxcam_output,
                                      self.monitor_origin)
                else:
                    b = _MssBackend(self.region, self.cfg.target_fps)
                if name != self.backend_name:
                    print(f"[capture] backend: {name}, область {self.region}")
                self.backend_name = name
                return b
            except Exception as e:
                last = e
                if self.cfg.backend == "auto":
                    print(f"[capture] {name} недоступен: {e}")
        raise RuntimeError(f"Нет доступного backend захвата: {last}")

    def _run(self) -> None:
        backend = None
        fid, t_rate, n_rate = 0, time.perf_counter(), 0
        try:
            backend = self._make_backend()
            while not self._stop.is_set():
                if self._restart.is_set():
                    self._restart.clear()
                    backend.close()
                    backend = self._make_backend()
                raw = backend.grab()
                if raw is None:
                    continue
                gray, color = self.pre(raw)
                fid += 1
                with self._cond:
                    self._frame = Frame(gray, color, time.perf_counter(), fid)
                    self._cond.notify_all()
                now = time.perf_counter()
                if now - t_rate > 1.0:
                    self.capture_fps = (fid - n_rate) / (now - t_rate)
                    t_rate, n_rate = now, fid
        except Exception as e:  # pragma: no cover - зависит от ОС
            self.error = e
            print(f"[capture] ошибка: {e}")
        finally:
            if backend is not None:
                backend.close()
            with self._cond:
                self._cond.notify_all()

    def latest(self, after: int = 0, timeout: float = 0.1) -> Frame | None:
        with self._cond:
            if self._frame is None or self._frame.fid <= after:
                self._cond.wait(timeout)
            f = self._frame
        return f if f is not None and f.fid > after else None

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)
