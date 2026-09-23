"""Среда «реальная osu!»: окно игры, захват, ввод, tosu. Интерфейс совпадает с SimEnv."""

from __future__ import annotations

import time

from . import winapi as w
from .actions import playfield_region
from .capture import Frame, ScreenSource
from .config import Config
from .controller import InputController
from .reward import GameCounters
from .tosu import TosuClient


class OsuEnv:
    realtime = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hwnd = w.find_window(cfg.capture.window_title, cfg.capture.process_name)
        if self.hwnd is None and cfg.capture.region in ("playfield", "window"):
            raise RuntimeError(
                f"Окно '{cfg.capture.window_title}' ({cfg.capture.process_name}) не найдено. "
                "Запустите osu! (лучше в оконном/borderless режиме).")
        self.region, self.origin = self._compute_region()
        l, t, r, b = self.region
        self.aspect = (b - t) / max(r - l, 1)
        print(f"[env] область захвата: {self.region} ({r - l}x{b - t})")
        self.controller = InputController(cfg.input)
        self.controller.set_region(self.region)
        self.source = ScreenSource(cfg.capture, cfg.obs, self.region, self.origin).start()
        self.tosu = TosuClient(cfg.tosu.url).start() if cfg.tosu.enabled else None
        # в записи учитываем и кнопки мыши — многие играют кликами
        self.human_keys = sorted(set(self.controller.keys) | {0x01, 0x02})
        self._human_held = False
        self._last_region_check = time.perf_counter()
        self._game_time = None
        self._game_time_changed = 0.0

    @property
    def has_events(self) -> bool:
        return self.tosu is not None and self.tosu.connected

    def _compute_region(self) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
        c = self.cfg.capture
        if c.region == "manual":
            region = tuple(int(v) for v in c.manual_region)
            origin = (0, 0)
        elif c.region == "monitor":
            import mss
            with mss.mss() as sct:
                m = sct.monitors[c.monitor]
            region = (m["left"], m["top"], m["left"] + m["width"], m["top"] + m["height"])
            origin = (m["left"], m["top"])
        else:
            client = w.client_rect(self.hwnd)
            region = client if c.region == "window" else playfield_region(
                client, c.margin_x, c.margin_y, c.playfield_y_offset)
            ml, mt, _, _ = w.monitor_rect_for(self.hwnd)
            origin = (ml, mt)
        l, t, r, b = region
        if r - l < 16 or b - t < 16:
            raise RuntimeError(f"Слишком маленькая область захвата: {region} (окно свёрнуто?)")
        return region, origin

    def _refresh_region(self) -> None:
        now = time.perf_counter()
        if now - self._last_region_check < 2.0 or self.cfg.capture.region not in ("playfield", "window"):
            return
        self._last_region_check = now
        try:
            region, origin = self._compute_region()
        except RuntimeError:
            return
        if region != self.region:
            print(f"[env] окно изменилось, новая область: {region}")
            self.region, self.origin = region, origin
            self.controller.set_region(region)
            self.source.set_region(region, origin)

    # ------------------------------------------------------------------ env API
    def status(self) -> tuple[bool, str]:
        if self.source.error is not None:
            raise RuntimeError(f"Захват экрана упал: {self.source.error}")
        self._refresh_region()
        if self.hwnd is not None and w.foreground_window() != self.hwnd:
            return False, "osu! не в фокусе"
        if self.tosu is not None and self.tosu.connected and self.cfg.tosu.gate_on_playing:
            st = self.tosu.latest()
            if not st.playing:
                return False, "не на карте (tosu)"
            now = time.perf_counter()
            if st.time_ms != self._game_time:
                self._game_time, self._game_time_changed = st.time_ms, now
            elif now - self._game_time_changed > 0.35:
                return False, "пауза в игре"
        return True, ""

    def observe(self, after: int = 0) -> Frame | None:
        return self.source.latest(after, timeout=0.1)

    def act(self, nx: float, ny: float, press: int) -> bool:
        return self.controller.apply(nx, ny, press)

    def game_counters(self) -> GameCounters | None:
        if not self.has_events:
            return None
        return self.tosu.latest().counters

    def human_step(self) -> tuple[float, float, int, bool]:
        """Режим записи: читаем курсор и клавиши человека."""
        x, y = w.cursor_pos()
        l, t, r, b = self.region
        nx = min(max((x - l) / max(r - l, 1), 0.0), 1.0)
        ny = min(max((y - t) / max(b - t, 1), 0.0), 1.0)
        press = int(any(w.key_down(vk) for vk in self.human_keys))
        onset = bool(press) and not self._human_held
        self._human_held = bool(press)
        return nx, ny, press, onset

    def focus(self) -> bool:
        return w.focus_window(self.hwnd) if self.hwnd else False

    def release(self) -> None:
        self.controller.release()

    def close(self) -> None:
        self.controller.release()
        self.source.stop()
        if self.tosu is not None:
            self.tosu.stop()
