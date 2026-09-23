"""Клиент tosu / gosumemory (WebSocket, формат gosumemory v1: ws://127.0.0.1:24050/ws).

tosu читает память osu! и отдаёт точные попадания 300/100/50/miss, комбо, HP, время карты
и состояние (2 = идёт игра). Это на порядки точнее и дешевле OCR по скриншоту.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from .reward import GameCounters

PLAYING_STATE = 2


@dataclass
class GameState:
    state: int = -1
    time_ms: float = 0.0
    counters: GameCounters = field(default_factory=GameCounters)
    hp: float = 0.0
    score: int = 0
    accuracy: float = 0.0
    received_at: float = 0.0

    @property
    def playing(self) -> bool:
        return self.state == PLAYING_STATE


def _num(d: dict, *keys, default=0):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if isinstance(d, (int, float)) else default


def parse_message(msg: dict, now: float) -> GameState:
    gp = msg.get("gameplay", {}) or {}
    hits = gp.get("hits", {}) or {}
    return GameState(
        state=int(_num(msg, "menu", "state", default=-1)),
        time_ms=float(_num(msg, "menu", "bm", "time", "current")),
        counters=GameCounters(
            h300=int(_num(hits, "300")), h100=int(_num(hits, "100")), h50=int(_num(hits, "50")),
            miss=int(_num(hits, "0")), slider_breaks=int(_num(hits, "sliderBreaks")),
            combo=int(_num(gp, "combo", "current")),
        ),
        hp=float(_num(gp, "hp", "normal")),
        score=int(_num(gp, "score")),
        accuracy=float(_num(gp, "accuracy")),
        received_at=now,
    )


class TosuClient:
    def __init__(self, url: str):
        self.url = url
        self._state = GameState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.connected = False
        self.thread = threading.Thread(target=self._run, name="tosu", daemon=True)

    def start(self) -> "TosuClient":
        self.thread.start()
        return self

    def latest(self) -> GameState:
        with self._lock:
            return self._state

    def _run(self) -> None:
        try:
            import websocket  # websocket-client
        except ImportError:
            print("[tosu] пакет websocket-client не установлен — награды только эвристические")
            return
        warned = False
        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(self.url, timeout=2)
                self.connected = True
                print(f"[tosu] подключено к {self.url}")
                warned = False
                while not self._stop.is_set():
                    raw = ws.recv()
                    if not raw:
                        continue
                    state = parse_message(json.loads(raw), time.perf_counter())
                    with self._lock:
                        self._state = state
                ws.close()
            except Exception as e:
                if self.connected or not warned:
                    print(f"[tosu] нет соединения ({type(e).__name__}); запустите tosu для точных наград")
                    warned = True
                self.connected = False
                self._stop.wait(2.0)

    def stop(self) -> None:
        self._stop.set()
