"""Полностью свёрточная dueling Q-сеть, выдающая Q-карту (2, H/4, W/4).

Почему так, а не Conv→Flatten→Dense, как было:
* исходная сеть имела ~30M параметров в одном Dense-слое (Flatten 30x30x128 → 256),
  медленно считалась и плохо обобщала координаты;
* здесь Q-значение для каждой ячейки вычисляется локальными свёртками — сеть учит
  «есть ли тут объект, по которому пора кликнуть», и знание переносится между всеми
  позициями экрана. ~0.33M параметров, инференс < 1 мс на RTX 2060.
* дилатированные residual-блоки дают рецептивное поле ~160 px входа — достаточно,
  чтобы видеть approach circle целиком и оценить, сколько осталось до удара.
Нормализации нет намеренно: BatchNorm плохо сочетается с нестационарными данными RL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, ObsConfig


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class QMapNet(nn.Module):
    def __init__(self, obs: ObsConfig, model: ModelConfig):
        super().__init__()
        self.stack = obs.stack
        self.h, self.w = obs.height, obs.width
        ch = model.channels
        self.stem = nn.Sequential(
            nn.Conv2d(obs.stack + 1, model.stem_channels, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(model.stem_channels, ch, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(ch, 2 ** (i % 4)) for i in range(model.blocks)])
        self.adv = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(), nn.Conv2d(ch, 2, 1))
        self.val = nn.Sequential(nn.Linear(2 * ch, 128), nn.ReLU(), nn.Linear(128, 1))
        # Маленькая инициализация выходов → стартовые Q≈0, без «уверенного» мусора.
        nn.init.normal_(self.adv[-1].weight, std=0.01)
        nn.init.zeros_(self.adv[-1].bias)
        nn.init.normal_(self.val[-1].weight, std=0.01)
        nn.init.zeros_(self.val[-1].bias)
        # Настраиваются владельцем (learner/актор).
        self.input_dtype = torch.float32
        self.channels_last = False

    def forward(self, frames: torch.Tensor, prev_press: torch.Tensor) -> torch.Tensor:
        """frames: (B, stack, H, W) uint8; prev_press: (B,) — была ли зажата клавиша.
        Возвращает Q: (B, 2 * H/4 * W/4)."""
        b = frames.shape[0]
        x = frames.to(self.input_dtype) * (1.0 / 255.0)
        press = prev_press.to(self.input_dtype).view(b, 1, 1, 1).expand(b, 1, self.h, self.w)
        x = torch.cat([x, press], dim=1)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.blocks(self.stem(x))
        a = self.adv(h)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        v = self.val(pooled)
        q = v.view(b, 1, 1, 1) + a - a.mean(dim=(1, 2, 3), keepdim=True)
        return q.flatten(1)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(obs: ObsConfig, model: ModelConfig, device: torch.device, *, half: bool = False,
                channels_last: bool = False) -> QMapNet:
    net = QMapNet(obs, model).to(device)
    if half:
        net = net.half()
        net.input_dtype = torch.float16
    if channels_last and device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)
        net.channels_last = True
    return net
