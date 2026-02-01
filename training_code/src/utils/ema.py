from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class EMA:
    decay: float
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def register(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().to(self.device or param.device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow, "EMA not registered"
            new = param.detach()
            old = self.shadow[name]
            if self.device is not None:
                new = new.to(self.device)
            self.shadow[name] = old * self.decay + (1.0 - self.decay) * new

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name].to(param.device))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if self.device is None:
            self.shadow = {k: v.clone() for k, v in state_dict.items()}
        else:
            self.shadow = {k: v.clone().to(self.device) for k, v in state_dict.items()}
