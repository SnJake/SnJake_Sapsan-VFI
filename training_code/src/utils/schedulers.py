import math
from typing import Any, Dict

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], total_steps: int):
    sched_type = str(cfg.get("type", "constant")).lower()
    warmup_steps = int(cfg.get("warmup_steps", 0))
    min_lr = float(cfg.get("min_lr", 0.0))

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(0.0, float(step) / float(max(1, warmup_steps)))

        if sched_type == "constant":
            return 1.0

        if sched_type == "cosine":
            if total_steps <= warmup_steps:
                return 1.0
            progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + (1.0 - min_lr) * cosine

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
