import os
from typing import Any, Dict, Optional, Tuple

import torch


def _rng_state() -> Dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path_full: str,
    path_weights: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[Any],
    epoch: int,
    step: int,
) -> None:
    os.makedirs(os.path.dirname(path_full), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "step": step,
        "rng": _rng_state(),
    }
    torch.save(payload, path_full)

    weights = ema.state_dict() if ema is not None else model.state_dict()
    torch.save({"model": weights, "epoch": epoch, "step": step}, path_weights)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[Any],
    map_location: str = "cpu",
) -> Tuple[int, int, Dict[str, bool]]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    loaded: Dict[str, bool] = {
        "optimizer": False,
        "scheduler": False,
        "scaler": False,
        "ema": False,
        "rng": False,
    }
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
        loaded["optimizer"] = True
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
        loaded["scheduler"] = True
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
        loaded["scaler"] = True
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
        loaded["ema"] = True
    if payload.get("rng") is not None:
        _set_rng_state(payload["rng"])
        loaded["rng"] = True
    return int(payload.get("epoch", 0)), int(payload.get("step", 0)), loaded
