import os
import random
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast_dtype(amp: str) -> Optional[torch.dtype]:
    amp = (amp or "no").lower()
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return None


def to_device(batch, device: torch.device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    if multiple <= 1:
        return x, (0, 0, 0, 0)
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return x, (pad_left, pad_right, pad_top, pad_bottom)


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    pad_left, pad_right, pad_top, pad_bottom = pad
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return x
    return x[..., pad_top : x.shape[-2] - pad_bottom, pad_left : x.shape[-1] - pad_right]


def maybe_compile(model: torch.nn.Module, cfg: Mapping[str, Any], name: str = "model") -> torch.nn.Module:
    perf = cfg.get("performance", {}) if cfg else {}
    if not bool(perf.get("torch_compile", False)):
        return model
    if not hasattr(torch, "compile"):
        print(f"[!] torch.compile is not available; using eager {name}.")
        return model
    kwargs = {}
    mode = perf.get("torch_compile_mode")
    backend = perf.get("torch_compile_backend")
    dynamic = perf.get("torch_compile_dynamic")
    fullgraph = perf.get("torch_compile_fullgraph")
    if mode is not None:
        kwargs["mode"] = str(mode)
    if backend is not None:
        kwargs["backend"] = str(backend)
    if dynamic is not None:
        kwargs["dynamic"] = bool(dynamic)
    if fullgraph is not None:
        kwargs["fullgraph"] = bool(fullgraph)
    try:
        compiled = torch.compile(model, **kwargs)
    except Exception as exc:
        print(f"[!] torch.compile failed for {name}: {exc}. Using eager.")
        return model
    details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    suffix = f" ({details})" if details else ""
    print(f"[*] torch.compile enabled for {name}{suffix}.")
    return compiled
