import os
import shutil
import urllib.request
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import yaml

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from comfy import utils as comfy_utils
except Exception:
    comfy_utils = None

from .training_code.src.models import VFIModel
from .training_code.src.utils.misc import pad_to_multiple, unpad

_CATEGORY = "😎 SnJake/VFI"
_MODEL_DIR_NAME = "sapsan_vfi"
_VALID_EXTS = (".pt", ".pth", ".ckpt", ".safetensors")
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "training_code", "config.yaml")
_WEIGHTS_BASE_URL = "https://huggingface.co/SnJake/Sapsan-VFI/resolve/main"
_REMOTE_WEIGHTS = {
    "Sapsan-VFI.safetensors": f"{_WEIGHTS_BASE_URL}/Sapsan-VFI.safetensors",
    "Sapsan-VFI.pt": f"{_WEIGHTS_BASE_URL}/Sapsan-VFI.pt",
}


class SapsanVFIModel:
    def __init__(self, model, cfg: dict, config_path: str, weights_path: str):
        self.model = model
        self.cfg = cfg or {}
        self.config_path = config_path
        self.weights_path = weights_path
        self.pad_multiple = int(self.cfg.get("infer", {}).get("pad_multiple", 8))
        self.amp_mode = str(self.cfg.get("train", {}).get("amp", "auto")).lower()
        self._compiled: Dict[str, torch.nn.Module] = {}
        self._compiled_device: str | None = None

    def get_model(self, device: torch.device, torch_compile: bool) -> torch.nn.Module:
        if device.type != "cuda":
            self._compiled.clear()
            self._compiled_device = None

        device_key = str(device)
        model = self.model.to(device)
        model.eval()
        if torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
            if self._compiled_device not in (None, device_key):
                self._compiled.clear()
            cached = self._compiled.get(device_key)
            if cached is None:
                try:
                    cached = torch.compile(model)
                except Exception as exc:
                    print(f"[Sapsan-VFI] torch.compile failed: {exc}. Using eager mode.")
                    return model
                self._compiled[device_key] = cached
                self._compiled_device = device_key
            return cached
        return model


_MODEL_CACHE: Dict[Tuple[str, str], SapsanVFIModel] = {}


def _resolve_models_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        base = folder_paths.models_dir
    else:
        base = os.path.join(os.getcwd(), "models")
    path = os.path.join(base, _MODEL_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    try:
        if folder_paths is not None and hasattr(folder_paths, "add_model_search_path"):
            folder_paths.add_model_search_path(_MODEL_DIR_NAME, path)
    except Exception:
        pass
    return path


def _list_models() -> list[str]:
    root = _resolve_models_dir()
    try:
        names = [
            name
            for name in os.listdir(root)
            if name.lower().endswith(_VALID_EXTS) and os.path.isfile(os.path.join(root, name))
        ]
    except Exception:
        names = []
    names.sort()
    return names


def _weight_choices() -> list[str]:
    local_names = _list_models()
    remote_names = list(_REMOTE_WEIGHTS.keys())
    choices = local_names + [name for name in remote_names if name not in local_names]
    return choices if choices else ["<none found>"]


def _download_weights(url: str, dst_path: str) -> None:
    tmp_path = f"{dst_path}.tmp"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-SnJake-Sapsan-VFI"},
    )
    try:
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _resolve_weights_path(weights_name: str) -> str:
    root = _resolve_models_dir()

    if not weights_name or weights_name in ("<none found>",):
        raise FileNotFoundError("Sapsan-VFI weights are not configured.")

    if os.path.basename(weights_name) != weights_name:
        raise ValueError("Weights name must be a filename without directories.")

    candidate = os.path.join(root, weights_name)
    if os.path.isfile(candidate):
        return candidate

    url = _REMOTE_WEIGHTS.get(weights_name)
    if url:
        if "<USER>" in url or "<REPO>" in url:
            raise FileNotFoundError(
                "Weights URL placeholders are not configured. Update _WEIGHTS_BASE_URL/_REMOTE_WEIGHTS "
                "or download weights manually to the models/sapsan_vfi folder."
            )
        print(f"Downloading Sapsan-VFI weights: {weights_name}")
        _download_weights(url, candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Sapsan-VFI weights not found: '{weights_name}'.")


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: '{path}'.")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_checkpoint_any(path: str):
    if comfy_utils is not None:
        return comfy_utils.load_torch_file(path, safe_load=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _configure_cuda_backends(cfg: dict) -> None:
    if not torch.cuda.is_available():
        return
    perf = cfg.get("performance", {}) if isinstance(cfg, dict) else {}
    torch.backends.cuda.enable_flash_sdp(bool(perf.get("flash_sdp", True)))
    torch.backends.cuda.enable_mem_efficient_sdp(bool(perf.get("mem_efficient_sdp", True)))
    torch.backends.cuda.enable_math_sdp(bool(perf.get("math_sdp", True)))


def _resolve_amp_dtype(mode: str, fallback_mode: str):
    mode_in = (mode or "auto").lower()
    fallback = (fallback_mode or "auto").lower()
    if fallback in ("no", "none"):
        fallback = "none"
    if mode_in == "auto":
        mode_in = fallback if fallback in ("bf16", "fp16", "auto", "none") else "auto"

    if mode_in == "none" or not torch.cuda.is_available():
        return None
    if mode_in == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode_in == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _match_brightness(pred: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    mean_inputs = 0.5 * (f0.mean(dim=(1, 2), keepdim=True) + f1.mean(dim=(1, 2), keepdim=True))
    mean_pred = pred.mean(dim=(1, 2), keepdim=True)
    scale = mean_inputs / (mean_pred + 1e-6)
    return pred * scale


class SnJakeSapsanVFICheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = _weight_choices()
        default_name = names[0]
        return {
            "required": {
                "weights_name": (
                    names,
                    {"default": default_name, "tooltip": "Select weights (auto-download if missing)."},
                ),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAPSAN_VFI_MODEL",)
    RETURN_NAMES = ("vfi_model",)
    FUNCTION = "load"
    CATEGORY = _CATEGORY

    def load(self, weights_name, force_reload):
        config_path = _CONFIG_PATH
        cfg = _load_config(config_path)

        weights_path = _resolve_weights_path(weights_name)
        cache_key = (weights_path, config_path)

        if not force_reload and cache_key in _MODEL_CACHE:
            cached = _MODEL_CACHE[cache_key]
            return (cached,)

        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        model = VFIModel(model_cfg)
        ckpt = _load_checkpoint_any(weights_path)
        state_dict = _extract_state_dict(ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to("cpu")

        wrapper = SapsanVFIModel(
            model=model,
            cfg=cfg,
            config_path=config_path,
            weights_path=weights_path,
        )
        _MODEL_CACHE[cache_key] = wrapper
        return (wrapper,)


class SnJakeSapsanVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vfi_model": ("SAPSAN_VFI_MODEL",),
                "images": ("IMAGE",),
                "t": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "pad_multiple": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "0 uses the value from config.yaml.",
                    },
                ),
                "match_brightness": ("BOOLEAN", {"default": True}),
                "amp": (["auto", "bf16", "fp16", "none"], {"default": "auto"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "console_progress": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "interpolate"
    CATEGORY = _CATEGORY

    def interpolate(
        self,
        vfi_model,
        images,
        t,
        pad_multiple,
        match_brightness,
        amp,
        device,
        torch_compile,
        console_progress,
        fps=None,
    ):
        if vfi_model is None:
            raise ValueError("Sapsan-VFI model is missing.")

        if images.dim() != 4:
            raise ValueError("Expected image tensor with shape [B, H, W, C].")

        if images.shape[-1] != 3:
            raise ValueError("Sapsan-VFI expects 3-channel RGB images.")

        if device == "auto":
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(device)

        _configure_cuda_backends(vfi_model.cfg)
        model = vfi_model.get_model(device_t, torch_compile)

        amp_dtype = _resolve_amp_dtype(amp, vfi_model.amp_mode)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None and device_t.type == "cuda"
            else nullcontext()
        )

        pad_multiple = int(pad_multiple) if pad_multiple and pad_multiple > 0 else int(vfi_model.pad_multiple)
        pad_multiple = max(1, pad_multiple)
        t = float(t)

        b, _, _, _ = images.shape
        if b <= 1:
            fps_out = float(fps) if fps is not None else 0.0
            return (images, fps_out)

        device_out = images.device
        dtype_out = images.dtype
        total_pairs = b - 1
        pbar = None
        if comfy_utils is not None and hasattr(comfy_utils, "ProgressBar"):
            pbar = comfy_utils.ProgressBar(total_pairs)
        log_every = max(1, total_pairs // 20) if total_pairs > 0 else 1
        next_log = log_every
        if console_progress:
            print(f"[Sapsan-VFI] Interpolating {total_pairs} frame pairs...")

        out_frames = []
        with torch.inference_mode():
            for idx in range(total_pairs):
                f0 = images[idx]
                f1 = images[idx + 1]
                out_frames.append(f0)

                t0 = f0.permute(2, 0, 1).contiguous().unsqueeze(0).to(device_t)
                t1 = f1.permute(2, 0, 1).contiguous().unsqueeze(0).to(device_t)

                t0_padded, pad0 = pad_to_multiple(t0, pad_multiple)
                t1_padded, pad1 = pad_to_multiple(t1, pad_multiple)

                with amp_ctx:
                    pred_padded, _aux = model(t0_padded, t1_padded, t=t)

                pred = unpad(pred_padded, pad0)[0]
                if match_brightness:
                    f0_orig = unpad(t0_padded, pad0)[0]
                    f1_orig = unpad(t1_padded, pad1)[0]
                    pred = _match_brightness(pred, f0_orig, f1_orig)

                pred = pred.float().clamp(0.0, 1.0)
                pred_out = pred.permute(1, 2, 0).contiguous().to(device_out, dtype=dtype_out)
                out_frames.append(pred_out)

                if pbar is not None:
                    pbar.update(1)
                if console_progress and (idx + 1 >= next_log or idx + 1 == total_pairs):
                    percent = (idx + 1) * 100.0 / float(total_pairs)
                    print(f"[Sapsan-VFI] Progress: {idx + 1}/{total_pairs} ({percent:.1f}%)")
                    next_log += log_every

            out_frames.append(images[-1])

        result = torch.stack(out_frames, dim=0)
        fps_out = float(fps) * 2.0 if fps is not None else 0.0
        return (result, fps_out)
