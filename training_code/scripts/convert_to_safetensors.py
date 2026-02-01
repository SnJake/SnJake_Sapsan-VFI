import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

try:
    from safetensors.torch import save_file
except Exception as exc:
    raise SystemExit(
        "Missing dependency: safetensors. Install with `pip install safetensors`."
    ) from exc


def _infer_output_path(input_path: Path, use_ema: bool) -> Path:
    if input_path.suffix.lower() == ".safetensors":
        return input_path
    stem = input_path.stem
    if use_ema and not stem.endswith("_ema"):
        stem = f"{stem}_ema"
    return input_path.with_name(f"{stem}.safetensors")


def _extract_state_dict(payload: object, prefer_ema: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    metadata: Dict[str, str] = {}
    if isinstance(payload, dict):
        if "epoch" in payload:
            metadata["epoch"] = str(payload["epoch"])
        if "step" in payload:
            metadata["step"] = str(payload["step"])

        if prefer_ema and isinstance(payload.get("ema"), dict):
            state_dict = payload["ema"]
            metadata["source"] = "ema"
        elif isinstance(payload.get("model"), dict):
            state_dict = payload["model"]
            metadata["source"] = "model"
        elif isinstance(payload.get("state_dict"), dict):
            state_dict = payload["state_dict"]
            metadata["source"] = "state_dict"
        else:
            state_dict = payload
            metadata["source"] = "raw"
    else:
        raise ValueError("Unsupported checkpoint format: expected a dict-like payload.")

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    non_tensors = [k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
    if non_tensors:
        preview = ", ".join(non_tensors[:5])
        raise ValueError(f"state_dict has non-tensor entries: {preview}")

    return state_dict, metadata


def _convert_one(input_path: Path, output_path: Path, use_ema: bool, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    payload = torch.load(input_path, map_location="cpu")
    state_dict, metadata = _extract_state_dict(payload, use_ema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path), metadata=metadata)


def _iter_inputs(path: Path) -> Iterable[Path]:
    if path.is_dir():
        exts = {".pt", ".pth", ".ckpt", ".bin"}
        for item in sorted(path.rglob("*")):
            if item.is_file() and item.suffix.lower() in exts:
                yield item
    else:
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoints to safetensors.")
    parser.add_argument("input", type=str, help="Checkpoint file or directory.")
    parser.add_argument("--output", type=str, default=None, help="Output file path.")
    parser.add_argument("--use-ema", action="store_true", help="Prefer EMA weights if present.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    inputs = list(_iter_inputs(input_path))
    if not inputs:
        raise SystemExit(f"No checkpoint files found under: {input_path}")

    if args.output and len(inputs) > 1:
        raise SystemExit("Cannot use --output when converting a directory.")

    for item in inputs:
        output_path = Path(args.output) if args.output else _infer_output_path(item, args.use_ema)
        _convert_one(item, output_path, args.use_ema, args.overwrite)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
