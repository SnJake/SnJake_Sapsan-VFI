import argparse
from typing import Tuple

import cv2
import torch
from tqdm import tqdm

from src.models import VFIModel
from src.utils.misc import get_autocast_dtype, pad_to_multiple, unpad
from src.infer import load_config, load_weights, to_tensor, match_brightness


def _l1_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def diagnose_video(
    input_path: str,
    model: VFIModel,
    amp: str,
    pad_multiple: int,
    max_pairs: int,
    match_bright: bool,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[*] Input video: {fps} FPS, {total_frames} frames")

    ok, prev = cap.read()
    if not ok:
        return

    device = next(model.parameters()).device
    autocast_dtype = get_autocast_dtype(amp)
    autocast_enabled = autocast_dtype is not None and device.type == "cuda"

    count = 0
    acc_in = 0.0
    acc_p0 = 0.0
    acc_p1 = 0.0
    closer_to_prev = 0
    closer_to_curr = 0

    with torch.no_grad():
        pbar = tqdm(total=min(max_pairs, max(total_frames - 1, 0)), desc="Diagnosing")
        while count < max_pairs:
            ok, curr = cap.read()
            if not ok:
                break

            t0_raw = to_tensor(prev).unsqueeze(0).to(device)
            t1_raw = to_tensor(curr).unsqueeze(0).to(device)

            t0_padded, pad0 = pad_to_multiple(t0_raw, pad_multiple)
            t1_padded, pad1 = pad_to_multiple(t1_raw, pad_multiple)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                pred_padded, _aux = model(t0_padded, t1_padded, t=0.5)

            pred = unpad(pred_padded, pad0)[0]
            f0_orig = unpad(t0_padded, pad0)[0]
            f1_orig = unpad(t1_padded, pad1)[0]
            if match_bright:
                pred = match_brightness(pred, f0_orig, f1_orig)

            in_diff = _l1_mean(f0_orig, f1_orig).item()
            p0_diff = _l1_mean(pred, f0_orig).item()
            p1_diff = _l1_mean(pred, f1_orig).item()

            acc_in += in_diff
            acc_p0 += p0_diff
            acc_p1 += p1_diff
            if p0_diff <= p1_diff:
                closer_to_prev += 1
            else:
                closer_to_curr += 1

            prev = curr
            count += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    if count == 0:
        print("[*] No pairs processed.")
        return

    mean_in = acc_in / count
    mean_p0 = acc_p0 / count
    mean_p1 = acc_p1 / count
    print(f"[*] Pairs processed: {count}")
    print(f"[*] Mean L1 input diff (f0 vs f1): {mean_in:.6f}")
    print(f"[*] Mean L1 pred diff (pred vs f0): {mean_p0:.6f}")
    print(f"[*] Mean L1 pred diff (pred vs f1): {mean_p1:.6f}")
    print(f"[*] Pred closer to f0: {closer_to_prev}/{count}")
    print(f"[*] Pred closer to f1: {closer_to_curr}/{count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--amp", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--match-brightness", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = VFIModel(cfg["model"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        perf = cfg.get("performance", {})
        torch.backends.cuda.enable_flash_sdp(bool(perf.get("flash_sdp", True)))
        torch.backends.cuda.enable_mem_efficient_sdp(bool(perf.get("mem_efficient_sdp", True)))
        torch.backends.cuda.enable_math_sdp(bool(perf.get("math_sdp", True)))
    model.to(device).eval()
    load_weights(model, args.weights)

    pad_multiple = int(cfg.get("infer", {}).get("pad_multiple", 8))
    diagnose_video(args.input, model, args.amp, pad_multiple, args.max_pairs, args.match_brightness)


if __name__ == "__main__":
    main()
