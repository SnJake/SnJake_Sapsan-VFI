import argparse
import os
from typing import Tuple

import cv2
import torch
import yaml
from tqdm import tqdm

from src.models import VFIModel
from src.utils.misc import get_autocast_dtype, pad_to_multiple, unpad


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_weights(model: torch.nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)


def to_tensor(frame) -> torch.Tensor:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return t


def to_frame(t: torch.Tensor) -> Tuple:
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def match_brightness(pred: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    """Выравнивает яркость предсказанного кадра по среднему значению между f0 и f1."""
    # Среднее значение яркости по каналам между входными кадрами
    mean_inputs = 0.5 * (f0.mean(dim=(1, 2), keepdim=True) + f1.mean(dim=(1, 2), keepdim=True))
    # Среднее значение яркости предсказанного кадра
    mean_pred = pred.mean(dim=(1, 2), keepdim=True)
    # Множитель коррекции
    scale = mean_inputs / (mean_pred + 1e-6)
    return pred * scale


def interpolate_video(input_path: str, output_path: str, model: VFIModel, amp: str, pad_multiple: int) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[*] Input video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"[*] Output video will be: {fps * 2.0} FPS")

    # Используем mp4v. Если видео не проигрывается плавно, попробуйте 'avc1'
    def _open_writer(path: str, fps_out: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, str]:
        for tag in ("mp4v", "avc1", "H264", "X264"):
            fourcc = cv2.VideoWriter_fourcc(*tag)
            writer = cv2.VideoWriter(path, fourcc, fps_out, size)
            if writer.isOpened():
                return writer, tag
            writer.release()
        raise RuntimeError("Failed to open VideoWriter with any supported codec")

    out, codec_tag = _open_writer(output_path, fps * 2.0, (width, height))
    print(f"[*] Using codec: {codec_tag}")

    ok, prev = cap.read()
    if not ok:
        return

    device = next(model.parameters()).device
    autocast_dtype = get_autocast_dtype(amp)
    autocast_enabled = autocast_dtype is not None and device.type == "cuda"

    frames_written = 0
    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="Interpolating")
        while True:
            ok, curr = cap.read()
            if not ok:
                break
            
            t0_raw = to_tensor(prev).unsqueeze(0).to(device)
            t1_raw = to_tensor(curr).unsqueeze(0).to(device)

            # Явно сохраняем pad0 и pad1
            t0_padded, pad0 = pad_to_multiple(t0_raw, pad_multiple)
            t1_padded, pad1 = pad_to_multiple(t1_raw, pad_multiple)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                # Избегаем конфликта имен с переменной "_", сохраняя в aux
                pred_padded, _aux = model(t0_padded, t1_padded, t=0.5)
            
            # Убираем паддинг
            pred = unpad(pred_padded, pad0)[0]
            f0_orig = unpad(t0_padded, pad0)[0]
            f1_orig = unpad(t1_padded, pad1)[0]

            # ПРИМЕНЯЕМ КОРРЕКЦИЮ (чтобы не было мерцания)
            pred = match_brightness(pred, f0_orig, f1_orig)

            # Записываем оригинальный кадр и интерполированный
            out.write(prev)
            out.write(to_frame(pred))
            frames_written += 2
            
            prev = curr
            pbar.update(1)
        
        # Дописываем самый последний кадр
        out.write(prev)
        frames_written += 1
        pbar.close()

    cap.release()
    out.release()
    try:
        cap_out = cv2.VideoCapture(output_path)
        out_fps = cap_out.get(cv2.CAP_PROP_FPS) or 0.0
        out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap_out.release()
        print(f"[*] Output video: {out_fps} FPS, {out_frames} frames (written {frames_written})")
    except Exception:
        print(f"[*] Output frames written: {frames_written}")
    print(f"[*] Done! Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--amp", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
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
    interpolate_video(args.input, args.output, model, args.amp, pad_multiple)


if __name__ == "__main__":
    main()
