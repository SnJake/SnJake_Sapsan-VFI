import argparse
import os
import time
from typing import Dict

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from src.dataset import (
    build_manifest,
    load_manifest,
    SampleConfig,
    VideoTripletDataset,
    NpzTripletDataset,
    WdsTripletDataset,
    scan_npz,
)
from src.models import VFIModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.ema import EMA
from src.utils.logging import JsonlLogger
from src.utils.losses import (
    build_perceptual_loss,
    charbonnier_loss,
    color_consistency_loss,
    brightness_consistency_loss,
    edge_aware_flow_smoothness,
    flow_smoothness,
    gradient_loss,
    laplacian_loss,
    ssim,
)
from src.utils.misc import get_autocast_dtype, maybe_compile, set_seed, to_device
from src.utils.schedulers import build_scheduler


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def save_config(cfg: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False, allow_unicode=False)


def build_dataset(cfg: Dict, run_dir: str) -> VideoTripletDataset:
    data_cfg = cfg["data"]
    fmt = str(data_cfg.get("format", "video")).lower()
    if fmt == "npz":
        npz_dir = data_cfg.get("npz_dir")
        npz_manifest = data_cfg.get("npz_manifest")
        if npz_manifest:
            records = load_manifest(npz_manifest)
            files = [rec["path"] for rec in records if rec.get("path")]
        elif npz_dir:
            files = scan_npz([npz_dir])
        else:
            raise ValueError("data.npz_dir or data.npz_manifest must be set for format=npz")
        if not files:
            raise ValueError("No .npz files found for format=npz")
        return NpzTripletDataset(files)
    if fmt == "wds":
        urls = data_cfg.get("wds_urls")
        if not urls:
            raise ValueError("data.wds_urls must be set for format=wds")
        shuffle_buf = int(data_cfg.get("wds_shuffle", 0))
        image_ext = data_cfg.get("wds_image_ext", "png")
        return WdsTripletDataset(urls, shuffle_buffer=shuffle_buf, image_ext=image_ext)
    manifest = data_cfg.get("manifest")
    if not manifest:
        if not data_cfg.get("roots"):
            raise ValueError("data.roots must be set when data.manifest is empty")
        manifest = os.path.join(run_dir, "dataset_manifest.jsonl")
        build_manifest(
            data_cfg["roots"],
            manifest,
            data_cfg.get("extensions"),
            backend=str(data_cfg.get("backend", "opencv")),
            errors_path=data_cfg.get("manifest_errors_log"),
            force_count=bool(data_cfg.get("manifest_force_count", False)),
        )
    records = load_manifest(manifest)
    sample_cfg = SampleConfig(
        crop_size=tuple(data_cfg.get("train_crop", [256, 256])),
        min_size=tuple(data_cfg.get("min_size", [256, 256])),
        resize_short_edge=int(data_cfg.get("resize_short_edge", 0)),
        random_flip=bool(data_cfg.get("random_horizontal_flip", True)),
        stride_min=int(data_cfg.get("frame_stride_min", 1)),
        stride_max=int(data_cfg.get("frame_stride_max", 1)),
    )
    return VideoTripletDataset(
        records=records,
        sample_cfg=sample_cfg,
        reader_cache=int(data_cfg.get("reader_cache", 4)),
        backend=str(data_cfg.get("backend", "opencv")),
        max_tries=int(data_cfg.get("max_tries", 10)),
        skip_failed=bool(data_cfg.get("skip_failed", True)),
        max_total_tries=data_cfg.get("max_total_tries"),
    )


def _downsample(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)


def compute_loss(
    cfg: Dict,
    output: torch.Tensor,
    target: torch.Tensor,
    aux: Dict,
    batch: Dict[str, torch.Tensor],
    perceptual,
) -> Dict[str, torch.Tensor]:
    loss_cfg = cfg["loss"]
    l1 = torch.nn.functional.l1_loss(output, target)
    charb = charbonnier_loss(output, target, eps=float(loss_cfg.get("charbonnier_eps", 1e-3)))
    ssim_val = ssim(output, target)
    ssim_loss = 1.0 - ssim_val
    flow_smooth = flow_smoothness(aux["flow01"]) + flow_smoothness(aux["flow10"])
    flow_edge = torch.tensor(0.0, device=output.device)
    if float(loss_cfg.get("flow_edge_weight", 0.0)) > 0.0:
        flow_edge = edge_aware_flow_smoothness(
            aux["flow01"],
            batch["frame0"],
            alpha=float(loss_cfg.get("flow_edge_alpha", 10.0)),
        )
        flow_edge = flow_edge + edge_aware_flow_smoothness(
            aux["flow10"],
            batch["frame1"],
            alpha=float(loss_cfg.get("flow_edge_alpha", 10.0)),
        )
    grad = torch.tensor(0.0, device=output.device)
    if float(loss_cfg.get("grad_weight", 0.0)) > 0.0:
        grad = gradient_loss(output, target)
    lap = torch.tensor(0.0, device=output.device)
    if float(loss_cfg.get("laplacian_weight", 0.0)) > 0.0:
        lap = laplacian_loss(output, target)
    warp = torch.tensor(0.0, device=output.device)
    occ_reg = torch.tensor(0.0, device=output.device)
    warp0 = aux.get("warp0_raw", aux.get("warp0"))
    warp1 = aux.get("warp1_raw", aux.get("warp1"))
    occ0 = aux.get("occ0")
    occ1 = aux.get("occ1")
    if float(loss_cfg.get("warp_weight", 0.0)) > 0.0 and warp0 is not None and warp1 is not None:
        if occ0 is not None and occ1 is not None:
            warp = (occ0 * (warp0 - target).abs()).mean() + (occ1 * (warp1 - target).abs()).mean()
            occ_reg = torch.abs(occ0 + occ1 - 1.0).mean()
        else:
            warp = torch.nn.functional.l1_loss(warp0, target) + torch.nn.functional.l1_loss(warp1, target)

    perceptual_val = torch.tensor(0.0, device=output.device)
    if perceptual is not None:
        perceptual_val = perceptual(output, target)

    color_cons = torch.tensor(0.0, device=output.device)
    bright_cons = torch.tensor(0.0, device=output.device)
    color_w = float(loss_cfg.get("color_consistency_weight", 0.0))
    bright_w = float(loss_cfg.get("brightness_consistency_weight", 0.0))
    if color_w > 0.0:
        color_cons = color_consistency_loss(
            output,
            target,
            use_std=bool(loss_cfg.get("color_consistency_use_std", True)),
        )
    if bright_w > 0.0:
        bright_cons = brightness_consistency_loss(
            output,
            target,
            use_std=bool(loss_cfg.get("brightness_consistency_use_std", True)),
        )

    ms_total = torch.tensor(0.0, device=output.device)
    ms_scales = loss_cfg.get("multiscale_scales")
    ms_weights = loss_cfg.get("multiscale_weights")
    if ms_scales:
        if not ms_weights:
            ms_weights = [1.0 for _ in ms_scales]
        l1_w = float(loss_cfg.get("l1_weight", 1.0))
        charb_w = float(loss_cfg.get("charbonnier_weight", 0.5))
        ssim_w = float(loss_cfg.get("ssim_weight", 0.1))
        grad_w = float(loss_cfg.get("grad_weight", 0.0))
        lap_w = float(loss_cfg.get("laplacian_weight", 0.0))
        for scale, weight in zip(ms_scales, ms_weights):
            scale = float(scale)
            weight = float(weight)
            if scale <= 0.0 or weight == 0.0 or scale == 1.0:
                continue
            out_s = _downsample(output, scale)
            tgt_s = _downsample(target, scale)
            ms_total = ms_total + weight * (
                l1_w * torch.nn.functional.l1_loss(out_s, tgt_s)
                + charb_w * charbonnier_loss(out_s, tgt_s, eps=float(loss_cfg.get("charbonnier_eps", 1e-3)))
                + ssim_w * (1.0 - ssim(out_s, tgt_s))
                + grad_w * gradient_loss(out_s, tgt_s)
                + lap_w * laplacian_loss(out_s, tgt_s)
            )

    total = (
        float(loss_cfg.get("l1_weight", 1.0)) * l1
        + float(loss_cfg.get("charbonnier_weight", 0.5)) * charb
        + float(loss_cfg.get("ssim_weight", 0.1)) * ssim_loss
        + float(loss_cfg.get("flow_smooth_weight", 0.01)) * flow_smooth
        + float(loss_cfg.get("flow_edge_weight", 0.0)) * flow_edge
        + float(loss_cfg.get("warp_weight", 0.0)) * warp
        + float(loss_cfg.get("occ_reg_weight", 0.0)) * occ_reg
        + float(loss_cfg.get("grad_weight", 0.0)) * grad
        + float(loss_cfg.get("laplacian_weight", 0.0)) * lap
        + float(loss_cfg.get("perceptual_weight", 0.0)) * perceptual_val
        + float(loss_cfg.get("multiscale_weight", 0.0)) * ms_total
        + color_w * color_cons
        + bright_w * bright_cons
    )
    return {
        "total": total,
        "l1": l1,
        "charbonnier": charb,
        "ssim": ssim_val,
        "flow_smooth": flow_smooth,
        "flow_edge": flow_edge,
        "warp": warp,
        "occ_reg": occ_reg,
        "grad": grad,
        "laplacian": lap,
        "perceptual": perceptual_val,
        "multiscale": ms_total,
        "color_consistency": color_cons,
        "brightness_consistency": bright_cons,
    }


def train(cfg: Dict) -> None:
    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    set_seed(int(cfg.get("seed", 42)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset = build_dataset(cfg, run_dir)
    loader_cfg = cfg["loader"]
    use_shuffle = not isinstance(dataset, IterableDataset)
    loader = DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 4)),
        shuffle=True if use_shuffle else False,
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=True,
        persistent_workers=bool(loader_cfg.get("persistent_workers", True)),
        prefetch_factor=int(loader_cfg.get("prefetch_factor", 2)) if int(loader_cfg.get("num_workers", 0)) > 0 else None,
    )

    steps_per_epoch = int(cfg["train"].get("steps_per_epoch") or len(loader))
    steps_per_epoch = int(cfg["train"].get("steps_per_epoch") or len(loader))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    total_steps = (steps_per_epoch * int(cfg["train"].get("epochs", 1))) // grad_accum

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        perf = cfg.get("performance", {})
        torch.backends.cuda.enable_flash_sdp(bool(perf.get("flash_sdp", True)))
        torch.backends.cuda.enable_mem_efficient_sdp(bool(perf.get("mem_efficient_sdp", True)))
        torch.backends.cuda.enable_math_sdp(bool(perf.get("math_sdp", True)))
    model = VFIModel(cfg["model"]).to(device)

    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 2e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
    )

    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), total_steps)

    amp = str(cfg["train"].get("amp", "no")).lower()
    autocast_dtype = get_autocast_dtype(amp)
    use_cuda = device.type == "cuda"
    autocast_enabled = autocast_dtype is not None and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=(amp == "fp16" and use_cuda))

    ema = None
    if cfg.get("ema", {}).get("enable", True):
        ema = EMA(decay=float(cfg["ema"].get("decay", 0.9999)), device=device)
        ema.register(model)

    start_epoch = 0
    global_step = 0
    ckpt_state = {}
    resume_path = cfg["train"].get("resume")
    if resume_path:
        start_epoch, global_step, ckpt_state = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, ema, "cpu"
        )
        if ema is not None and not ckpt_state.get("ema", False):
            ema = EMA(decay=float(cfg["ema"].get("decay", 0.9999)), device=device)
            ema.register(model)
            print("[!] EMA state not found in checkpoint; reinitialized from loaded weights.")

    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    clip_grad = float(cfg["train"].get("clip_grad_norm", 0.0))
    log_interval = int(cfg["train"].get("log_interval", 50))
    save_interval = int(cfg["train"].get("save_interval_epochs", 1))

    logger = JsonlLogger(os.path.join(run_dir, "logs", "train.jsonl"))

    perceptual = build_perceptual_loss(cfg.get("loss", {}), device)
    loss_cfg = cfg.get("loss", {})
    temporal_every = max(1, int(loss_cfg.get("temporal_every", 1)))
    temporal_accum = 0.0
    temporal_count = 0

    model_fwd = maybe_compile(model, cfg, name="train model")
    model.train()
    model_fwd.train()
    data_iter = iter(loader)
    profile = bool(cfg["train"].get("profile", False))
    prof_interval = int(cfg["train"].get("profile_interval", log_interval))
    prof_warmup = int(cfg["train"].get("profile_warmup_steps", 10))
    prof_max = int(cfg["train"].get("profile_max_steps", 200))
    prof_count = 0
    accum = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "step": 0.0, "total": 0.0}
    for epoch in range(start_epoch, int(cfg["train"].get("epochs", 1))):
        epoch_start = time.time()
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}")
        for step_in_epoch in pbar:
            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_total0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = to_device(batch, device)
            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_data = time.perf_counter()

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                output, aux = model_fwd(batch["frame0"], batch["frame1"], t=0.5)
                losses = compute_loss(cfg, output, batch["gt"], aux, batch, perceptual)

                temporal = torch.tensor(0.0, device=output.device)
                temporal_weight = float(loss_cfg.get("temporal_weight", 0.0))
                if temporal_weight > 0.0 and (global_step % temporal_every == 0):
                    t_inner = float(loss_cfg.get("temporal_t", 0.25))
                    t_inner = max(1e-3, min(0.499, t_inner))
                    pred_a, _ = model_fwd(batch["frame0"], batch["frame1"], t=t_inner)
                    pred_b, _ = model_fwd(batch["frame0"], batch["frame1"], t=1.0 - t_inner)
                    if bool(loss_cfg.get("temporal_detach", True)):
                        pred_a = pred_a.detach()
                        pred_b = pred_b.detach()
                    pred_mid, _ = model_fwd(pred_a, pred_b, t=0.5)
                    temporal = torch.nn.functional.l1_loss(pred_mid, output)
                    losses["total"] = losses["total"] + temporal_weight * temporal
                    temporal_accum += float(temporal.item())
                    temporal_count += 1
                losses["temporal"] = temporal
                loss = losses["total"] / grad_accum
            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.perf_counter()

            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_bwd = time.perf_counter()

            if (step_in_epoch + 1) % grad_accum == 0:
                if clip_grad > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)
            if profile and device.type == "cuda":
                torch.cuda.synchronize()
            t_step = time.perf_counter()

            global_step += 1
            if profile and global_step > prof_warmup and prof_count < prof_max:
                accum["data"] += t_data - t_total0
                accum["fwd"] += t_fwd - t_data
                accum["bwd"] += t_bwd - t_fwd
                accum["step"] += t_step - t_bwd
                accum["total"] += t_step - t_total0
                prof_count += 1
                if prof_count % prof_interval == 0:
                    denom = float(prof_count)
                    log_data = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "prof_data_s": accum["data"] / denom,
                        "prof_fwd_s": accum["fwd"] / denom,
                        "prof_bwd_s": accum["bwd"] / denom,
                        "prof_step_s": accum["step"] / denom,
                        "prof_total_s": accum["total"] / denom,
                    }
                    logger.log(log_data)
                    pbar.set_postfix(
                        {
                            "loss": f"{losses['total'].item():.4f}",
                            "dt": f"{log_data['prof_data_s']:.2f}",
                            "ft": f"{log_data['prof_fwd_s']:.2f}",
                            "bt": f"{log_data['prof_bwd_s']:.2f}",
                        }
                    )
            if global_step % log_interval == 0:
                temporal_log = temporal_accum / temporal_count if temporal_count > 0 else 0.0
                log_data = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(losses["total"].item()),
                    "l1": float(losses["l1"].item()),
                    "charbonnier": float(losses["charbonnier"].item()),
                    "ssim": float(losses["ssim"].item()),
                    "flow_smooth": float(losses["flow_smooth"].item()),
                    "flow_edge": float(losses["flow_edge"].item()),
                    "warp": float(losses["warp"].item()),
                    "occ_reg": float(losses["occ_reg"].item()),
                    "grad": float(losses["grad"].item()),
                    "laplacian": float(losses["laplacian"].item()),
                    "perceptual": float(losses["perceptual"].item()),
                    "multiscale": float(losses["multiscale"].item()),
                    "color_consistency": float(losses["color_consistency"].item()),
                    "brightness_consistency": float(losses["brightness_consistency"].item()),
                    "temporal": float(temporal_log),
                    "temporal_n": int(temporal_count),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                logger.log(log_data)
                pbar.set_postfix({"loss": f"{log_data['loss']:.4f}", "lr": f"{log_data['lr']:.2e}"})
                temporal_accum = 0.0
                temporal_count = 0

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == int(cfg["train"].get("epochs", 1)):
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            full_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}_full.pt")
            weights_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}_weights.pt")
            save_checkpoint(full_path, weights_path, model, optimizer, scheduler, scaler, ema, epoch + 1, global_step)
            save_checkpoint(
                os.path.join(ckpt_dir, "last_full.pt"),
                os.path.join(ckpt_dir, "last_weights.pt"),
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
                epoch + 1,
                global_step,
            )
        epoch_time = time.time() - epoch_start
        logger.log({"epoch": epoch + 1, "epoch_time_sec": epoch_time})

    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.device:
        cfg["device"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()
