import argparse
import json
import os
import time
import tarfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.dataset import build_manifest, load_manifest, SampleConfig, VideoTripletDataset


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def encode_png(frame_rgb: np.ndarray) -> bytes:
    bgr = frame_rgb[:, :, ::-1]
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()


def encode_jpg(frame_rgb: np.ndarray, quality: int) -> bytes:
    bgr = frame_rgb[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPG")
    return buf.tobytes()


def encode_image(frame_rgb: np.ndarray, fmt: str, jpeg_quality: int) -> bytes:
    fmt = (fmt or "png").lower()
    if fmt in ("jpg", "jpeg"):
        return encode_jpg(frame_rgb, jpeg_quality)
    return encode_png(frame_rgb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--loader-timeout", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state-path", type=str, default=None)
    args = parser.parse_args()

    try:
        import webdataset as wds
    except Exception as exc:
        raise RuntimeError("webdataset is not installed") from exc

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    if not args.log_path:
        args.log_path = os.path.join(args.out, "cache_wds.log.jsonl")
    if not args.state_path:
        args.state_path = os.path.join(args.out, "cache_wds.state.json")
    if args.state_path:
        state_dir = os.path.dirname(args.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

    data_cfg = cfg["data"]
    manifest = data_cfg.get("manifest")
    if not manifest:
        if not data_cfg.get("roots"):
            raise ValueError("data.roots must be set when data.manifest is empty")
        manifest = os.path.join(args.out, "dataset_manifest.jsonl")
        build_manifest(
            data_cfg["roots"],
            manifest,
            data_cfg.get("extensions"),
            backend=str(args.backend or data_cfg.get("backend", "opencv")),
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

    dataset = VideoTripletDataset(
        records=records,
        sample_cfg=sample_cfg,
        reader_cache=int(data_cfg.get("reader_cache", 4)),
        backend=str(args.backend or data_cfg.get("backend", "opencv")),
        max_tries=int(data_cfg.get("max_tries", 10)),
        skip_failed=bool(data_cfg.get("skip_failed", True)),
        max_total_tries=data_cfg.get("max_total_tries"),
        return_path=True,
    )

    def build_loader() -> DataLoader:
        timeout = int(args.loader_timeout) if args.num_workers > 0 else 0
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            timeout=timeout,
            persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
        )

    shard_id = -1
    shard_count = 0
    writer = None
    log_fp: Optional[object] = None
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        log_fp = open(args.log_path, "a", encoding="utf-8")

    def log_event(payload: Dict) -> None:
        if log_fp is None:
            return
        payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        log_fp.flush()

    def save_state(total_count: int, shard_id_val: int) -> None:
        tmp_path = f"{args.state_path}.tmp"
        payload = {
            "total": int(total_count),
            "shard": int(shard_id_val),
            "shard_size": int(args.shard_size),
            "image_format": (args.image_format or "png").lower(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
        os.replace(tmp_path, args.state_path)

    def _as_file_url(path: str) -> str:
        path = os.path.abspath(path).replace("\\", "/")
        return f"file:{path}"

    def open_writer(sid: int):
        name = os.path.join(args.out, f"shard-{sid:06d}.tar")
        return wds.TarWriter(_as_file_url(name))

    def find_shards(out_dir: str) -> List[Tuple[int, str]]:
        shards: List[Tuple[int, str]] = []
        for name in os.listdir(out_dir):
            if not (name.startswith("shard-") and name.endswith(".tar")):
                continue
            sid_str = name[len("shard-") : -len(".tar")]
            if not sid_str.isdigit():
                continue
            sid = int(sid_str)
            shards.append((sid, os.path.join(out_dir, name)))
        shards.sort(key=lambda x: x[0])
        return shards

    def count_samples_in_tar(path: str, ext: str) -> int:
        count = 0
        try:
            with tarfile.open(path, "r") as tf:
                for member in tf.getmembers():
                    name = member.name
                    if name.endswith(f"frame0.{ext}"):
                        count += 1
        except Exception as exc:
            log_event({"event": "tar_read_error", "path": path, "error": str(exc)})
            return 0
        return count

    total = 0
    fmt = (args.image_format or "png").lower()
    ext = "jpg" if fmt in ("jpg", "jpeg") else "png"
    if args.resume:
        resume_total = 0
        resume_shard = -1
        if args.state_path and os.path.exists(args.state_path):
            try:
                with open(args.state_path, "r", encoding="utf-8") as fp:
                    state = json.load(fp)
                resume_total = int(state.get("total", 0))
                resume_shard = int(state.get("shard", resume_total // args.shard_size))
                log_event(
                    {
                        "event": "resume_state",
                        "total": int(resume_total),
                        "shard": int(resume_shard),
                        "state_path": args.state_path,
                    }
                )
            except Exception as exc:
                log_event({"event": "resume_state_error", "error": str(exc), "state_path": args.state_path})
        if resume_total <= 0:
            shards = find_shards(args.out)
            if shards:
                contiguous = []
                expected = 0
                for sid, path in shards:
                    if sid != expected:
                        log_event(
                            {
                                "event": "resume_gap",
                                "expected_shard": int(expected),
                                "found_shard": int(sid),
                                "path": path,
                            }
                        )
                        break
                    contiguous.append((sid, path))
                    expected += 1
                if contiguous:
                    last_sid, last_path = contiguous[-1]
                    last_count = count_samples_in_tar(last_path, ext)
                    if last_count >= args.shard_size:
                        resume_total = len(contiguous) * args.shard_size
                    else:
                        resume_total = max(len(contiguous) - 1, 0) * args.shard_size
                        try:
                            os.remove(last_path)
                            log_event({"event": "resume_drop_partial", "path": last_path, "count": int(last_count)})
                        except Exception as exc:
                            log_event({"event": "resume_drop_error", "path": last_path, "error": str(exc)})
                log_event(
                    {
                        "event": "resume_scan",
                        "total": int(resume_total),
                        "shards": int(len(contiguous)),
                    }
                )
        total = max(0, min(int(resume_total), int(args.count)))
        if total % args.shard_size != 0:
            floored = (total // args.shard_size) * args.shard_size
            log_event({"event": "resume_floor", "from": int(total), "to": int(floored)})
            total = floored
        if total % args.shard_size == 0:
            resume_path = os.path.join(args.out, f"shard-{total // args.shard_size:06d}.tar")
            if os.path.exists(resume_path):
                try:
                    os.remove(resume_path)
                    log_event({"event": "resume_drop_current", "path": resume_path})
                except Exception as exc:
                    log_event({"event": "resume_drop_error", "path": resume_path, "error": str(exc)})
        save_state(total, total // args.shard_size)
    pbar = tqdm(total=args.count, initial=total, desc="wds-cache")
    loader = build_loader()
    data_iter = iter(loader)
    try:
        while total < args.count:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                continue
            except Exception as exc:
                log_event({"event": "loader_error", "total": int(total), "error": str(exc)})
                loader = build_loader()
                data_iter = iter(loader)
                continue

            paths = batch.get("path")
            batch_size = batch["frame0"].shape[0]
            for i in range(batch_size):
                sid = total // args.shard_size
                if sid != shard_id:
                    if writer is not None:
                        writer.close()
                    writer = open_writer(sid)
                    shard_id = sid
                    shard_count = 0
                    save_state(total, shard_id)
                    log_event(
                        {
                            "event": "shard_open",
                            "shard": int(shard_id),
                            "total": int(total),
                            "path": os.path.join(args.out, f"shard-{sid:06d}.tar"),
                        }
                    )
                path = paths[i] if paths is not None else None
                f0 = (batch["frame0"][i].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
                f1 = (batch["frame1"][i].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
                gt = (batch["gt"][i].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
                try:
                    sample = {
                        "__key__": f"{total:08d}",
                        f"frame0.{ext}": encode_image(f0, fmt, args.jpeg_quality),
                        f"frame1.{ext}": encode_image(f1, fmt, args.jpeg_quality),
                        f"gt.{ext}": encode_image(gt, fmt, args.jpeg_quality),
                    }
                    writer.write(sample)
                except Exception as exc:
                    log_event(
                        {
                            "event": "write_error",
                            "total": int(total),
                            "path": path,
                            "error": str(exc),
                        }
                    )
                    continue
                shard_count += 1
                total += 1
                pbar.update(1)
                if args.log_interval > 0 and total % int(args.log_interval) == 0:
                    log_event(
                        {
                            "event": "progress",
                            "total": int(total),
                            "shard": int(shard_id),
                            "path": path,
                        }
                    )
                    if total % args.shard_size == 0:
                        save_state(total, shard_id)
                if total >= args.count:
                    break
    finally:
        pbar.close()
        if writer is not None:
            writer.close()
        if log_fp is not None:
            log_fp.close()
    return


if __name__ == "__main__":
    main()
