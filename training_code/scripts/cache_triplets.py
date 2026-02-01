import argparse
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.dataset import build_manifest, load_manifest, SampleConfig, VideoTripletDataset


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def save_triplet(out_dir: str, idx: int, sample: Dict[str, torch.Tensor]) -> str:
    f0 = (sample["frame0"].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    f1 = (sample["frame1"].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    gt = (sample["gt"].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    name = f"triplet_{idx:08d}.npz"
    path = os.path.join(out_dir, name)
    np.savez_compressed(path, frame0=f0, frame1=f1, gt=gt)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

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
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    total = 0
    pbar = tqdm(total=args.count, desc="cache")
    for batch in loader:
        batch_size = batch["frame0"].shape[0]
        for i in range(batch_size):
            sample = {k: v[i].cpu() for k, v in batch.items()}
            save_triplet(args.out, total, sample)
            total += 1
            pbar.update(1)
            if total >= args.count:
                pbar.close()
                return


if __name__ == "__main__":
    main()
