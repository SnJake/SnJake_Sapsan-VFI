import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from src.utils.video_backend import open_reader


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".webm", ".flv", ".mov", ".mpg", ".mpeg"}
NPZ_EXTS = {".npz"}


def scan_videos(roots: Sequence[str], extensions: Optional[Sequence[str]] = None) -> List[str]:
    exts = {e.lower() for e in (extensions or VIDEO_EXTS)}
    files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    files.append(os.path.join(dirpath, name))
    return files


def scan_npz(roots: Sequence[str]) -> List[str]:
    files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in NPZ_EXTS:
                    files.append(os.path.join(dirpath, name))
    return files


def build_manifest(
    roots: Sequence[str],
    manifest_path: str,
    extensions: Optional[Sequence[str]] = None,
    backend: str = "opencv",
    errors_path: Optional[str] = None,
    force_count: bool = False,
) -> int:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    files = scan_videos(roots, extensions)
    count = 0
    err_fp = None
    if errors_path:
        os.makedirs(os.path.dirname(errors_path), exist_ok=True)
        err_fp = open(errors_path, "a", encoding="utf-8")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        for path in files:
            try:
                reader = open_reader(path, backend)
            except Exception as exc:
                if err_fp is not None:
                    err_fp.write(json.dumps({"path": path, "error": str(exc)}, ensure_ascii=False) + "\n")
                continue
            try:
                num_frames = reader.get_count()
                if not num_frames or num_frames <= 0:
                    if force_count:
                        num_frames = reader.count_frames()
                    else:
                        if err_fp is not None:
                            err_fp.write(json.dumps({"path": path, "error": "no_frame_count"}, ensure_ascii=False) + "\n")
                        continue
                rec = {"path": path, "num_frames": int(num_frames)}
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            finally:
                reader.close()
    if err_fp is not None:
        err_fp.close()
    return count


def load_manifest(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass
class SampleConfig:
    crop_size: Tuple[int, int]
    min_size: Tuple[int, int]
    resize_short_edge: int
    random_flip: bool
    stride_min: int
    stride_max: int


class VideoReaderCache:
    def __init__(self, max_items: int, backend: str):
        self.max_items = max_items
        self.backend = backend
        self.cache: OrderedDict[str, object] = OrderedDict()

    def get(self, path: str):
        if path in self.cache:
            cap = self.cache.pop(path)
            self.cache[path] = cap
            return cap
        try:
            cap = open_reader(path, self.backend)
        except Exception:
            return None
        self.cache[path] = cap
        if len(self.cache) > self.max_items:
            _, old = self.cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return cap

    def close(self) -> None:
        for cap in self.cache.values():
            try:
                cap.close()
            except Exception:
                pass
        self.cache.clear()


class VideoTripletDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        sample_cfg: SampleConfig,
        reader_cache: int = 4,
        backend: str = "opencv",
        max_tries: int = 10,
        skip_failed: bool = True,
        max_total_tries: Optional[int] = None,
        return_path: bool = False,
    ) -> None:
        self.records = records
        if sample_cfg.stride_max < sample_cfg.stride_min:
            sample_cfg.stride_max = sample_cfg.stride_min
        self.sample_cfg = sample_cfg
        self.max_tries = max_tries
        self.skip_failed = skip_failed
        if max_total_tries is None:
            max_total_tries = max_tries if not skip_failed else max(max_tries * 50, max_tries)
        self.max_total_tries = int(max_total_tries)
        self.return_path = return_path
        self.reader_cache = VideoReaderCache(reader_cache, backend)

    def __len__(self) -> int:
        return len(self.records)

    def _read_frame(self, cap, index: int) -> Optional[np.ndarray]:
        try:
            return cap.get_frame(index)
        except Exception:
            return None

    def _sample_indices(self, num_frames: int) -> Optional[Tuple[int, int, int]]:
        stride = random.randint(self.sample_cfg.stride_min, self.sample_cfg.stride_max)
        max_start = num_frames - 1 - 2 * stride
        if max_start <= 0:
            return None
        start = random.randint(0, max_start)
        mid = start + stride
        end = start + 2 * stride
        return start, mid, end

    def _resize_min(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        h, w = frames[0].shape[:2]
        min_h, min_w = self.sample_cfg.min_size
        scale = 1.0
        if h < min_h or w < min_w:
            scale = max(min_h / h, min_w / w)
        if self.sample_cfg.resize_short_edge and min(h, w) != self.sample_cfg.resize_short_edge:
            scale = max(scale, self.sample_cfg.resize_short_edge / min(h, w))
        if scale == 1.0:
            return frames
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_CUBIC) for f in frames]

    def _random_crop(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        crop_h, crop_w = self.sample_cfg.crop_size
        h, w = frames[0].shape[:2]
        if crop_h <= 0 or crop_w <= 0:
            return frames
        if h == crop_h and w == crop_w:
            return frames
        if h < crop_h or w < crop_w:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            frames = [
                cv2.copyMakeBorder(f, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
                for f in frames
            ]
            h, w = frames[0].shape[:2]
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        return [f[y : y + crop_h, x : x + crop_w] for f in frames]

    def _maybe_flip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self.sample_cfg.random_flip and random.random() < 0.5:
            return [np.ascontiguousarray(f[:, ::-1]) for f in frames]
        return frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_total = max(self.max_total_tries, self.max_tries)
        for _ in range(max_total):
            rec = self.records[idx % len(self.records)]
            path = rec["path"]
            cap = self.reader_cache.get(path)
            if cap is None:
                idx = random.randint(0, len(self.records) - 1)
                continue
            num_frames = rec.get("num_frames")
            if not num_frames:
                try:
                    num_frames = cap.get_count()
                except Exception:
                    num_frames = 0
            if num_frames < 3:
                idx = random.randint(0, len(self.records) - 1)
                continue
            indices = self._sample_indices(num_frames)
            if indices is None:
                idx = random.randint(0, len(self.records) - 1)
                continue
            i0, i1, i2 = indices
            f0 = self._read_frame(cap, i0)
            f1 = self._read_frame(cap, i1)
            f2 = self._read_frame(cap, i2)
            if f0 is None or f1 is None or f2 is None:
                idx = random.randint(0, len(self.records) - 1)
                continue
            frames = [f0, f1, f2]
            frames = self._resize_min(frames)
            frames = self._random_crop(frames)
            frames = self._maybe_flip(frames)
            tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
            sample = {"frame0": tensors[0], "frame1": tensors[2], "gt": tensors[1]}
            if self.return_path:
                sample["path"] = path
            return sample
        raise RuntimeError("Failed to sample a valid triplet after max_total_tries")


class NpzTripletDataset(Dataset):
    def __init__(self, files: List[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx % len(self.files)]
        with np.load(path) as data:
            f0 = data["frame0"]
            f1 = data["frame1"]
            gt = data["gt"]
        f0 = torch.from_numpy(f0).permute(2, 0, 1).float() / 255.0
        f1 = torch.from_numpy(f1).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
        return {"frame0": f0, "frame1": f1, "gt": gt}


class WdsTripletDataset(IterableDataset):
    def __init__(self, urls, shuffle_buffer: int = 0, image_ext: str = "png") -> None:
        super().__init__()
        self.urls = urls
        self.shuffle_buffer = shuffle_buffer
        ext = (image_ext or "png").lower()
        self.image_ext = "jpg" if ext in ("jpg", "jpeg") else ext

    def __iter__(self):
        try:
            import webdataset as wds
        except Exception as exc:
            raise RuntimeError("webdataset is not installed") from exc

        dataset = wds.WebDataset(self.urls, handler=wds.handlers.warn_and_continue, shardshuffle=False)
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)
        ext = self.image_ext
        dataset = dataset.decode("rgb8").to_tuple(f"frame0.{ext}", f"frame1.{ext}", f"gt.{ext}")
        for f0, f1, gt in dataset:
            if not isinstance(f0, np.ndarray):
                f0 = np.asarray(f0)
            if not isinstance(f1, np.ndarray):
                f1 = np.asarray(f1)
            if not isinstance(gt, np.ndarray):
                gt = np.asarray(gt)
            if not f0.flags.writeable:
                f0 = f0.copy()
            if not f1.flags.writeable:
                f1 = f1.copy()
            if not gt.flags.writeable:
                gt = gt.copy()
            f0_t = torch.from_numpy(f0).permute(2, 0, 1).float() / 255.0
            f1_t = torch.from_numpy(f1).permute(2, 0, 1).float() / 255.0
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
            yield {"frame0": f0_t, "frame1": f1_t, "gt": gt_t}
