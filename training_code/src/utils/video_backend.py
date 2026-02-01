from __future__ import annotations

from typing import Optional

import numpy as np


class VideoReaderBase:
    def get_count(self) -> Optional[int]:
        raise NotImplementedError

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def count_frames(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVReader(VideoReaderBase):
    def __init__(self, path: str) -> None:
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open")

    def get_count(self) -> Optional[int]:
        count = int(self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0)
        return count if count > 0 else None

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def count_frames(self) -> int:
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            count += 1
        return count

    def close(self) -> None:
        self.cap.release()


class DecordReader(VideoReaderBase):
    def __init__(self, path: str) -> None:
        import decord

        self.decord = decord
        self.vr = decord.VideoReader(path, ctx=decord.cpu(0))

    def get_count(self) -> Optional[int]:
        return int(len(self.vr))

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        try:
            frame = self.vr[int(index)].asnumpy()
        except Exception:
            return None
        return frame

    def count_frames(self) -> int:
        return int(len(self.vr))

    def close(self) -> None:
        self.vr = None


class PyAVReader(VideoReaderBase):
    def __init__(self, path: str) -> None:
        import av

        self.av = av
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.fps = None
        if self.stream.average_rate is not None:
            try:
                self.fps = float(self.stream.average_rate)
            except Exception:
                self.fps = None

    def get_count(self) -> Optional[int]:
        if self.stream.frames is not None and self.stream.frames > 0:
            return int(self.stream.frames)
        return None

    def _time_to_index(self, time_sec: float) -> Optional[int]:
        if self.fps is None:
            return None
        return int(round(time_sec * self.fps))

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if self.fps is None:
            return None
        target_sec = float(index) / self.fps
        try:
            target_pts = int(target_sec / float(self.stream.time_base))
        except Exception:
            target_pts = None
        if target_pts is not None:
            try:
                self.container.seek(target_pts, stream=self.stream, any_frame=False, backward=True)
            except Exception:
                pass
        for frame in self.container.decode(self.stream):
            if frame is None:
                continue
            if frame.time is None:
                continue
            idx = self._time_to_index(float(frame.time))
            if idx is None:
                continue
            if idx >= index:
                try:
                    return frame.to_rgb().to_ndarray()
                except Exception:
                    return None
        return None

    def count_frames(self) -> int:
        count = 0
        for frame in self.container.decode(self.stream):
            if frame is None:
                continue
            count += 1
        return count

    def close(self) -> None:
        self.container.close()


def _load_backend(name: str):
    name = (name or "opencv").lower()
    if name == "opencv":
        return OpenCVReader
    if name == "decord":
        return DecordReader
    if name == "pyav":
        return PyAVReader
    if name == "auto":
        return None
    raise ValueError(f"Unknown backend: {name}")


def open_reader(path: str, backend: str) -> VideoReaderBase:
    if backend == "auto":
        for name in ("decord", "pyav", "opencv"):
            try:
                cls = _load_backend(name)
                return cls(path)
            except Exception:
                continue
        raise RuntimeError("Failed to open with any backend")
    cls = _load_backend(backend)
    return cls(path)
