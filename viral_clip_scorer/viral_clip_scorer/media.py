from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import VideoFileClip


@dataclass(frozen=True)
class VideoMeta:
    duration_s: float
    fps: float
    w: int
    h: int


def read_video_meta(video_path: Path) -> VideoMeta:
    with VideoFileClip(str(video_path)) as clip:
        return VideoMeta(
            duration_s=float(clip.duration or 0.0),
            fps=float(clip.fps or 0.0),
            w=int(clip.w or 0),
            h=int(clip.h or 0),
        )


def sample_frames(video_path: Path, sample_fps: float = 2.0, max_frames: int = 2400) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (times_s, frames_bgr) where frames_bgr is uint8 [N,H,W,3].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / max(0.1, sample_fps))))
    times: list[float] = []
    frames: list[np.ndarray] = []

    idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        grabbed = cap.grab()
        if not grabbed:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            times.append(float(t_ms) / 1000.0)
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return np.asarray(times, dtype=np.float32), np.stack(frames, axis=0).astype(np.uint8)

