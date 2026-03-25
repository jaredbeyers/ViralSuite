from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import re
import subprocess

from viral_clip_scorer.utils import clamp01, moving_avg, quantile, safe_div, softstep


@dataclass(frozen=True)
class VisualSignals:
    fps: float
    times_s: list[float]
    motion: list[float]  # normalized-ish optical flow magnitude proxy
    cut_strength: list[float]  # histogram distance proxy per frame
    face_presence: list[float]  # [0,1] per sampled frame
    face_centered: list[float]  # [0,1] per sampled frame (only when face present)

def compute_visual_signals(
    video_path: Path,
    sample_fps: float = 6.0,
    max_seconds: float | None = None,
) -> VisualSignals:
    """
    FFmpeg-only visual baseline:
    - scene cuts via ffmpeg `select=gt(scene,THRESH),showinfo` parsing pts_time
    - motion/face are left as 0.0 proxies (keeps pipeline runnable on Python 3.14 without numpy/opencv)
    """
    # Detect scene cut timestamps (where a cut is likely)
    thresh = 0.30
    vf = f"select='gt(scene,{thresh})',showinfo"
    cmd = ["ffmpeg", "-hide_banner", "-i", str(video_path), "-vf", vf, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg and ensure `ffmpeg -version` works.") from e

    log = (proc.stderr or "") + "\n" + (proc.stdout or "")
    # showinfo lines include: pts_time:12.345
    times = [float(x) for x in re.findall(r"pts_time:([0-9]*\\.?[0-9]+)", log)]
    if max_seconds is not None:
        times = [t for t in times if t <= max_seconds]

    # Convert cut times into sampled timeline: emit a small window around each cut
    # We’ll create a per-sample-fps “strength” signal: 1.0 at cut bins, else 0.0
    if not times:
        return VisualSignals(
            fps=float(sample_fps),
            times_s=[],
            motion=[],
            cut_strength=[],
            face_presence=[],
            face_centered=[],
        )

    t_end = max(times) + 0.5
    step = 1.0 / float(sample_fps)
    bins = int(max(1, t_end / step))
    tline = [i * step for i in range(bins)]
    cut = [0.0 for _ in range(bins)]
    for t in times:
        i = int(round(t / step))
        if 0 <= i < bins:
            cut[i] = 1.0

    cut_sm = moving_avg(cut, win=3)
    motion = [0.0 for _ in range(bins)]
    fp = [0.0 for _ in range(bins)]
    fc = [0.0 for _ in range(bins)]
    return VisualSignals(
        fps=float(sample_fps),
        times_s=tline,
        motion=motion,
        cut_strength=cut_sm,
        face_presence=fp,
        face_centered=fc,
    )


def segment_visual_scores(v: VisualSignals, start_s: float, end_s: float) -> dict[str, float]:
    # Compute summary statistics in window
    if not v.times_s:
        return {"visual_dynamics": 0.0, "scene_changes": 0.0, "face_engagement": 0.0}

    idxs = [i for i, t in enumerate(v.times_s) if start_s <= t <= end_s]
    if not idxs:
        return {"visual_dynamics": 0.0, "scene_changes": 0.0, "face_engagement": 0.0}

    mot = [v.motion[i] for i in idxs]
    cuts = [v.cut_strength[i] for i in idxs]
    fp = [v.face_presence[i] for i in idxs]
    fc = [v.face_centered[i] for i in idxs]

    mot_q90 = quantile(mot, 0.9)
    mot_mean = (sum(mot) / float(len(mot))) if mot else 0.0
    cut_q95 = quantile(cuts, 0.95)

    # Normalize heuristically: typical diff magnitudes are small (~0.01-0.08)
    motion_score = 0.6 * softstep(mot_mean, 0.01, 0.06) + 0.4 * softstep(mot_q90, 0.03, 0.12)
    scene_score = softstep(cut_q95, 0.08, 0.22)

    face_ratio = float(np.mean(fp)) if fp else 0.0
    face_center = safe_div(sum(fc), max(1e-6, sum(fp)), default=0.0)
    face_engagement = 0.65 * face_ratio + 0.35 * clamp01(face_center)

    return {
        "visual_dynamics": clamp01(0.7 * motion_score + 0.3 * scene_score),
        "scene_changes": clamp01(scene_score),
        "face_engagement": clamp01(face_engagement),
    }

