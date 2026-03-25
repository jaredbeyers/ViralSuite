from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from viral_clip_scorer.utils import slugify_filename


def _default_broll_dir() -> Path:
    # Suite root is two levels up from package directory:
    # ViralSuite/viral_clip_scorer/viral_clip_scorer/broll_manage.py
    pkg_dir = Path(__file__).resolve().parent
    suite_root = pkg_dir.parents[2]
    return (suite_root / "broll_videos").resolve()


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg and ensure `ffmpeg -version` works.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH. Install FFmpeg and ensure `ffprobe -version` works.")


def _is_video_file(path: Path) -> bool:
    exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    return path.is_file() and path.suffix.lower() in exts


def _trim_with_ffmpeg(in_path: Path, out_path: Path, trim_start_s: float, max_seconds: float) -> Path:
    _require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trim_start_s = max(0.0, float(trim_start_s))
    max_seconds = max(0.0, float(max_seconds))

    cmd = ["ffmpeg", "-y"]
    if trim_start_s > 0:
        cmd += ["-ss", f"{trim_start_s:.3f}"]
    cmd += ["-i", str(in_path)]
    if max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-c", "copy", str(out_path)]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        cmd2 = ["ffmpeg", "-y"]
        if trim_start_s > 0:
            cmd2 += ["-ss", f"{trim_start_s:.3f}"]
        cmd2 += ["-i", str(in_path)]
        if max_seconds > 0:
            cmd2 += ["-t", f"{max_seconds:.3f}"]
        cmd2 += [
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            "12",
            "-tune",
            "film",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-x264-params",
            "aq-mode=3:aq-strength=1.1:deblock=-1,-1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "384k",
            str(out_path),
        ]
        subprocess.check_call(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path.resolve()


def add_broll_video(
    *,
    video_path: Path,
    broll_dir: Path | None,
    label: str | None,
    trim_start_s: float = 0.0,
    max_seconds: float = 0.0,
) -> Path:
    """
    Copy a local video into the b-roll folder, optionally trimming/capping it.
    Returns destination file path.
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not _is_video_file(video_path):
        raise ValueError(f"Unsupported file type: {video_path.suffix}")

    dest_dir = (broll_dir or _default_broll_dir()).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = slugify_filename(label or video_path.stem, max_len=48)
    out_name = f"broll_{stamp}_{base}.mp4"
    out_path = dest_dir / out_name

    # If no trimming requested, just copy and return (keep original container/ext)
    if (trim_start_s <= 0) and (max_seconds <= 0):
        # Keep original extension to preserve quality; still make name safe
        out_path = dest_dir / f"broll_{stamp}_{base}{video_path.suffix.lower()}"
        shutil.copy2(video_path, out_path)
        return out_path.resolve()

    # For trimming, normalize to mp4 for consistent downstream use
    return _trim_with_ffmpeg(video_path, out_path, trim_start_s=float(trim_start_s), max_seconds=float(max_seconds))

