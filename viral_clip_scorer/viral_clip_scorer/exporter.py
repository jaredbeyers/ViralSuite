from __future__ import annotations

import subprocess
from pathlib import Path


def export_clip_ffmpeg(video_path: Path, start_s: float, end_s: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.01, float(end_s - start_s))
    # NOTE:
    # - Stream-copy cuts (-c copy) often produce black frames or odd fps at segment boundaries
    #   because cuts may start on non-keyframes.
    # - For a "creator tool" UX, it's better to default to a stable re-encode with fixed fps/pix_fmt.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        # Accurate seek (after -i)
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{dur:.3f}",
        # Normalize playback + compatibility
        "-vf",
        "format=yuv420p",
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
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

