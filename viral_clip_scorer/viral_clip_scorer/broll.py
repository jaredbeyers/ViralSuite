from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path


def _ffprobe_streams(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return json.loads(out)


def has_video_stream(video_path: Path) -> bool:
    info = _ffprobe_streams(video_path)
    for s in info.get("streams", []) or []:
        if s.get("codec_type") == "video":
            return True
    return False


def duration_s(video_path: Path) -> float:
    info = _ffprobe_streams(video_path)
    fmt = info.get("format") or {}
    d = fmt.get("duration")
    try:
        return float(d)
    except Exception:
        return 0.0


def _cut_density(video_path: Path, probe_seconds: float = 180.0) -> float:
    """
    Quick 'interestingness' proxy based on how many scene cuts FFmpeg detects in the first N seconds.
    """
    vf = "select='gt(scene,0.30)',showinfo"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-t",
        f"{probe_seconds:.3f}",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log = (proc.stderr or "") + "\n" + (proc.stdout or "")
    # Each showinfo for a selected frame implies a detected cut
    cuts = log.count("showinfo")
    return cuts / max(1.0, float(probe_seconds))


def should_replace_visuals(video_path: Path, mode: str) -> bool:
    mode = (mode or "never").lower().strip()
    if mode == "never":
        return False
    if mode == "always":
        return True
    # auto
    if not has_video_stream(video_path):
        return True
    dens = _cut_density(video_path, probe_seconds=180.0)
    # Heuristic threshold: < ~0.01 cuts/sec => very static (fewer than ~2 cuts in 3 minutes)
    return dens < 0.01


def _pick_broll_files(broll_dir: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    files = [p for p in broll_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _make_broll_video(
    *,
    audio_source: Path,
    out_path: Path,
    broll_files: list[Path],
    seed: int = 7,
) -> Path:
    """
    Create a new mp4 with visuals from b-roll (looped) and audio from the source.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = duration_s(audio_source)
    if dur <= 0.0:
        raise RuntimeError("Could not determine duration for audio source.")

    rng = random.Random(seed)
    chosen = rng.choice(broll_files)

    # Build a simple looped visual from one b-roll file for duration, keep original audio.
    # -stream_loop -1 loops the visual input
    # scale/crop to 9:16 (1080x1920) for short-form by default
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(chosen),
        "-i",
        str(audio_source),
        "-t",
        f"{dur:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        vf,
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
        "-shortest",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def maybe_replace_visuals(
    *,
    video_path: Path,
    workdir: Path,
    broll_dir: Path | None,
    mode: str,
    seed: int = 7,
) -> Path:
    """
    If enabled, return a path to a derived video with replaced visuals (keeps original audio).
    Otherwise return the original video_path.
    """
    if not broll_dir:
        return video_path
    if not broll_dir.exists():
        return video_path

    if not should_replace_visuals(video_path, mode=mode):
        return video_path

    broll_files = _pick_broll_files(broll_dir)
    if not broll_files:
        return video_path

    out_path = workdir / "visual_replaced.mp4"
    if out_path.exists():
        return out_path

    return _make_broll_video(audio_source=video_path, out_path=out_path, broll_files=broll_files, seed=seed)

