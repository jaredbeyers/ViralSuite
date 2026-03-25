from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class DownloadError(RuntimeError):
    pass


def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise DownloadError(f"{name} not found on PATH. Install FFmpeg and ensure `{name} -version` works.")


def _trim_with_ffmpeg(
    *,
    in_path: Path,
    out_path: Path,
    trim_start_s: float,
    max_seconds: float,
) -> Path:
    _require_bin("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trim_start_s = max(0.0, float(trim_start_s))
    max_seconds = max(0.0, float(max_seconds))

    cmd = ["ffmpeg", "-y"]
    if trim_start_s > 0:
        # Fast seek (may cut on keyframe). Good enough for b-roll.
        cmd += ["-ss", f"{trim_start_s:.3f}"]
    cmd += ["-i", str(in_path)]
    if max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-c", "copy", str(out_path)]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Fallback: re-encode for accuracy/compatibility
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
            "slow",
            "-crf",
            "14",
            "-tune",
            "film",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            str(out_path),
        ]
        subprocess.check_call(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path.resolve()


def download_video(
    *,
    url: str,
    outdir: Path,
    format_selector: str = "bv*+ba/b",
    base_filename: str | None = None,
    no_playlist: bool = False,
    cookies_path: Path | None = None,
    trim_start_s: float = 0.0,
    max_seconds: float = 0.0,
) -> Path:
    """
    Download a URL using yt-dlp and return the final file path.

    Notes:
    - Requires FFmpeg on PATH for best compatibility/merging.
    - Respects site TOS; use responsibly.
    """
    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except Exception as e:
        raise DownloadError("yt-dlp is not installed. Run: python -m pip install -r requirements.txt") from e

    outdir.mkdir(parents=True, exist_ok=True)
    tmpl_name = base_filename if base_filename else "%(title).120s_%(id)s"
    outtmpl = str((outdir / f"{tmpl_name}.%(ext)s").resolve())

    has_ffmpeg = shutil.which("ffmpeg") is not None
    # If FFmpeg isn't available, we must avoid formats that require merging.
    # `best` tends to select a single progressive file when available.
    if (not has_ffmpeg) and format_selector.strip() == "bv*+ba/b":
        format_selector = "best"

    ydl_opts: dict = {
        "outtmpl": outtmpl,
        "format": format_selector,
        "noplaylist": bool(no_playlist),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Make downloads more resilient to transient HTTP issues
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep_functions": {"http": lambda n: min(5 * n, 30)},
        "socket_timeout": 30,
        "concurrent_fragment_downloads": 1,
        "http_chunk_size": 10485760,
        # Basic UA helps with some edge cases
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    }
    if has_ffmpeg:
        ydl_opts["merge_output_format"] = "mp4"
    if cookies_path:
        ydl_opts["cookiefile"] = str(cookies_path)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Playlist returns dict with entries
        if isinstance(info, dict) and info.get("_type") == "playlist":
            entries = info.get("entries") or []
            # After download=True, yt-dlp usually gives resolved entries; pick first.
            info0 = next((e for e in entries if isinstance(e, dict)), None)
            if not info0:
                raise DownloadError("Playlist download produced no entries.")
            info = info0

        if not isinstance(info, dict):
            raise DownloadError("yt-dlp returned unexpected info type.")

        # Determine output path robustly
        path_str = info.get("filepath") or info.get("_filename")
        if not path_str:
            try:
                path_str = ydl.prepare_filename(info)
            except Exception:
                path_str = None

        def _resolve(p0: str | None) -> Path | None:
            if not p0:
                return None
            p = Path(p0)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            return p if p.exists() else None

        p = _resolve(path_str)
        if p is not None:
            return p.resolve()

        # Fallback: search in outdir by id/title template
        vid = str(info.get("id") or "").strip()
        patterns = []
        if vid:
            patterns.append(f"*_{vid}.*")
            patterns.append(f"*{vid}*.*")
        patterns.append(f"{tmpl_name}.*")

        candidates: list[Path] = []
        for pat in patterns:
            candidates.extend(list(outdir.glob(pat)))
        candidates = sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            downloaded = candidates[0].resolve()
        else:
            raise DownloadError("Download finished but could not locate the output file in the output directory.")

        # Optional trim/limit
        if (trim_start_s and trim_start_s > 0) or (max_seconds and max_seconds > 0):
            trimmed = outdir / f"{(base_filename or downloaded.stem)}_trimmed.mp4"
            return _trim_with_ffmpeg(
                in_path=downloaded,
                out_path=trimmed,
                trim_start_s=float(trim_start_s),
                max_seconds=float(max_seconds),
            )

        return downloaded

