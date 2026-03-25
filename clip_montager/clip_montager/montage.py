from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
import re
from pathlib import Path

from clip_montager.utils import slugify_filename


@dataclass(frozen=True)
class Clip:
    start_s: float
    end_s: float
    overall_score: float
    caption: str | None = None
    # Optional transcript text corresponding to this clip (if available)
    text: str | None = None

    @property
    def length_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg and ensure `ffmpeg -version` works.")


def _load_report(report_path: Path) -> tuple[Path, list[Clip]]:
    data = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
    video_path = Path(data["video_path"])
    clips_raw = data.get("clips") or []
    clips: list[Clip] = []
    for c in clips_raw:
        try:
            debug = c.get("debug") or {}
            clips.append(
                Clip(
                    start_s=float(c["start_s"]),
                    end_s=float(c["end_s"]),
                    overall_score=float(c.get("overall_score", 0.0)),
                    caption=(c.get("best_caption") or None),
                    # Use the transcript preview that viral_clip_scorer stores per clip, if present.
                    text=(debug.get("clip_text_preview") or None),
                )
            )
        except Exception:
            continue
    return video_path, clips


_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize_text_to_tokens(text: str) -> set[str]:
    """
    Simple tokenization + normalization for transcript text.
    Used to estimate how much *new* information a clip adds vs. already selected clips.
    """
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    # Tiny, hand-rolled stopword list to avoid overweighting filler words.
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "so",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "you",
        "i",
        "we",
        "they",
    }
    return {t for t in tokens if t not in stop}


def _pick_clips_simple(clips: list[Clip], target_seconds: float, top_k: int) -> list[Clip]:
    """
    Original behavior: take top-k by score, then sort by time, then accumulate until target.
    Used as a fallback when we don't have transcript text.
    """
    top = sorted(clips, key=lambda c: c.overall_score, reverse=True)[: max(1, top_k)]
    top = sorted(top, key=lambda c: c.start_s)
    picked: list[Clip] = []
    total = 0.0
    for c in top:
        if c.length_s <= 0.5:
            continue
        if total >= target_seconds:
            break
        picked.append(c)
        total += c.length_s
    return picked


def _pick_clips_summarized(clips: list[Clip], target_seconds: float, top_k: int) -> list[Clip]:
    """
    Transcript-aware summarization:
    - Start from highest-scoring clips.
    - Prefer clips whose transcript adds *new* information vs. what we've already selected.
    - Keep selection under target_seconds.

    This tries to keep key narrative moments while cutting redundant / low-content segments.
    """
    if not clips:
        return []

    # Only use summarization logic if we actually have transcript text for at least
    # some clips. Otherwise, fall back to the original behavior.
    has_any_text = any((c.text or "").strip() for c in clips)
    if not has_any_text:
        return _pick_clips_simple(clips, target_seconds=target_seconds, top_k=top_k)

    # Start from best clips by score; allow a wider pool so we can trade off redundancy.
    pool_size = max(top_k * 3, top_k, 1)
    pool = sorted(clips, key=lambda c: c.overall_score, reverse=True)[:pool_size]

    picked: list[Clip] = []
    total = 0.0
    covered_tokens: set[str] = set()

    for c in pool:
        if c.length_s <= 0.5:
            continue
        if total >= target_seconds:
            break

        clip_text = (c.text or "").strip()
        novelty_ok = True
        if clip_text:
            tokens = _normalize_text_to_tokens(clip_text)
            if tokens:
                new_tokens = tokens - covered_tokens
                # Fraction of this clip's tokens that are "new" vs. already covered.
                novelty = len(new_tokens) / float(len(tokens))
                # If we already have some coverage and this clip is mostly redundant,
                # skip it to keep the summary focused.
                if covered_tokens and novelty < 0.25:
                    novelty_ok = False
                else:
                    covered_tokens |= new_tokens

        if not novelty_ok:
            continue

        picked.append(c)
        total += c.length_s

    # If summarization didn't manage to pick anything (e.g., degenerate text),
    # gracefully fall back to the simple scheme.
    if not picked:
        return _pick_clips_simple(clips, target_seconds=target_seconds, top_k=top_k)

    # Always output in chronological order for narrative coherence.
    picked.sort(key=lambda c: c.start_s)
    return picked


def create_montage(
    *,
    report_path: Path,
    out_path: Path,
    target_seconds: float = 120.0,
    top_k: int = 12,
    reencode: bool = True,
    auto_title: bool = True,
) -> None:
    _require_ffmpeg()
    video_path, clips = _load_report(report_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not clips:
        raise RuntimeError("No clips found in report.json")

    # Use transcript-aware summarization when possible; fall back to the original
    # scoring-based picker if we don't have transcript text.
    picked = _pick_clips_summarized(clips, target_seconds=float(target_seconds), top_k=int(top_k))
    if not picked:
        raise RuntimeError("No usable clips selected for montage.")

    # Auto-title output file based on clip captions (best-effort)
    final_out_path = out_path
    if auto_title:
        caps = [c.caption.strip() for c in picked if c.caption and c.caption.strip()]
        if caps:
            # Prefer earliest strong caption (since montage is time-ordered)
            title = caps[0]
            slug = slugify_filename(title, max_len=64)
            if slug:
                final_out_path = out_path.with_name(f"montage_{slug}.mp4")

    # Build FFmpeg filter_complex: trim video+audio per segment, then concat.
    parts: list[str] = []
    labels: list[str] = []
    for i, c in enumerate(picked):
        vlab = f"v{i}"
        alab = f"a{i}"
        # trim uses seconds in timeline; setpts/asetpts reset timestamps
        # Normalize formats so concat is stable across segments:
        # - video: fixed fps + yuv420p
        # - audio: fixed sample rate + stereo
        parts.append(
            f"[0:v]trim=start={c.start_s}:end={c.end_s},setpts=PTS-STARTPTS,fps=60,format=yuv420p[{vlab}]"
        )
        parts.append(
            f"[0:a]atrim=start={c.start_s}:end={c.end_s},asetpts=PTS-STARTPTS,"
            f"aformat=sample_rates=48000:channel_layouts=stereo[{alab}]"
        )
        # IMPORTANT: concat expects inputs interleaved per segment: v0 a0 v1 a1 ...
        labels.append(f"[{vlab}]")
        labels.append(f"[{alab}]")
    n = len(picked)
    parts.append("".join(labels) + f"concat=n={n}:v=1:a=1[v][a]")
    fc = ";".join(parts)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-filter_complex", fc, "-map", "[v]", "-map", "[a]"]
    if reencode:
        # Higher quality default: lower CRF and a slower preset (still reasonable)
        cmd += [
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
        ]
    else:
        cmd += ["-c", "copy"]
    cmd += [str(final_out_path)]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # Re-run once with stderr visible for debugging
        subprocess.check_call(cmd)

