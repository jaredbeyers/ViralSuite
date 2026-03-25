from __future__ import annotations

from pathlib import Path

from viral_clip_scorer.models import Sentence, Transcript
from viral_clip_scorer.utils import split_sentences

import re
import subprocess
import sys


def load_transcript_txt(path: Path) -> Transcript:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # No timestamps in plain txt; keep sentences for caption scoring
    sents = split_sentences(text)
    return Transcript(text=text.strip(), sentences=[Sentence(text=s) for s in sents])


def try_faster_whisper_transcribe(audio_wav: Path) -> Transcript | None:
    """
    Optional dependency path.
    Returns Transcript with no reliable timestamps (fast baseline).
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_wav), vad_filter=True)
    parts: list[str] = []
    for seg in segments:
        if seg.text:
            parts.append(seg.text.strip())
    text = " ".join(parts).strip()
    if not text:
        return Transcript(text="", sentences=[])
    sents = split_sentences(text)
    return Transcript(text=text, sentences=[Sentence(text=s) for s in sents])


_VTT_TS_LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d)\.(\d\d\d)")
_VTT_CUE_SETTINGS = re.compile(r"\b(align|position|size|line):[^\s]+")


def _ts_to_s(hh: str, mm: str, ss: str, ms: str) -> float:
    return int(hh) * 3600.0 + int(mm) * 60.0 + int(ss) + int(ms) / 1000.0


def parse_vtt_to_sentences(vtt: str) -> list[Sentence]:
    """
    Parse WebVTT into timestamped sentences (cue blocks).
    Best-effort; ignores styling/settings.
    """
    out: list[Sentence] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_lines
        if cur_start is None or cur_end is None:
            cur_lines = []
            return
        txt = " ".join(cur_lines).strip()
        txt = re.sub(r"<[^>]+>", "", txt).strip()
        txt = _VTT_CUE_SETTINGS.sub(" ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            out.append(Sentence(text=txt, start_s=float(cur_start), end_s=float(cur_end)))
        cur_start = None
        cur_end = None
        cur_lines = []

    for raw in (vtt or "").splitlines():
        ln = raw.strip()
        if not ln:
            flush()
            continue
        if ln == "WEBVTT" or ln.startswith(("NOTE", "STYLE", "REGION")):
            continue
        m = _VTT_TS_LINE.match(ln)
        if m:
            flush()
            cur_start = _ts_to_s(m.group(1), m.group(2), m.group(3), m.group(4))
            cur_end = _ts_to_s(m.group(5), m.group(6), m.group(7), m.group(8))
            continue
        # Skip cue numbers
        if ln.isdigit() and cur_start is None:
            continue
        cur_lines.append(ln)
    flush()
    return out


def _vtt_to_text(vtt: str) -> str:
    sents = parse_vtt_to_sentences(vtt)
    # de-dupe consecutive repeats
    out: list[str] = []
    prev = None
    for s in sents:
        t = s.text.strip()
        if t and t != prev:
            out.append(t)
        prev = t
    return " ".join(out).strip()


def try_ytdlp_captions(url: str, workdir: Path, lang: str = "en") -> Transcript | None:
    """
    Fetch captions/subtitles via yt-dlp (best effort).
    Works well for YouTube URLs (manual subs or auto-subs).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str((workdir / "captions.%(ext)s").resolve())

    # Prefer embedding-free approach: ask yt-dlp to dump automatic subs if needed.
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-lang",
        lang,
        "--sub-format",
        "vtt",
        "--output",
        outtmpl,
        "--print-json",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None

    if proc.returncode != 0:
        # Helpful debug artifact for troubleshooting (safe to ignore)
        try:
            (workdir / "captions_ytdlp_error.txt").write_text(
                f"cmd: {cmd}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n",
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            pass
        return None

    # Find the newest .vtt in workdir (yt-dlp may add language codes)
    vtts = sorted(workdir.glob("captions*.vtt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vtts:
        # Sometimes output includes title/id; fallback: any vtt file
        vtts = sorted(workdir.glob("*.vtt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vtts:
        return None

    vtt_text = vtts[0].read_text(encoding="utf-8", errors="ignore")
    cue_sents = parse_vtt_to_sentences(vtt_text)
    text = " ".join(s.text for s in cue_sents).strip()
    if not text:
        return Transcript(text="", sentences=[])
    return Transcript(text=text, sentences=cue_sents)


def text_for_timerange(tr: Transcript, start_s: float, end_s: float) -> str:
    """
    Extract transcript text overlapping [start_s, end_s].
    If no timestamps exist, returns the full transcript text.
    """
    if not tr.sentences:
        return tr.text or ""
    has_ts = any(s.start_s is not None and s.end_s is not None for s in tr.sentences)
    if not has_ts:
        return tr.text or ""
    parts: list[str] = []
    for s in tr.sentences:
        if s.start_s is None or s.end_s is None:
            continue
        if s.end_s < start_s:
            continue
        if s.start_s > end_s:
            continue
        parts.append(s.text)
    return " ".join(parts).strip()

