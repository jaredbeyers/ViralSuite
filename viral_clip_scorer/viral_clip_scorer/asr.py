from __future__ import annotations

from pathlib import Path

from viral_clip_scorer.models import Transcript
from viral_clip_scorer.utils import normalize_whitespace, split_sentences


def load_transcript_txt(path: Path) -> Transcript:
    text = normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
    sents = split_sentences(text)
    return Transcript(text=text, sentences=[{"text": s} for s in sents])


def try_transcribe_wav(wav_path: Path) -> Transcript | None:
    """
    Optional ASR using faster-whisper (if installed).
    Returns None if dependency is unavailable or transcription fails.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None

    try:
        # Small model is a good default for speed; users can swap later.
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(wav_path), vad_filter=True)
        texts: list[str] = []
        sentences = []
        for seg in segments:
            t = normalize_whitespace(seg.text)
            if not t:
                continue
            texts.append(t)
            # store as sentence-like chunks with timestamps
            sentences.append({"start_s": float(seg.start), "end_s": float(seg.end), "text": t})
        full = normalize_whitespace(" ".join(texts))
        return Transcript(text=full, sentences=sentences)
    except Exception:
        return None

