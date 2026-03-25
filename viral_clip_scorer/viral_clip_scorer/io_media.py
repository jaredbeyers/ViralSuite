from __future__ import annotations

import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class AudioData:
    pcm: array  # signed 16-bit mono
    sr: int


def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"{name} not found on PATH. Install FFmpeg (includes ffmpeg + ffprobe) and ensure `{name} -version` works."
        )


def ffprobe_duration(video_path: Path) -> float:
    _require_bin("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    return float(out)


def extract_audio_wav(video_path: Path, wav_path: Path, sr: int = 16000) -> None:
    _require_bin("ffmpeg")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio(video_path: Path, workdir: Path, sr: int = 16000) -> AudioData:
    wav = workdir / "audio.wav"
    if not wav.exists():
        extract_audio_wav(video_path, wav, sr=sr)
    with wave.open(str(wav), "rb") as wf:
        ch = wf.getnchannels()
        sampw = wf.getsampwidth()
        sr2 = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if ch != 1 or sampw != 2:
        # ffmpeg should have produced 1ch 16-bit PCM wav; if not, re-extract safely.
        extract_audio_wav(video_path, wav, sr=sr)
        with wave.open(str(wav), "rb") as wf:
            sr2 = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    pcm = array("h")
    pcm.frombytes(raw)
    return AudioData(pcm=pcm, sr=int(sr2))


def write_audio_segment(audio: AudioData, start_s: float, end_s: float, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    i0 = max(0, int(round(start_s * audio.sr)))
    i1 = min(len(audio.pcm), int(round(end_s * audio.sr)))
    seg = audio.pcm[i0:i1]
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.sr)
        wf.writeframes(seg.tobytes())

