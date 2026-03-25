from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from viral_clip_scorer.utils import clamp01, safe_div, soft_clip_score


@dataclass(frozen=True)
class AudioAnalysis:
    sr: int
    hop_length: int
    times_s: np.ndarray  # [T]
    rms: np.ndarray  # [T]

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1]) if self.times_s.size else 0.0


def analyze_audio(wav_path: str, sr: int = 16000, hop_length: int = 512) -> AudioAnalysis:
    y, sr2 = librosa.load(wav_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr2, hop_length=hop_length)
    rms = rms.astype(np.float32)
    times = times.astype(np.float32)
    return AudioAnalysis(sr=sr2, hop_length=hop_length, times_s=times, rms=rms)


def slice_curve(times_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    if times_s.size == 0:
        return values[:0]
    m = (times_s >= start_s) & (times_s <= end_s)
    return values[m]


def sound_intensity_score(audio: AudioAnalysis, start_s: float, end_s: float) -> float:
    seg = slice_curve(audio.times_s, audio.rms, start_s, end_s)
    if seg.size < 4:
        return 0.0
    # peakiness: high max vs median
    med = float(np.median(seg))
    mx = float(np.max(seg))
    peak = safe_div(mx - med, mx + 1e-6, 0.0)
    return clamp01(soft_clip_score(peak, 0.10, 0.55))


def sound_curve_shape_score(audio: AudioAnalysis, start_s: float, end_s: float) -> float:
    seg = slice_curve(audio.times_s, audio.rms, start_s, end_s)
    if seg.size < 8:
        return 0.0
    n = seg.size
    a = seg[: n // 3].mean()
    b = seg[n // 3 : 2 * n // 3].mean()
    c = seg[2 * n // 3 :].mean()
    # reward "build then resolve" where middle is highest
    build = (b > a) and (b > c)
    strength = float((b - (a + c) / 2.0) / (b + 1e-6))
    return clamp01((0.35 if build else 0.0) + soft_clip_score(strength, 0.05, 0.35) * 0.65)


def silence_ratio(audio: AudioAnalysis, start_s: float, end_s: float) -> float:
    seg = slice_curve(audio.times_s, audio.rms, start_s, end_s)
    if seg.size == 0:
        return 1.0
    thr = float(np.percentile(audio.rms, 20)) if audio.rms.size else 0.0
    return float(np.mean(seg <= thr))


def audio_spike_first_seconds(audio: AudioAnalysis, start_s: float, seconds: float = 2.5) -> float:
    seg = slice_curve(audio.times_s, audio.rms, start_s, start_s + seconds)
    if seg.size < 3:
        return 0.0
    mx = float(np.max(seg))
    base = float(np.percentile(audio.rms, 40)) if audio.rms.size else 1e-6
    rel = safe_div(mx, base + 1e-6, 0.0)
    return clamp01(soft_clip_score(rel, 1.3, 3.2))

