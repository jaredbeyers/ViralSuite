from __future__ import annotations

import random
from dataclasses import dataclass

from viral_clip_scorer.features_audio import AudioSignals
from viral_clip_scorer.features_visual import VisualSignals
from viral_clip_scorer.models import ClipCandidate
from viral_clip_scorer.utils import clamp01, quantile


@dataclass(frozen=True)
class CandidateConfig:
    min_clip_s: float = 12.0
    max_clip_s: float = 70.0
    target_clips: int = 12
    seed: int = 7


def _top_peaks(times: list[float], values: list[float], k: int, min_sep_s: float) -> list[float]:
    if not times or not values:
        return []
    idxs = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    picks: list[float] = []
    for i in idxs:
        t = float(times[int(i)])
        if all(abs(t - p) >= min_sep_s for p in picks):
            picks.append(t)
        if len(picks) >= k:
            break
    return sorted(picks)


def propose_candidates(duration_s: float, a: AudioSignals, v: VisualSignals, cfg: CandidateConfig) -> list[ClipCandidate]:
    rng = random.Random(cfg.seed)

    # Peak times from audio and cuts/motion
    audio_peaks = _top_peaks(a.times_s, a.rms, k=max(8, cfg.target_clips), min_sep_s=3.5)
    cut_peaks = _top_peaks(v.times_s, v.cut_strength, k=max(8, cfg.target_clips), min_sep_s=4.0)
    motion_peaks = _top_peaks(v.times_s, v.motion, k=max(8, cfg.target_clips), min_sep_s=4.0)

    anchor_times = sorted(set(audio_peaks + cut_peaks + motion_peaks))
    if not anchor_times:
        # fallback: evenly spaced anchors
        step = max(cfg.min_clip_s, duration_s / max(1, cfg.target_clips))
        anchor_times = [min(duration_s - 1.0, i * step) for i in range(cfg.target_clips)]

    # Candidate lengths: prefer 20-55s
    len_choices = [15, 20, 25, 30, 35, 42, 50, 60]

    cands: list[ClipCandidate] = []
    for t in anchor_times:
        L = float(rng.choice(len_choices))
        L = max(cfg.min_clip_s, min(cfg.max_clip_s, L))
        start = max(0.0, t - 3.0)  # begin slightly before the peak
        end = start + L
        if end > duration_s:
            end = duration_s
            start = max(0.0, end - L)
        if end - start < cfg.min_clip_s * 0.95:
            continue
        cands.append(ClipCandidate(start_s=float(start), end_s=float(end), reason="anchor_peak"))

    # Add some random windows for diversity (helps when peaks are noisy)
    for _ in range(max(3, cfg.target_clips // 3)):
        L = float(rng.choice(len_choices))
        L = max(cfg.min_clip_s, min(cfg.max_clip_s, L))
        start = rng.random() * max(0.0, duration_s - L)
        cands.append(ClipCandidate(start_s=float(start), end_s=float(start + L), reason="random_window"))

    # Deduplicate roughly by overlap
    cands = sorted(cands, key=lambda c: (c.start_s, c.end_s))
    dedup: list[ClipCandidate] = []
    for c in cands:
        if not dedup:
            dedup.append(c)
            continue
        prev = dedup[-1]
        overlap = max(0.0, min(prev.end_s, c.end_s) - max(prev.start_s, c.start_s))
        shorter = min(prev.end_s - prev.start_s, c.end_s - c.start_s)
        if shorter > 0 and overlap / shorter > 0.75:
            continue
        dedup.append(c)

    # Keep a bit more than target, scoring stage will rank
    return dedup[: max(cfg.target_clips * 3, cfg.target_clips)]

