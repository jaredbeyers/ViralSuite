from __future__ import annotations

from dataclasses import dataclass

from viral_clip_scorer.io_media import AudioData
from viral_clip_scorer.utils import clamp01, moving_avg, quantile, safe_div, softstep


@dataclass(frozen=True)
class AudioSignals:
    hop_s: float
    times_s: list[float]
    rms: list[float]  # root-mean-square energy


def compute_audio_signals(audio: AudioData, hop_s: float = 0.05) -> AudioSignals:
    hop = max(1, int(round(hop_s * audio.sr)))
    win = max(hop, int(round(0.2 * audio.sr)))
    y = audio.pcm

    rms: list[float] = []
    times: list[float] = []
    for i in range(0, max(1, len(y) - win), hop):
        seg = y[i : i + win]
        if len(seg) == 0:
            r = 0.0
        else:
            # seg is array('h'); compute RMS in float
            s2 = 0.0
            for v in seg:
                fv = float(v) / 32768.0
                s2 += fv * fv
            r = (s2 / float(len(seg))) ** 0.5
        rms.append(r)
        times.append(i / float(audio.sr))

    # Smooth a bit
    rms_sm = moving_avg(rms, win=3)
    return AudioSignals(hop_s=float(hop_s), times_s=times, rms=rms_sm)


def _window(a: AudioSignals, start_s: float, end_s: float) -> list[float]:
    if not a.times_s:
        return []
    return [a.rms[i] for i, t in enumerate(a.times_s) if start_s <= t <= end_s]


def segment_audio_scores(a: AudioSignals, start_s: float, end_s: float) -> dict[str, float]:
    xs = _window(a, start_s, end_s)
    if not xs:
        return {"sound_curve": 0.0, "audio_peakiness": 0.0, "silence_ratio": 1.0}

    mean = (sum(xs) / float(len(xs))) if xs else 0.0
    q90 = quantile(xs, 0.9)
    q10 = quantile(xs, 0.1)
    peakiness = safe_div(q90 - mean, mean + 1e-6, default=0.0)

    # Silence ratio: fraction below small threshold (adaptive by q10/q90)
    thr = max(0.003, 0.4 * q10 + 0.1 * q90)
    silence = sum(1 for x in xs if x <= thr) / float(len(xs))

    # Curve quality: build-up then resolution (simple 3-part shape)
    n = len(xs)
    p1 = xs[: max(1, n // 3)]
    p2 = xs[max(1, n // 3) : max(2, 2 * n // 3)]
    p3 = xs[max(2, 2 * n // 3) :]
    a1 = (sum(p1) / float(len(p1))) if p1 else 0.0
    a2 = (sum(p2) / float(len(p2))) if p2 else 0.0
    a3 = (sum(p3) / float(len(p3))) if p3 else 0.0
    build = clamp01(softstep(a2 - a1, 0.001, 0.02))
    resolve = clamp01(softstep(a2 - a3, 0.001, 0.02))
    curve = clamp01(0.5 * build + 0.5 * resolve)

    # Normalize mean loudness and peakiness lightly
    loud = softstep(mean, 0.005, 0.05)
    peak = clamp01(softstep(peakiness, 0.05, 0.8))
    sound_curve = clamp01(0.45 * curve + 0.35 * peak + 0.2 * loud)

    return {
        "sound_curve": float(sound_curve),
        "audio_peakiness": float(peak),
        "silence_ratio": float(clamp01(silence)),
    }


def hook_audio_spike(a: AudioSignals, start_s: float, hook_len_s: float = 2.5) -> float:
    xs = _window(a, start_s, start_s + hook_len_s)
    if len(xs) < 3:
        return 0.0
    q90 = quantile(xs, 0.9)
    q20 = quantile(xs, 0.2)
    return clamp01(softstep(q90 - q20, 0.001, 0.03))


def loop_similarity_audio(a: AudioSignals, start_s: float, end_s: float, edge_s: float = 1.0) -> float:
    xs1 = _window(a, start_s, start_s + edge_s)
    xs2 = _window(a, end_s - edge_s, end_s)
    if len(xs1) < 5 or len(xs2) < 5:
        return 0.0
    n = min(len(xs1), len(xs2))
    v1 = xs1[:n]
    v2 = xs2[:n]
    m1 = sum(v1) / float(n)
    m2 = sum(v2) / float(n)
    num = 0.0
    d1 = 0.0
    d2 = 0.0
    for i in range(n):
        a1 = v1[i] - m1
        a2 = v2[i] - m2
        num += a1 * a2
        d1 += a1 * a1
        d2 += a2 * a2
    denom = (d1**0.5) * (d2**0.5) + 1e-6
    corr = num / denom
    # map [-1,1] -> [0,1], but we only care about positive similarity
    return clamp01((corr + 1.0) / 2.0)

