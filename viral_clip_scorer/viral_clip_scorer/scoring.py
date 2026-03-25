from __future__ import annotations

from dataclasses import dataclass

from viral_clip_scorer.features_audio import AudioSignals, hook_audio_spike, loop_similarity_audio, segment_audio_scores
from viral_clip_scorer.features_text import TextScores, hook_rewrites, score_text_block, title_suggestions
from viral_clip_scorer.features_visual import VisualSignals, segment_visual_scores
from viral_clip_scorer.models import ClipFeatureScores, PlatformScores
from viral_clip_scorer.utils import clamp01, safe_div, softstep


@dataclass(frozen=True)
class Weights:
    hook_strength: float = 1.25
    emotional_intensity: float = 1.00
    retention_pred: float = 1.15
    speech_highlight: float = 1.00
    visual_dynamics: float = 0.95
    reaction_laughter: float = 0.55
    viral_topic_match: float = 0.55
    caption_potential: float = 0.80
    face_engagement: float = 0.65
    pacing: float = 0.75
    pattern_interrupt: float = 0.85
    sound_curve: float = 0.75
    loop_potential: float = 0.55
    comment_trigger: float = 0.70


def compute_pacing_score(text: str, length_s: float, silence_ratio: float) -> float:
    # words per second and silence balance
    words = len((text or "").split())
    wps = safe_div(float(words), max(1e-6, float(length_s)), default=0.0)
    wps_score = 1.0 - abs(clamp01((wps - 2.6) / 3.0) - 0.0)  # prefer ~2.6 wps
    sil_score = 1.0 - clamp01(softstep(silence_ratio, 0.22, 0.55))  # too much silence hurts
    return clamp01(0.6 * clamp01(wps_score) + 0.4 * sil_score)


def compute_pattern_interrupt(audio_peakiness: float, scene_changes: float, visual_dynamics: float) -> float:
    return clamp01(0.45 * audio_peakiness + 0.35 * scene_changes + 0.20 * visual_dynamics)


def compute_emotional_intensity(face_engagement: float, visual_dynamics: float, sound_curve: float) -> float:
    # Proxy: expressive visuals + energetic audio, plus face engagement
    return clamp01(0.45 * sound_curve + 0.30 * visual_dynamics + 0.25 * face_engagement)


def compute_retention_pred(
    hook_strength: float,
    pacing: float,
    visual_dynamics: float,
    pattern_interrupt: float,
    loop_potential: float,
) -> float:
    # Proxy retention: strong hook, good pacing, dynamics, interrupts, and loop
    return clamp01(0.30 * hook_strength + 0.22 * pacing + 0.18 * visual_dynamics + 0.20 * pattern_interrupt + 0.10 * loop_potential)


def overall_from_features(f: ClipFeatureScores, w: Weights) -> float:
    num = (
        f.hook_strength * w.hook_strength
        + f.emotional_intensity * w.emotional_intensity
        + f.retention_pred * w.retention_pred
        + f.speech_highlight * w.speech_highlight
        + f.visual_dynamics * w.visual_dynamics
        + f.reaction_laughter * w.reaction_laughter
        + f.viral_topic_match * w.viral_topic_match
        + f.caption_potential * w.caption_potential
        + f.face_engagement * w.face_engagement
        + f.pacing * w.pacing
        + f.pattern_interrupt * w.pattern_interrupt
        + f.sound_curve * w.sound_curve
        + f.loop_potential * w.loop_potential
        + f.comment_trigger * w.comment_trigger
    )
    den = (
        w.hook_strength
        + w.emotional_intensity
        + w.retention_pred
        + w.speech_highlight
        + w.visual_dynamics
        + w.reaction_laughter
        + w.viral_topic_match
        + w.caption_potential
        + w.face_engagement
        + w.pacing
        + w.pattern_interrupt
        + w.sound_curve
        + w.loop_potential
        + w.comment_trigger
    )
    return clamp01(safe_div(num, den, default=0.0))


def platform_scores(f: ClipFeatureScores) -> PlatformScores:
    # Platform-specific weights (simple baseline)
    tiktok = clamp01(
        0.26 * f.hook_strength
        + 0.18 * f.emotional_intensity
        + 0.18 * f.retention_pred
        + 0.12 * f.pattern_interrupt
        + 0.10 * f.visual_dynamics
        + 0.08 * f.comment_trigger
        + 0.08 * f.loop_potential
    )
    shorts = clamp01(
        0.22 * f.retention_pred
        + 0.18 * f.speech_highlight
        + 0.16 * f.hook_strength
        + 0.14 * f.pacing
        + 0.10 * f.sound_curve
        + 0.10 * f.visual_dynamics
        + 0.10 * f.caption_potential
    )
    reels = clamp01(
        0.20 * f.visual_dynamics
        + 0.18 * f.face_engagement
        + 0.18 * f.sound_curve
        + 0.16 * f.hook_strength
        + 0.12 * f.caption_potential
        + 0.10 * f.loop_potential
        + 0.06 * f.emotional_intensity
    )
    return PlatformScores(
        tiktok=float(100.0 * tiktok),
        youtube_shorts=float(100.0 * shorts),
        instagram_reels=float(100.0 * reels),
    )


def score_clip(
    *,
    a_sig: AudioSignals,
    v_sig: VisualSignals,
    clip_text: str,
    trends: list[str],
    start_s: float,
    end_s: float,
) -> tuple[ClipFeatureScores, PlatformScores, dict]:
    length_s = float(end_s - start_s)

    # Visual + audio segment stats
    vis = segment_visual_scores(v_sig, start_s, end_s)
    aud = segment_audio_scores(a_sig, start_s, end_s)

    # Text stats (no timestamps yet: best-effort by using full transcript)
    txt: TextScores = score_text_block(clip_text, trends=trends)

    # Hook: audio spike + early motion + hook phrases (first 2.5s)
    hook_audio = hook_audio_spike(a_sig, start_s, hook_len_s=2.5)
    hook_text = txt.hook_text
    hook_motion = softstep(vis["visual_dynamics"], 0.25, 0.8)
    hook_strength = clamp01(0.45 * hook_audio + 0.35 * hook_text + 0.20 * hook_motion)

    loop = loop_similarity_audio(a_sig, start_s, end_s, edge_s=1.0)
    pacing = compute_pacing_score(clip_text, length_s=length_s, silence_ratio=aud["silence_ratio"])
    pattern = compute_pattern_interrupt(aud["audio_peakiness"], vis["scene_changes"], vis["visual_dynamics"])
    emotion = compute_emotional_intensity(vis["face_engagement"], vis["visual_dynamics"], aud["sound_curve"])
    retention = compute_retention_pred(hook_strength, pacing, vis["visual_dynamics"], pattern, loop)

    features = ClipFeatureScores(
        hook_strength=float(hook_strength),
        emotional_intensity=float(emotion),
        retention_pred=float(retention),
        speech_highlight=float(txt.speech_highlight),
        visual_dynamics=float(vis["visual_dynamics"]),
        reaction_laughter=float(txt.reaction_laughter),
        viral_topic_match=float(txt.viral_topic_match),
        caption_potential=float(txt.caption_score),
        face_engagement=float(vis["face_engagement"]),
        pacing=float(pacing),
        pattern_interrupt=float(pattern),
        sound_curve=float(aud["sound_curve"]),
        loop_potential=float(loop),
        comment_trigger=float(txt.comment_trigger),
    )
    plat = platform_scores(features)

    debug = {
        "hook_audio": hook_audio,
        "hook_text": hook_text,
        "hook_motion": hook_motion,
        "silence_ratio": aud["silence_ratio"],
        "audio_peakiness": aud["audio_peakiness"],
        "scene_changes": vis["scene_changes"],
        "caption": txt.caption,
        "hook_rewrites": hook_rewrites((clip_text or "")[:160]),
        "titles": title_suggestions(txt.caption),
    }
    return features, plat, debug

