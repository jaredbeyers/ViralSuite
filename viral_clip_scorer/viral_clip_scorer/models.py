from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Sentence:
    text: str
    start_s: float | None = None
    end_s: float | None = None


@dataclass(slots=True)
class Transcript:
    text: str
    sentences: list[Sentence] = field(default_factory=list)


@dataclass(slots=True)
class ClipCandidate:
    start_s: float
    end_s: float
    reason: str


@dataclass(slots=True)
class ClipFeatureScores:
    hook_strength: float
    emotional_intensity: float
    retention_pred: float
    speech_highlight: float
    visual_dynamics: float
    reaction_laughter: float
    viral_topic_match: float
    caption_potential: float
    face_engagement: float
    pacing: float
    pattern_interrupt: float
    sound_curve: float
    loop_potential: float
    comment_trigger: float


@dataclass(slots=True)
class PlatformScores:
    tiktok: float
    youtube_shorts: float
    instagram_reels: float


@dataclass(slots=True)
class ClipResult:
    rank: int
    start_s: float
    end_s: float
    length_s: float
    overall_score: float
    platform_scores: PlatformScores
    features: ClipFeatureScores
    best_caption: str | None = None
    hook_rewrite_suggestions: list[str] = field(default_factory=list)
    title_suggestions: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Report:
    video_path: str
    duration_s: float
    transcript_used: bool
    candidates_considered: int
    clips: list[ClipResult]
    scoring_version: Literal["0.1.0"] = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

