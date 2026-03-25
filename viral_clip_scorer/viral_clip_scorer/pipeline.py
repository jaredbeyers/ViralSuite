from __future__ import annotations

from pathlib import Path

from viral_clip_scorer.candidates import CandidateConfig, propose_candidates
from viral_clip_scorer.exporter import export_clip_ffmpeg
from viral_clip_scorer.features_audio import compute_audio_signals
from viral_clip_scorer.features_visual import compute_visual_signals
from viral_clip_scorer.io_media import ffprobe_duration, load_audio
from viral_clip_scorer.models import ClipResult, Report
from viral_clip_scorer.scoring import Weights, overall_from_features, score_clip
from viral_clip_scorer.transcript import (
    load_transcript_txt,
    text_for_timerange,
    try_faster_whisper_transcribe,
    try_ytdlp_captions,
)
from viral_clip_scorer.utils import load_lines, write_json
from viral_clip_scorer.utils import slugify_filename


def run_pipeline(
    *,
    video_path: Path,
    outdir: Path,
    transcript_path: Path | None,
    source_url: str | None,
    export_top: int,
    min_clip_s: float,
    max_clip_s: float,
    target_clips: int,
    seed: int,
    trends_path: Path | None,
    allow_asr: bool,
    broll_dir: Path | None,
    broll_mode: str,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = outdir / "_work"
    workdir.mkdir(parents=True, exist_ok=True)

    # Optionally replace visuals (keep audio) for audio-only or low-dynamics videos.
    from viral_clip_scorer.broll import maybe_replace_visuals

    video_path_for_analysis = maybe_replace_visuals(
        video_path=video_path,
        workdir=workdir,
        broll_dir=broll_dir,
        mode=broll_mode,
    )

    duration_s = ffprobe_duration(video_path_for_analysis)

    # Trends
    if trends_path and trends_path.exists():
        trends = load_lines(trends_path)
    else:
        default_trends = Path(__file__).resolve().parent / "data" / "trends_default.txt"
        trends = load_lines(default_trends) if default_trends.exists() else []

    # Audio + visual signals
    audio = load_audio(video_path_for_analysis, workdir=workdir, sr=16000)
    a_sig = compute_audio_signals(audio, hop_s=0.05)
    v_sig = compute_visual_signals(video_path_for_analysis, sample_fps=6.0, max_seconds=None)

    # Transcript (optional)
    transcript_used = False
    tr = None

    if transcript_path and transcript_path.exists():
        tr = load_transcript_txt(transcript_path)
        transcript_used = True
    elif source_url:
        tr3 = try_ytdlp_captions(source_url, workdir=workdir)
        if tr3 is not None and tr3.text.strip():
            tr = tr3
            transcript_used = True
    elif allow_asr:
        wav = workdir / "audio.wav"
        tr2 = try_faster_whisper_transcribe(wav)
        if tr2 is not None:
            tr = tr2
            transcript_used = True

    # Candidates
    cfg = CandidateConfig(min_clip_s=min_clip_s, max_clip_s=max_clip_s, target_clips=target_clips, seed=seed)
    candidates = propose_candidates(duration_s=duration_s, a=a_sig, v=v_sig, cfg=cfg)

    # Score
    weights = Weights()
    clip_results: list[ClipResult] = []
    for c in candidates:
        clip_text = text_for_timerange(tr, c.start_s, c.end_s) if tr is not None else ""
        features, plat, debug = score_clip(
            a_sig=a_sig,
            v_sig=v_sig,
            clip_text=clip_text,
            trends=trends,
            start_s=c.start_s,
            end_s=c.end_s,
        )
        overall = overall_from_features(features, weights)
        best_caption = debug.get("caption")
        hook_rewrites = debug.get("hook_rewrites") or []
        titles = debug.get("titles") or []
        clip_results.append(
            ClipResult(
                rank=0,
                start_s=float(c.start_s),
                end_s=float(c.end_s),
                length_s=float(c.end_s - c.start_s),
                overall_score=float(100.0 * overall),
                platform_scores=plat,
                features=features,
                best_caption=best_caption,
                hook_rewrite_suggestions=list(hook_rewrites)[:4],
                title_suggestions=list(titles)[:3],
                debug={
                    "reason": c.reason,
                    "clip_text_preview": clip_text[:280],
                    **{k: v for k, v in debug.items() if k not in ("hook_rewrites", "titles")},
                },
            )
        )

    clip_results.sort(key=lambda r: (r.overall_score, r.platform_scores.tiktok, r.platform_scores.youtube_shorts), reverse=True)
    for i, r in enumerate(clip_results, start=1):
        r.rank = i

    report = Report(
        video_path=str(video_path_for_analysis),
        duration_s=float(duration_s),
        transcript_used=bool(transcript_used),
        candidates_considered=int(len(candidates)),
        clips=clip_results[: max(target_clips, 1)],
    )

    # Write report
    report_path = outdir / "report.json"
    write_json(report_path, report.to_dict())

    # Export
    if export_top and export_top > 0:
        clips_dir = outdir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        for r in report.clips[: int(export_top)]:
            cap = (r.best_caption or "").strip()
            slug = slugify_filename(cap, max_len=60) if cap else f"clip-{r.rank:03d}"
            out_path = clips_dir / f"clip_{r.rank:03d}_score_{int(round(r.overall_score)):02d}_{slug}.mp4"
            export_clip_ffmpeg(video_path_for_analysis, r.start_s, r.end_s, out_path)

