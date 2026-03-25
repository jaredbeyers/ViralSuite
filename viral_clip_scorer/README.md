# Viral Clip Scorer (prototype)

Local CLI tool that **auto-finds clips** inside a video and **scores them for TikTok / YouTube Shorts / Instagram Reels** using practical heuristics (motion, pacing, audio energy, scene changes, “hook-y” phrases, comment triggers, etc).

This is a **working baseline** you can extend into the full “Grammarly for viral videos” product by swapping heuristics for trained models.

## Requirements

- Windows 10+
- Python 3.10+ recommended (works on Python 3.14 too)
- **FFmpeg installed** and available on PATH (`ffmpeg -version`)

## Install

```bash
cd C:\Users\jared\Documents\viral_clip_scorer
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional (auto-transcription):

```bash
pip install -r requirements-optional.txt
```

## Usage

Score a video (auto-transcribe if optional whisper is installed; otherwise you can pass `--transcript`):

```bash
python -m viral_clip_scorer score --video "C:\path\to\input.mp4" --outdir "C:\path\to\out"
```

If you already have a transcript text file:

```bash
python -m viral_clip_scorer score --video "C:\path\to\input.mp4" --transcript "C:\path\to\transcript.txt" --outdir "C:\path\to\out"
```

Export top clips:

```bash
python -m viral_clip_scorer score --video "C:\path\to\input.mp4" --outdir "C:\path\to\out" --export-top 5
```

## Optional: Replace visuals with B-roll (keep original audio)

Put background videos into `viral_clip_scorer\broll_videos\` (or any folder you want), then run:

```bash
python -m viral_clip_scorer score --video "C:\path\to\input.mp4" --outdir "C:\path\to\out" --broll-dir "C:\Users\jared\Documents\ViralSuite\broll_videos" --broll-mode auto
```

Modes:
- `never` (default): keep original video
- `always`: always replace visuals with b-roll
- `auto`: replace visuals only if input is audio-only or very low-dynamics

## Add a local video into the B-roll folder (“upload”)

Copy a local video into the b-roll library:

```bash
python -m viral_clip_scorer broll add --video "C:\path\to\broll.mp4" --label "minecraft"
```

Optionally trim/cap it:

```bash
python -m viral_clip_scorer broll add --video "C:\path\to\broll.mp4" --label "gta" --trim-start 7 --max-seconds 180
```

## Output

The tool writes:

- `report.json`: ranked clips with per-feature scores and per-platform scores
- `clips/clip_001.mp4` …: exported top clips (if `--export-top` used)

## What’s implemented (baseline)

- Hook strength (first ~2.5s): audio spike + motion burst + hook phrases
- Emotional intensity (proxy): audio excitement + cut density (face detection is not enabled in this minimal-deps build)
- Retention prediction (proxy): pacing consistency + pattern interrupts + scene dynamics
- Speech highlight detection: scores sentences for story/controversy/questions/curiosity
- Scene change / visual dynamics: FFmpeg scene-cut detection (and cut density)
- Reaction detection (proxy): laughter tokens in transcript + high-energy audio bursts
- Viral topic matching: keyword trends list; optional embeddings
- Caption/quote potential: selects best one-liner
- Face engagement: placeholder (0 in minimal build; can be upgraded with OpenCV later)
- Pacing optimization: words/sec + silence ratio
- Pattern interrupt: sudden changes in audio energy or visual motion/cuts
- Sound intensity curve: RMS curve, peakiness, build/resolve
- Loop potential: similarity between first/last second (audio + frame)
- Comment trigger: “hot take” language, controversial phrasing, questions
- Multi-platform scoring: platform-specific weighting
- Auto-clip generation: candidate clip boundaries from cuts + audio peaks + highlight timestamps
- Hook rewriter + title generator: simple, editable templates (no external API)

## Notes

- This is a **heuristic MVP**. It’s designed so you can later replace `features/*.py` with ML models.
- For best results, install optional dependencies and use videos with clear speech.

