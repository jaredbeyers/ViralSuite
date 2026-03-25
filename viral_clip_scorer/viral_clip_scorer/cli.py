import argparse
from pathlib import Path

from viral_clip_scorer.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="viral_clip_scorer", description="Auto-generate and score viral clips.")
    sub = p.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score", help="Score a video and optionally export top clips.")
    score.add_argument("--video", required=True, help="Path to input video (mp4/mov/etc).")
    score.add_argument("--outdir", required=True, help="Directory to write report + exported clips.")
    score.add_argument("--transcript", default=None, help="Optional path to transcript .txt (skip ASR).")
    score.add_argument(
        "--source-url",
        default=None,
        help="Optional original URL (YouTube/etc). Used to fetch captions via yt-dlp when no transcript file is provided.",
    )
    score.add_argument("--export-top", type=int, default=0, help="Export top N clips as mp4 into outdir/clips.")
    score.add_argument(
        "--broll-dir",
        default=None,
        help="Optional folder of background videos to overlay when replacing uninteresting/absent visuals.",
    )
    score.add_argument(
        "--broll-mode",
        choices=["auto", "always", "never"],
        default="never",
        help="B-roll behavior: auto (only if audio-only/low-dynamics), always (force), never (default).",
    )
    score.add_argument("--min-clip", type=float, default=12.0, help="Minimum candidate clip length (seconds).")
    score.add_argument("--max-clip", type=float, default=70.0, help="Maximum candidate clip length (seconds).")
    score.add_argument("--target-clips", type=int, default=12, help="How many candidate clips to propose before ranking.")
    score.add_argument("--seed", type=int, default=7, help="Deterministic seed for tie-breaks.")
    score.add_argument("--trends", default=None, help="Optional path to a trends keywords file (one per line).")
    score.add_argument("--no-asr", action="store_true", help="Do not attempt auto-transcription even if available.")

    broll = sub.add_parser("broll", help="Manage the b-roll library.")
    bsub = broll.add_subparsers(dest="broll_cmd", required=True)
    add = bsub.add_parser("add", help="Add a local video file into the b-roll folder.")
    add.add_argument("--video", required=True, help="Path to a local video file to add to b-roll.")
    add.add_argument(
        "--broll-dir",
        default=None,
        help="Destination b-roll folder (default: <project>/broll_videos).",
    )
    add.add_argument("--label", default=None, help="Optional label used in the destination filename.")
    add.add_argument("--trim-start", type=float, default=0.0, help="Trim N seconds from the start.")
    add.add_argument("--max-seconds", type=float, default=0.0, help="Limit output length (seconds). 0 = no limit.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "score":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        run_pipeline(
            video_path=Path(args.video),
            outdir=outdir,
            transcript_path=Path(args.transcript) if args.transcript else None,
            source_url=str(args.source_url) if args.source_url else None,
            export_top=args.export_top,
            min_clip_s=float(args.min_clip),
            max_clip_s=float(args.max_clip),
            target_clips=int(args.target_clips),
            seed=int(args.seed),
            trends_path=Path(args.trends) if args.trends else None,
            allow_asr=not bool(args.no_asr),
            broll_dir=Path(args.broll_dir) if args.broll_dir else None,
            broll_mode=str(args.broll_mode),
        )
        return 0

    if args.cmd == "broll" and args.broll_cmd == "add":
        from viral_clip_scorer.broll_manage import add_broll_video

        dest = add_broll_video(
            video_path=Path(args.video),
            broll_dir=Path(args.broll_dir) if args.broll_dir else None,
            label=str(args.label) if args.label else None,
            trim_start_s=float(args.trim_start),
            max_seconds=float(args.max_seconds),
        )
        print(str(dest))
        return 0

    parser.error(f"Unknown command: {args.cmd}")
    return 2

