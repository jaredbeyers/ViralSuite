import argparse
from pathlib import Path

from video_downloader.downloader import download_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="video_downloader", description="Download videos from URLs using yt-dlp.")
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download a video URL.")
    dl.add_argument("--url", required=True, help="Video URL (YouTube/TikTok/Instagram/etc).")
    dl.add_argument("--outdir", required=True, help="Output directory for downloaded files.")
    dl.add_argument(
        "--format",
        default="bv*+ba/b",
        help="yt-dlp format selector. Default: bv*+ba/b (best video+audio, fallback best).",
    )
    dl.add_argument("--filename", default=None, help="Optional base filename (without extension).")
    dl.add_argument("--no-playlist", action="store_true", help="If URL is a playlist, download only one item.")
    dl.add_argument("--cookies", default=None, help="Optional cookies.txt for authenticated downloads.")
    dl.add_argument("--trim-start", type=float, default=0.0, help="Trim N seconds from the start (e.g. skip intro).")
    dl.add_argument("--max-seconds", type=float, default=0.0, help="Limit output length (seconds). 0 = no limit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "download":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = download_video(
            url=str(args.url),
            outdir=outdir,
            format_selector=str(args.format),
            base_filename=str(args.filename) if args.filename else None,
            no_playlist=bool(args.no_playlist),
            cookies_path=Path(args.cookies) if args.cookies else None,
            trim_start_s=float(args.trim_start),
            max_seconds=float(args.max_seconds),
        )
        print(str(path))
        return 0

    raise SystemExit(2)

