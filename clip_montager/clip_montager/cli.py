import argparse
from pathlib import Path

from clip_montager.montage import create_montage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clip_montager", description="Create a montage from a viral_clip_scorer report.")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("montage", help="Create montage video from report.json")
    m.add_argument("--report", required=True, help="Path to viral_clip_scorer report.json")
    m.add_argument("--out", required=True, help="Output montage mp4 path")
    m.add_argument("--target-seconds", type=float, default=120.0, help="Target montage length in seconds")
    m.add_argument("--top-k", type=int, default=12, help="Only consider top K clips from report")
    m.add_argument("--reencode", action="store_true", help="Force re-encode (default)")
    m.add_argument("--no-reencode", action="store_true", help="Try stream-copy (faster, less reliable)")
    m.add_argument("--auto-title", action="store_true", help="Auto-title the output filename based on clip captions (default).")
    m.add_argument("--no-auto-title", action="store_true", help="Do not auto-title the output filename.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "montage":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        reencode = True
        if args.no_reencode:
            reencode = False
        if args.reencode:
            reencode = True
        create_montage(
            report_path=Path(args.report),
            out_path=out,
            target_seconds=float(args.target_seconds),
            top_k=int(args.top_k),
            reencode=bool(reencode),
            auto_title=not bool(args.no_auto_title),
        )
        print(str(out.resolve()))
        return 0

    raise SystemExit(2)

