# Clip Montager

Takes the top-scoring clips from a `viral_clip_scorer` `report.json` and creates a single montage video (e.g. ~2 minutes).

## Requirements

- Windows 10+
- Python 3.10+ (works on Python 3.14)
- FFmpeg installed and on PATH (`ffmpeg -version`)

## Usage

```bash
cd C:\Users\jared\Documents\clip_montager
python -m clip_montager montage --report "C:\path\to\report.json" --out "C:\path\to\montage.mp4" --target-seconds 120
```

Options:
- `--top-k`: consider only top K clips from the report (default 12)
- `--target-seconds`: montage length target (default 120)
- `--reencode` / `--no-reencode`: whether to re-encode (default: re-encode for reliability)

