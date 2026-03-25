# Video Downloader (yt-dlp)

Small standalone CLI that **downloads a video from a link** (YouTube/TikTok/Instagram/X/etc) using **`yt-dlp`**.

## Requirements

- Windows 10+
- Python 3.10+ (works on Python 3.14)
- **FFmpeg installed** and on PATH (`ffmpeg -version`) for best compatibility (merging A/V, some sites)

## Install

```bash
cd C:\Users\jared\Documents\video_downloader
python -m pip install -r requirements.txt
```

## Usage

Download the best quality to an output directory:

```bash
python -m video_downloader download --url "https://..." --outdir "C:\Users\jared\Documents\downloads"
```

Pick a specific format selector (advanced):

```bash
python -m video_downloader download --url "https://..." --outdir "C:\Users\jared\Documents\downloads" --format "bv*+ba/b"
```

## Output

Prints the saved file path to stdout and writes the file into `--outdir`.

