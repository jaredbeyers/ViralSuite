# Viral Web App

Web UI for:
- downloading a video by URL and scoring it (exports top clips + montage)
- downloading a video by URL into the B-roll library
- viewing `report.json` results in a human-friendly way

## Requirements

- Python 3.10+ (works on 3.14)
- FFmpeg + FFprobe on PATH

## Install

```bash
cd C:\Users\jared\Documents\ViralSuite\viral_web_app
python -m pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`

## Notes

This UI calls your existing tools:
- `video_downloader`
- `viral_clip_scorer`
- `clip_montager`

## Outputs

All runs are written into:

- `ViralSuite\out\`
