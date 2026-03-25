from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]  # ...\viral_web_app
# Suite root (one folder containing everything)
SUITE_ROOT = ROOT.parent

VIRAL_CLIP_SCORER_DIR = SUITE_ROOT / "viral_clip_scorer"
VIDEO_DOWNLOADER_DIR = SUITE_ROOT / "video_downloader"
CLIP_MONTAGER_DIR = SUITE_ROOT / "clip_montager"

# B-roll library lives at suite root
BROLL_DIR_DEFAULT = SUITE_ROOT / "broll_videos"
# All job outputs live here:
RUNS_DIR = SUITE_ROOT / "out"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found on PATH. Make sure FFmpeg is installed and `{name} -version` works.")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def py() -> str:
    return shutil.which("python") or "python"


@dataclass
class Job:
    id: str
    kind: Literal["score", "broll_add"]
    status: Literal["queued", "running", "done", "error"] = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message: str = ""
    outdir: str | None = None
    result: dict[str, Any] | None = None


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def new_job(kind: Job.kind) -> Job:
    jid = str(uuid.uuid4())
    job = Job(id=jid, kind=kind)
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def set_job(jid: str, **kwargs: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[jid]
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = time.time()


def job_dict(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "message": job.message,
        "outdir": job.outdir,
        "result": job.result,
    }


def list_broll_videos(broll_dir: Path) -> list[str]:
    if not broll_dir.exists():
        return []
    exts = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    files = [p.name for p in broll_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = name.replace("\\", "_").replace("/", "_")
    name = "".join(ch for ch in name if ch not in '<>:"|?*')
    name = name.strip().strip(".")
    return name or f"upload_{int(time.time())}"


def _run_broll_add_url(jid: str, url: str, label: str | None, trim_start: float, max_seconds: float) -> None:
    try:
        require_bin("ffmpeg")
        outdir = BROLL_DIR_DEFAULT
        outdir.mkdir(parents=True, exist_ok=True)

        fname = label or f"broll_{int(time.time())}"
        cmd = [
            py(),
            "-m",
            "video_downloader",
            "download",
            "--url",
            url,
            "--outdir",
            str(outdir),
            "--format",
            "best[ext=mp4]/best",
            "--filename",
            fname,
            "--no-playlist",
            "--trim-start",
            str(trim_start),
            "--max-seconds",
            str(max_seconds),
        ]
        set_job(jid, status="running", message="Downloading to b-roll…")
        run_cmd(cmd, cwd=VIDEO_DOWNLOADER_DIR)
        set_job(jid, status="done", message="Added to b-roll.", result={"broll_dir": str(outdir)})
    except Exception as e:
        set_job(jid, status="error", message=str(e))


def _run_score_url(
    jid: str,
    url: str,
    keep_visuals: Literal["keep", "auto", "replace"],
    selected_broll: str | None,
    export_top: int,
    montage_target_s: float,
) -> None:
    run_dir = RUNS_DIR / jid
    run_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = run_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    # Write outputs directly into RUNS_DIR/<job_id>/ (no extra nested out/ folder)
    outdir = run_dir

    try:
        require_bin("ffmpeg")
        require_bin("ffprobe")

        set_job(jid, status="running", message="Downloading video…", outdir=str(outdir))
        # Download main video
        cmd_dl = [
            py(),
            "-m",
            "video_downloader",
            "download",
            "--url",
            url,
            "--outdir",
            str(downloads_dir),
            "--format",
            "best[ext=mp4]/best",
            "--filename",
            "input",
            "--no-playlist",
        ]
        run_cmd(cmd_dl, cwd=VIDEO_DOWNLOADER_DIR)

        # Find downloaded file (latest)
        vids = sorted(downloads_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not vids:
            raise RuntimeError("Download completed but no file found.")
        video_path = vids[0]

        # B-roll mode mapping
        broll_mode = "never"
        broll_dir = None
        if keep_visuals == "replace":
            broll_mode = "always"
            broll_dir = BROLL_DIR_DEFAULT
        elif keep_visuals == "auto":
            broll_mode = "auto"
            broll_dir = BROLL_DIR_DEFAULT

        # If user chose a specific b-roll file, create a temp b-roll dir with only that file
        if broll_dir and selected_broll:
            src = (broll_dir / selected_broll).resolve()
            if not src.exists():
                raise RuntimeError("Selected b-roll file not found.")
            tmp = run_dir / "broll_selected"
            tmp.mkdir(parents=True, exist_ok=True)
            dst = tmp / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            broll_dir = tmp

        set_job(jid, status="running", message="Scoring + exporting clips…", outdir=str(outdir))
        cmd_score = [
            py(),
            "-m",
            "viral_clip_scorer",
            "score",
            "--video",
            str(video_path),
            "--source-url",
            url,
            "--outdir",
            str(outdir),
            "--export-top",
            str(export_top),
            "--no-asr",
        ]
        if broll_dir:
            cmd_score += ["--broll-dir", str(broll_dir), "--broll-mode", broll_mode]
        run_cmd(cmd_score, cwd=VIRAL_CLIP_SCORER_DIR)

        report_path = outdir / "report.json"
        if not report_path.exists():
            raise RuntimeError("Scoring finished but report.json not found.")

        set_job(jid, status="running", message="Building montage…", outdir=str(outdir))
        cmd_mont = [
            py(),
            "-m",
            "clip_montager",
            "montage",
            "--report",
            str(report_path),
            "--out",
            str(outdir / "montage.mp4"),
            "--target-seconds",
            str(montage_target_s),
            "--top-k",
            "12",
        ]
        run_cmd(cmd_mont, cwd=CLIP_MONTAGER_DIR)

        # Load report for UI rendering
        report = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        set_job(
            jid,
            status="done",
            message="Done.",
            outdir=str(outdir),
            result={
                "report": report,
                "files": {
                    "report_json": "report.json",
                    "montage": _find_first(outdir, "montage_*.mp4") or _find_first(outdir, "montage.mp4"),
                    "clips_dir": "clips",
                },
            },
        )
    except Exception as e:
        set_job(jid, status="error", message=str(e), outdir=str(outdir))


def _find_first(dirpath: Path, pattern: str) -> str | None:
    xs = sorted(dirpath.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xs:
        return None
    return xs[0].name


env = Environment(
    loader=FileSystemLoader(str((ROOT / "app" / "templates").resolve())),
    autoescape=select_autoescape(["html", "xml"]),
)

app = FastAPI(title="Viral Clip Scorer Web")

# Serve generated run outputs
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    tpl = env.get_template("index.html")
    return tpl.render(
        broll=list_broll_videos(BROLL_DIR_DEFAULT),
    )


@app.get("/api/broll/list")
def api_broll_list() -> JSONResponse:
    return JSONResponse({"broll": list_broll_videos(BROLL_DIR_DEFAULT), "broll_dir": str(BROLL_DIR_DEFAULT)})


@app.post("/api/broll/add_url")
def api_broll_add_url(
    url: str = Form(...),
    label: str | None = Form(None),
    trim_start: float = Form(0.0),
    max_seconds: float = Form(180.0),
) -> JSONResponse:
    job = new_job("broll_add")
    t = threading.Thread(
        target=_run_broll_add_url,
        args=(job.id, url, label, float(trim_start), float(max_seconds)),
        daemon=True,
    )
    t.start()
    return JSONResponse({"job_id": job.id})


@app.post("/api/broll/upload")
async def api_broll_upload(
    file: UploadFile = File(...),
    label: str | None = Form(None),
) -> JSONResponse:
    """
    Upload a local file into the suite b-roll library.
    (No trimming here; use URL add for trim/cap, or add later if needed.)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    BROLL_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
    base = _safe_name(label or Path(file.filename).stem)
    ext = Path(file.filename).suffix.lower() or ".mp4"
    dest = (BROLL_DIR_DEFAULT / f"{base}{ext}").resolve()
    # avoid overwrite
    if dest.exists():
        dest = (BROLL_DIR_DEFAULT / f"{base}_{int(time.time())}{ext}").resolve()
    content = await file.read()
    dest.write_bytes(content)
    return JSONResponse({"ok": True, "saved_as": dest.name, "broll_dir": str(BROLL_DIR_DEFAULT)})


@app.post("/api/score/url")
def api_score_url(
    url: str = Form(...),
    keep_visuals: str = Form("keep"),
    selected_broll: str | None = Form(None),
    export_top: int = Form(5),
    montage_seconds: float = Form(240.0),
) -> JSONResponse:
    if keep_visuals not in ("keep", "auto", "replace"):
        raise HTTPException(status_code=400, detail="keep_visuals must be keep|auto|replace")
    job = new_job("score")
    t = threading.Thread(
        target=_run_score_url,
        args=(
            job.id,
            url,
            keep_visuals,  # type: ignore[arg-type]
            selected_broll,
            int(export_top),
            float(montage_seconds),
        ),
        daemon=True,
    )
    t.start()
    return JSONResponse({"job_id": job.id})


@app.get("/api/job/{job_id}")
def api_job(job_id: str) -> JSONResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job_dict(job))


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    tpl = env.get_template("job.html")
    return tpl.render(job=job_dict(job))

