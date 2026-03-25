from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


def clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def softstep(x: float, edge0: float, edge1: float) -> float:
    """Map x to [0,1] with smooth-ish interpolation between edges."""
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = (x - edge0) / (edge1 - edge0)
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def zscore(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
    s = math.sqrt(v) if v > 0 else 1.0
    return [(x - m) / s for x in xs]


def moving_avg(xs: list[float], win: int) -> list[float]:
    if win <= 1 or not xs:
        return xs[:]
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > win:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def load_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    q = clamp01(q)
    idx = int(round(q * (len(xs2) - 1)))
    return float(xs2[idx])


def iter_range_pairs(starts: Iterable[float], ends: Iterable[float]) -> list[tuple[float, float]]:
    s = list(starts)
    e = list(ends)
    n = min(len(s), len(e))
    return [(float(s[i]), float(e[i])) for i in range(n)]


_INVALID_WIN_CHARS = re.compile(r'[<>:"/\\\\|?*]+')
_CONTROL = re.compile(r"[\x00-\x1f]+")


def slugify_filename(text: str, max_len: int = 72) -> str:
    """
    Create a Windows-safe filename slug from text.
    """
    s = (text or "").strip().lower()
    s = _CONTROL.sub(" ", s)
    s = _INVALID_WIN_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = s.replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s).strip("-_. ")
    if not s:
        return "clip"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-_. ")
    return s or "clip"

