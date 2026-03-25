import re


_INVALID_WIN_CHARS = re.compile(r'[<>:"/\\\\|?*]+')
_CONTROL = re.compile(r"[\x00-\x1f]+")


def slugify_filename(text: str, max_len: int = 72) -> str:
    s = (text or "").strip().lower()
    s = _CONTROL.sub(" ", s)
    s = _INVALID_WIN_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = s.replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s).strip("-_. ")
    if not s:
        return "montage"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-_. ")
    return s or "montage"

