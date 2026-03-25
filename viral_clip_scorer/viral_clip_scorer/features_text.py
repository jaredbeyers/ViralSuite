from __future__ import annotations

import math
import re
from dataclasses import dataclass

from viral_clip_scorer.utils import clamp01, split_sentences, softstep


HOOK_PHRASES = [
    "wait until you see",
    "nobody talks about",
    "here's the thing",
    "you won't believe",
    "this is why",
    "stop doing this",
    "biggest mistake",
    "i wish i knew",
    "i can't believe",
    "don't make this mistake",
]

COMMENT_TRIGGERS = [
    "unpopular opinion",
    "hot take",
    "change my mind",
    "prove me wrong",
    "everyone is wrong",
    "nobody wants to hear",
    "controversial",
    "you’re doing it wrong",
    "you're doing it wrong",
]

LAUGHTER_TOKENS = ["haha", "lol", "[laughter]", "(laughter)", "laugh", "laughing", "lmao"]


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def score_hook_text(text: str) -> float:
    t = _norm(text)
    if not t:
        return 0.0
    score = 0.0
    for p in HOOK_PHRASES:
        if p in t:
            score += 0.22
    if "?" in t:
        score += 0.12
    if any(w in t for w in ["secret", "truth", "mistake", "nobody", "never", "always"]):
        score += 0.12
    if any(t.startswith(w) for w in ["wait", "stop", "don't", "dont", "here", "listen"]):
        score += 0.10
    # shorter opening lines tend to be hookier
    words = len(t.split())
    score += 0.18 * (1.0 - clamp01(softstep(words, 14, 38)))
    return clamp01(score)


def score_comment_trigger(text: str) -> float:
    t = _norm(text)
    if not t:
        return 0.0
    score = 0.0
    for p in COMMENT_TRIGGERS:
        if p in t:
            score += 0.25
    if "?" in t:
        score += 0.12
    if any(w in t for w in ["should", "shouldn't", "shouldnt", "agree", "disagree", "wrong", "right"]):
        score += 0.10
    if any(w in t for w in ["everyone", "no one", "nobody", "always", "never"]):
        score += 0.10
    return clamp01(score)


def score_laughter(text: str) -> float:
    t = _norm(text)
    if not t:
        return 0.0
    hits = sum(1 for tok in LAUGHTER_TOKENS if tok in t)
    return clamp01(softstep(hits, 0, 3))


def score_caption_potential(sentence: str) -> float:
    s = (sentence or "").strip()
    if not s:
        return 0.0
    t = _norm(s)
    words = len(t.split())
    # Prefer punchy 6-18 words
    len_score = 1.0 - abs(clamp01((words - 12) / 18.0) - 0.0)
    # Curiosity / contrast / strong claims
    bonus = 0.0
    if any(x in t for x in ["biggest", "secret", "truth", "lie", "mistake", "nobody", "everyone"]):
        bonus += 0.18
    if any(x in t for x in ["is", "are"]) and any(x in t for x in ["not", "never", "always"]):
        bonus += 0.08
    if ":" in s or "..." in s:
        bonus += 0.08
    if "?" in s:
        bonus += 0.08
    return clamp01(0.65 * clamp01(len_score) + 0.35 * clamp01(bonus))


def best_caption(text: str) -> tuple[str | None, float]:
    sents = split_sentences(text or "")
    best_s = None
    best_sc = 0.0
    for s in sents:
        sc = score_caption_potential(s)
        if sc > best_sc:
            best_sc = sc
            best_s = s.strip()
    return best_s, float(best_sc)


@dataclass(frozen=True)
class TextScores:
    speech_highlight: float
    viral_topic_match: float
    comment_trigger: float
    reaction_laughter: float
    hook_text: float
    caption: str | None
    caption_score: float


def score_text_block(text: str, trends: list[str]) -> TextScores:
    t = _norm(text)
    if not t:
        cap, cap_sc = (None, 0.0)
        return TextScores(
            speech_highlight=0.0,
            viral_topic_match=0.0,
            comment_trigger=0.0,
            reaction_laughter=0.0,
            hook_text=0.0,
            caption=cap,
            caption_score=cap_sc,
        )

    # Speech highlight: questions, story beats, surprises, stakes
    sh = 0.0
    if "?" in text:
        sh += 0.15
    if any(x in t for x in ["this happened", "then i", "and then", "i was", "we were"]):
        sh += 0.20
    if any(x in t for x in ["almost", "couldn't", "never", "biggest mistake", "got me fired", "ruined"]):
        sh += 0.20
    if any(x in t for x in ["here's how", "step", "tip", "hack", "mistake", "rule"]):
        sh += 0.12
    # length normalization: too long blocks are less highlight-y
    words = len(t.split())
    sh *= clamp01(1.0 - softstep(words, 140, 340))
    sh = clamp01(sh)

    # Topic match: count trend keyword hits (basic baseline)
    trends_norm = [x.strip().lower() for x in trends if x.strip()]
    hits = 0
    for kw in trends_norm:
        if kw and kw in t:
            hits += 1
    topic = clamp01(softstep(hits, 0, 4))

    ct = score_comment_trigger(text)
    laugh = score_laughter(text)

    # Hook text: compute on first sentence / first ~20 words
    first = split_sentences(text)[:1]
    first_text = first[0] if first else text[:140]
    hook = score_hook_text(first_text)

    cap, cap_sc = best_caption(text)
    return TextScores(
        speech_highlight=float(sh),
        viral_topic_match=float(topic),
        comment_trigger=float(ct),
        reaction_laughter=float(laugh),
        hook_text=float(hook),
        caption=cap,
        caption_score=float(cap_sc),
    )


def hook_rewrites(opening: str) -> list[str]:
    o = (opening or "").strip()
    if not o:
        return []
    base = _norm(o)
    out: list[str] = []
    templates = [
        "If you're struggling with {topic}, this is why.",
        "Stop doing {topic} like this — do this instead.",
        "Nobody talks about {topic} — but it matters.",
        "Wait until you see what {topic} is really doing to you.",
    ]
    # crude topic extraction: pick most “content-y” word
    toks = [t for t in re.findall(r"[a-zA-Z']+", base) if len(t) > 3]
    topic = toks[0] if toks else "this"
    for tpl in templates:
        out.append(tpl.format(topic=topic))
    return out[:4]


def title_suggestions(caption: str | None) -> list[str]:
    if not caption:
        return []
    c = caption.strip()
    base = _norm(c)
    # quick “punch-up”
    variants = [
        c,
        f"The biggest mistake about {base.split('about')[-1].strip()}" if "about" in base else f"This is the mistake nobody talks about",
        f"You’re wasting time doing this" if "time" in base or "hours" in base else "You’re doing this wrong",
    ]
    # de-dupe while preserving order
    seen = set()
    out: list[str] = []
    for v in variants:
        vv = v.strip()
        if vv and vv.lower() not in seen:
            out.append(vv)
            seen.add(vv.lower())
    return out[:3]

