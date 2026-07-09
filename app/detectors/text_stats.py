"""Deterministic stylometric signals — no network, pure statistics.

These are weak individually (style is easy to imitate and varies by register),
so they carry low ensemble weights; the heavy lifting is done by the logprob
signals in text_llm. They still matter: they work offline, they're free, and
they catch the classic LLM register (uniform sentences, listicle scaffolding,
the over-used lexicon).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..ensemble import Signal, sig_score

# Words/phrases heavily over-represented in 2023-2025 assistant-style output
# relative to human web text. Rate is per 1k words; any single hit means little.
AI_LEXICON = [
    "delve", "delves", "delving", "tapestry", "multifaceted", "underscores",
    "underscore", "pivotal", "crucial", "fostering", "foster", "leverage",
    "leveraging", "seamless", "seamlessly", "realm", "landscape", "testament",
    "vibrant", "intricate", "intricacies", "boasts", "showcasing", "showcases",
    "elevate", "embark", "unlock", "unleash", "robust", "holistic", "paramount",
    "meticulous", "meticulously", "commendable", "noteworthy", "furthermore",
    "moreover", "additionally", "consequently", "comprehensive", "invaluable",
    "ever-evolving", "game-changer", "cutting-edge", "in conclusion",
    "it's important to note", "it is important to note", "it's worth noting",
    "in today's fast-paced", "plays a vital role", "plays a crucial role",
    "in the realm of", "navigating the", "dive into", "delve into",
    "a testament to", "stands as a", "serves as a", "not only", "but also",
]
_LEXICON_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in AI_LEXICON) + r")\b", re.IGNORECASE
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“(])|\n{2,}")
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ']+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 1]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def analyze(text: str) -> list[Signal]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    sentences = _sentences(text)
    n_words = len(words)
    signals: list[Signal] = []
    if n_words < 30 or len(sentences) < 2:
        return [Signal("stylometry", "Writing style", None, 0,
                       "Text too short for style statistics")]

    # --- Sentence-length burstiness: humans vary cadence more than LLMs do.
    lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    cv = _std([float(x) for x in lengths]) / mean_len if mean_len else 0.0
    # Human prose cv commonly ~0.55-0.9; assistant prose ~0.3-0.5.
    s_burst = sig_score(cv, center=0.50, scale=0.10, invert=True)
    signals.append(Signal(
        "burstiness", "Sentence rhythm", s_burst, 0.5,
        f"Sentence length variation {cv:.2f} (uniform rhythm reads as AI)",
        {"cv": round(cv, 3), "mean_sentence_words": round(mean_len, 1),
         "sentences": len(sentences)},
    ))

    # --- Lexical diversity (windowed type/token ratio to control for length).
    window = 200
    if n_words >= window:
        ttrs = [len(set(words[i:i + window])) / window
                for i in range(0, n_words - window + 1, window)]
        mattr = sum(ttrs) / len(ttrs)
    else:
        mattr = len(set(words)) / n_words
    # LLM output tends to recycle framing vocabulary: lower diversity.
    s_div = sig_score(mattr, center=0.555, scale=0.03, invert=True)
    signals.append(Signal(
        "diversity", "Vocabulary diversity", s_div, 0.2,
        f"Windowed type/token ratio {mattr:.3f}",
        {"mattr": round(mattr, 4), "words": n_words},
    ))

    # --- AI-register lexicon rate per 1k words.
    hits = _LEXICON_RE.findall(text)
    rate = 1000 * len(hits) / n_words
    s_lex = sig_score(rate, center=5.5, scale=2.2)
    top = ", ".join(w for w, _ in Counter(h.lower() for h in hits).most_common(4))
    signals.append(Signal(
        "lexicon", "AI-typical phrasing", s_lex, 0.6,
        f"{len(hits)} hits/1k-rate {rate:.1f}" + (f" ({top})" if top else ""),
        {"rate_per_1k": round(rate, 2), "hits": len(hits)},
    ))

    # --- Formatting scaffold: bullets, bold headers, numbered lists, em-dashes.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    bullets = sum(1 for ln in lines if re.match(r"\s*([-*•]|\d+[.)])\s", ln))
    bullet_frac = bullets / len(lines) if lines else 0.0
    emdash = (text.count("—") + text.count(" - ")) * 1000 / max(len(text), 1)
    bold = text.count("**") // 2
    scaffold = min(1.0, bullet_frac * 1.6 + min(bold, 8) / 12)
    s_fmt = sig_score(scaffold, center=0.38, scale=0.13)
    signals.append(Signal(
        "formatting", "Listicle scaffolding", s_fmt, 0.3,
        f"{bullets} list lines ({bullet_frac:.0%}), {bold} bold spans",
        {"bullet_frac": round(bullet_frac, 3), "bold_spans": bold,
         "emdash_per_1k_chars": round(emdash, 2)},
    ))

    # --- Repetition: recycled trigrams (template-y transitions).
    if n_words >= 60:
        trigrams = Counter(zip(words, words[1:], words[2:]))
        repeated = sum(c - 1 for c in trigrams.values() if c > 1)
        rep_rate = repeated / max(n_words - 2, 1)
        s_rep = sig_score(rep_rate, center=0.045, scale=0.02)
        signals.append(Signal(
            "repetition", "Phrase recycling", s_rep, 0.25,
            f"Repeated trigram rate {rep_rate:.3f}",
            {"repeated_trigrams": repeated, "rate": round(rep_rate, 4)},
        ))

    return signals
