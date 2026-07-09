"""Model-based text signals via the ai-router gateway.

Three independent families:

1. Perplexity / rank statistics (deterministic). We score the text's own
   tokens under qwen3-30b-tq (`prompt_logprobs`): AI text sits in the model's
   high-probability manifold (low perplexity, most tokens rank-1), human text
   surprises the model far more often. GPTZero/GLTR-style, but with exact
   ranks from vLLM instead of top-k guessing.

2. Cross-model agreement (deterministic). The same text scored under a second,
   unrelated model (gpt-oss-120b). Tokenizers differ, so we compare
   bits-per-character. Text that is low-entropy under BOTH independent models
   is very likely machine-written; one model merely liking it is weaker
   evidence (Binoculars-inspired, document-level).

3. LLM judge. qwen3-30b-tq reads the text with a rubric and returns a
   probability + named indicators. Non-deterministic evidence, bounded weight.

Per-signal failures degrade to `score=None` — the ensemble just fuses whatever
evidence survived.
"""
from __future__ import annotations

import asyncio
import logging
import math

from .. import config, llm
from ..ensemble import Signal, sig_score

log = logging.getLogger(__name__)


def _metrics(tokens: list[llm.ScoredToken]) -> dict:
    lps = [t.logprob for t in tokens]
    n = len(lps)
    mean_lp = sum(lps) / n
    ranks = [t.rank for t in tokens]
    n_chars = sum(len(t.text) for t in tokens) or 1
    windows = [lps[i:i + 25] for i in range(0, n - 24, 25)] or [lps]
    win_means = [sum(w) / len(w) for w in windows]
    wm_mean = sum(win_means) / len(win_means)
    win_std = math.sqrt(
        sum((m - wm_mean) ** 2 for m in win_means) / max(len(win_means) - 1, 1)
    )
    return {
        "tokens": n,
        "mean_logprob": round(mean_lp, 4),
        "perplexity": round(math.exp(-mean_lp), 2),
        "bits_per_char": round(-sum(lps) / math.log(2) / n_chars, 4),
        "frac_rank1": round(sum(r == 1 for r in ranks) / n, 4),
        "frac_top10": round(sum(r <= 10 for r in ranks) / n, 4),
        "frac_over_100": round(sum(r > 100 for r in ranks) / n, 4),
        "window_std": round(win_std, 4),
    }


async def _primary_signals(text: str) -> list[Signal]:
    tokens = await llm.score_prompt(text, config.SCORING_MODEL)
    if len(tokens) < 40:
        return [Signal("perplexity", "Predictability", None, 0, "Too few tokens to score")]
    m = _metrics(tokens)

    # Centers below are calibrated on live samples (scripts/calibrate.py):
    # qwen3-30b-tq finds most polished prose predictable, so the boundaries sit
    # much higher than GPT-2-era detectors would put them. Known failure mode:
    # canonical/classic text the model has memorized scores AI-ish here — the
    # rhythm and judge signals are what pull those back.

    # Perplexity: observed mean logprob ≈ -0.8..-1.9 for AI, -1.6..-2.5+ human.
    s_ppl = sig_score(m["mean_logprob"], center=-2.1, scale=0.4)
    # Rank statistics (GLTR): tokens ranked >100 are "human surprises" —
    # observed ≤2% in AI text, 2-8% in human text. Rank-1 share backs it up.
    s_over100 = sig_score(m["frac_over_100"], center=0.03, scale=0.012, invert=True)
    s_rank1 = sig_score(m["frac_rank1"], center=0.60, scale=0.06)
    s_rank = 0.6 * s_over100 + 0.4 * s_rank1
    # Windowed logprob variance: humans spike and dip (0.7-1.8); AI stays flat
    # (0.3-0.6).
    s_flat = sig_score(m["window_std"], center=0.75, scale=0.22, invert=True)

    return [
        Signal("perplexity", "Predictability (perplexity)", s_ppl, 1.3,
               f"Perplexity {m['perplexity']} under {config.SCORING_MODEL} "
               f"(low = model finds it unsurprising)", m),
        Signal("rank", "Token rank profile", s_rank, 0.9,
               f"{m['frac_rank1']:.0%} of tokens were the model's #1 prediction; "
               f"{m['frac_over_100']:.1%} ranked past 100 (human surprises)", m),
        Signal("flatness", "Predictability evenness", s_flat, 0.7,
               f"Windowed logprob std {m['window_std']} (flat = AI-like)",
               {"window_std": m["window_std"]}),
    ]


def _cross_model_signal(tokens: list[llm.ScoredToken], primary_bits: dict) -> Signal:
    if len(tokens) < 40 or not primary_bits.get("bits_per_char"):
        return Signal("crossmodel", "Cross-model ratio", None, 0, "Unavailable")
    m = _metrics(tokens)
    # Binoculars-lite: gpt-oss-120b scores ALL raw text as high-entropy (its
    # harmony chat format makes plain text out-of-distribution), so absolute
    # bits are useless — but the RATIO of the primary model's bits to the
    # second model's is measurably lower for machine text (observed ~0.06-0.17
    # AI vs ~0.15-0.24 human). Weak separator; low weight.
    ratio = primary_bits["bits_per_char"] / m["bits_per_char"]
    score = sig_score(ratio, center=0.155, scale=0.045, invert=True)
    return Signal(
        "crossmodel", "Cross-model ratio", score, 0.5,
        f"Entropy ratio {ratio:.3f} vs {config.SECONDARY_SCORING_MODEL} "
        "(low = both models find it predictable)",
        {"ratio": round(ratio, 4), "secondary": m},
    )


_JUDGE_PROMPT = """You are a forensic writing analyst. Estimate the probability that the TEXT below was written by an AI language model rather than a human.

Consider: uniformity of sentence rhythm; hedged, balanced, comprehensive-but-shallow coverage; assistant register ("it's important to note", enumerated everything); absence of genuine specificity (names, dates, sensory or first-hand detail, errors, opinions with cost); listicle scaffolding; over-tidy transitions. Human text often has typos, tangents, inconsistency, strong unhedged claims, in-jokes, or personal stakes.

Reply with ONLY a JSON object:
{{"probability_ai": <integer 0-100>, "indicators": ["<up to 4 short concrete observations>"], "summary": "<one sentence>"}}

TEXT:
---
{body}
---"""


async def _judge_signal(text: str) -> Signal:
    body = text[:6000]
    raw = await llm.chat(
        [{"role": "user", "content": _JUDGE_PROMPT.format(body=body)}],
        model=config.JUDGE_MODEL, max_tokens=400,
    )
    parsed = llm.extract_json(raw)
    if not parsed or "probability_ai" not in parsed:
        return Signal("judge", "LLM judge", None, 0, "Judge reply unparseable")
    p = min(max(float(parsed["probability_ai"]) / 100.0, 0.0), 1.0)
    indicators = [str(i) for i in parsed.get("indicators", [])][:4]
    return Signal(
        "judge", f"LLM judge ({config.JUDGE_MODEL})", p, 1.0,
        parsed.get("summary", "") or "; ".join(indicators),
        {"probability_ai": parsed.get("probability_ai"), "indicators": indicators},
    )


async def analyze(text: str) -> list[Signal]:
    """Run all model-based signals concurrently; tolerate individual failures."""
    body = text[:config.SCORING_MAX_CHARS]

    primary_task = asyncio.create_task(_primary_signals(body))
    secondary_task = asyncio.create_task(
        llm.score_prompt(body, config.SECONDARY_SCORING_MODEL)
    )
    judge_task = asyncio.create_task(_judge_signal(body))

    signals: list[Signal] = []
    primary_bits: dict = {}
    try:
        primary = await primary_task
        signals.extend(primary)
        for s in primary:
            if s.name == "perplexity" and s.raw:
                primary_bits = s.raw
    except Exception as exc:
        log.warning("primary scoring failed: %s", exc)
        signals.append(Signal("perplexity", "Predictability", None, 0,
                              f"Scoring unavailable ({type(exc).__name__})"))

    try:
        signals.append(_cross_model_signal(await secondary_task, primary_bits))
    except Exception as exc:
        log.warning("cross-model scoring failed: %s", exc)
        signals.append(Signal("crossmodel", "Second-model agreement", None, 0,
                              f"Unavailable ({type(exc).__name__})"))

    try:
        signals.append(await judge_task)
    except Exception as exc:
        log.warning("judge failed: %s", exc)
        signals.append(Signal("judge", "LLM judge", None, 0,
                              f"Unavailable ({type(exc).__name__})"))
    return signals
