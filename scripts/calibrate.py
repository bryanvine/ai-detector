"""Sanity-check detector calibration against known-AI and known-human samples.

AI samples are generated live via the router (qwen3 + gpt-oss); human samples
are pre-LLM-era text fetched from the web (Project Gutenberg, Paul Graham) or
files under samples/human/. Prints per-sample metrics and the final verdicts —
used to hand-tune the sigmoid centers in the detectors, and to catch
regressions after changing them.

Usage: .venv/bin/python scripts/calibrate.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app import config, llm
from app.main import analyze_text_content

AI_PROMPTS = [
    ("qwen3-30b-tq", "Write a ~300 word blog post about the benefits of morning exercise."),
    ("qwen3-30b-tq", "Explain in ~300 words how photosynthesis works, for a curious adult."),
    ("openai/gpt-oss-120b", "Write a ~300 word product review of a mid-range espresso machine."),
    ("openai/gpt-oss-120b", "Write a ~300 word personal-sounding story about learning to ride a bike. Make it feel human: include hesitations, a typo or two, informal asides."),
]

HUMAN_SOURCES = [
    ("gutenberg-twain", "https://www.gutenberg.org/cache/epub/76/pg76.txt", 8000, 10500),
    ("gutenberg-darwin", "https://www.gutenberg.org/cache/epub/1228/pg1228.txt", 12000, 14500),
    ("paulgraham-essay", "http://www.paulgraham.com/ds.html", None, None),
]


async def gen_ai_samples() -> list[tuple[str, str]]:
    out = []
    for model, prompt in AI_PROMPTS:
        try:
            text = await llm.chat([{"role": "user", "content": prompt}],
                                  model=model, max_tokens=600, temperature=0.8)
            out.append((f"AI[{model.split('/')[-1]}]:{prompt[:34]}", text.strip()))
        except Exception as exc:
            print(f"  ! generation failed ({model}): {exc}")
    return out


def fetch_human_samples() -> list[tuple[str, str]]:
    import html
    import re

    out = []
    for name, url, start, end in HUMAN_SOURCES:
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            body = r.text
            if url.endswith(".html"):
                body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.S | re.I)
                body = html.unescape(re.sub(r"<[^>]+>", " ", body))
                body = re.sub(r"\s+", " ", body)
                body = body[1000:4200]
            else:
                body = body[start:end]
            out.append((f"HUMAN:{name}", body))
        except Exception as exc:
            print(f"  ! fetch failed ({name}): {exc}")
    for f in sorted((Path(__file__).parent.parent / "samples" / "human").glob("*.txt")):
        out.append((f"HUMAN:{f.stem}", f.read_text()[:6000]))
    return out


async def main():
    print(f"router={config.ROUTER_BASE_URL} scorer={config.SCORING_MODEL}\n")
    print("Generating AI samples via router ...")
    samples = await gen_ai_samples()
    print("Fetching human samples ...")
    samples += fetch_human_samples()

    print(f"\n{'sample':<44} {'ppl':>6} {'rank1':>6} {'b/char':>7} {'judge':>6} {'VERDICT':>8}")
    for name, text in samples:
        try:
            result = await analyze_text_content(text)
        except Exception as exc:
            print(f"{name:<44} FAILED: {exc}")
            continue
        m = {}
        judge = ppl = rank1 = bpc = None
        for s in result["signals"]:
            raw = s.get("raw") or {}
            if s["name"] == "perplexity" and raw:
                ppl, rank1, bpc = raw.get("perplexity"), raw.get("frac_rank1"), raw.get("bits_per_char")
            if s["name"] == "judge" and raw:
                judge = raw.get("probability_ai")
        print(f"{name:<44} {ppl or '-':>6} {rank1 or '-':>6} {bpc or '-':>7} "
              f"{judge if judge is not None else '-':>6} {result['percent']:>7}%")
        for s in result["signals"]:
            if s["score"] is not None:
                print(f"    {s['name']:<12} score={s['score']:.2f} w={s['weight']}  {s['detail'][:90]}")


if __name__ == "__main__":
    asyncio.run(main())
