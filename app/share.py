"""Shareable verdict permalinks + Open Graph card images.

/r/{id}        server-rendered verdict page (crawler-friendly OG tags; shows
               the verdict and signal breakdown, never the submitted content)
/og/{id}.png   1200x630 unfurl card drawn with PIL (gauge, percent, stamp)

Analysis ids are 16 hex chars (64 bits) — the link is a capability: nothing
is listed anywhere, only someone given the id can view it.
"""
from __future__ import annotations

import html
import io
import json
import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (15, 20, 17)
PANEL = (24, 31, 26)
BONE = (233, 227, 211)
BONE_DIM = (154, 160, 147)
HAIR = (56, 58, 52)
HUMAN = (127, 176, 105)
GOLD = (200, 180, 106)
AI = (255, 149, 56)

_FONTS = Path(__file__).resolve().parent / "static" / "fonts"


def stamp_for(pct: float) -> tuple[str, tuple[int, int, int], str]:
    """(label, rgb, css_class) — mirrors stampFor() in app.js."""
    if pct >= 78:
        return "LIKELY SYNTHETIC", AI, "ai"
    if pct >= 58:
        return "LEANING SYNTHETIC", AI, "ai"
    if pct > 42:
        return "INCONCLUSIVE", GOLD, "mid"
    if pct > 22:
        return "LEANING HUMAN", HUMAN, "human"
    return "LIKELY HUMAN", HUMAN, "human"


@lru_cache(maxsize=16)
def _fraunces(size: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(_FONTS / "Fraunces-var.ttf"), size)
    f.set_variation_by_axes([144, 900, 0, 0])  # opsz, wght, SOFT, WONK
    return f


@lru_cache(maxsize=16)
def _mono(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "IBMPlexMono-SemiBold.ttf" if bold else "IBMPlexMono-Regular.ttf"
    return ImageFont.truetype(str(_FONTS / name), size)


def _spaced(draw: ImageDraw.ImageDraw, xy, text, font, fill, tracking=6):
    """Letter-spaced text (the masthead look)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def og_card(analysis: dict) -> bytes:
    pct = float(analysis["percent"] or 0)
    label, color, _ = stamp_for(pct)
    kind = analysis.get("kind", "content")
    confidence = analysis.get("confidence") or "—"

    img = Image.new("RGB", (1200, 630), INK)
    d = ImageDraw.Draw(img)
    d.rectangle([22, 22, 1177, 607], outline=HAIR, width=2)

    # --- gauge, left
    cx, cy, r, w = 300, 400, 205, 24
    box = [cx - r, cy - r, cx + r, cy + r]
    for start, end, col in [(180, 240, HUMAN), (240, 300, GOLD), (300, 360, AI)]:
        d.arc(box, start, end, fill=col, width=w)
    ang = math.radians(180 + 1.8 * pct)
    nr = r - w * 1.6
    d.line([cx, cy, cx + nr * math.cos(ang), cy + nr * math.sin(ang)],
           fill=BONE, width=10)
    d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=BONE)
    d.text((cx - r, cy + 28), "HUMAN", font=_mono(20), fill=BONE_DIM)
    syn_w = d.textlength("SYNTHETIC", font=_mono(20))
    d.text((cx + r - syn_w, cy + 28), "SYNTHETIC", font=_mono(20), fill=BONE_DIM)

    # --- text block, right
    x = 590
    _spaced(d, (x, 78), "DETECTOR", _mono(30, bold=True), BONE, tracking=10)
    _spaced(d, (x, 122), "SYNTHETIC MEDIA FORENSICS", _mono(16), BONE_DIM, tracking=4)

    numeral = f"{pct:.0f}%" if pct == int(pct) else f"{pct:.1f}%"
    d.text((x - 8, 160), numeral, font=_fraunces(190), fill=BONE)
    d.text((x, 385), "LIKELIHOOD AI-GENERATED", font=_mono(22), fill=BONE_DIM)

    # stamp: bordered, slightly rotated
    sf = _fraunces(44)
    tw = int(sf.getlength(label))
    stamp = Image.new("RGBA", (tw + 76, 110), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.rectangle([6, 18, tw + 66, 92], outline=color, width=5)
    sd.text((36, 26), label, font=sf, fill=color)
    stamp = stamp.rotate(2, expand=True, resample=Image.BICUBIC)
    img.paste(stamp, (x - 12, 425), stamp)

    footer = f"{kind} specimen · confidence {confidence} · detector.vineai.tech"
    size = 20
    while size > 12 and d.textlength(footer, font=_mono(size)) > 1160 - x:
        size -= 1
    d.text((x, 560), footer, font=_mono(size), fill=BONE_DIM)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="DETECTOR — synthetic media forensics">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{base}/r/{id}">
<meta property="og:image" content="{base}/og/{id}.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{base}/og/{id}.png">
<meta name="theme-color" content="#0f1411">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,900&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/styles.css?v=5">
<link rel="icon" href="/icon-192.png">
</head>
<body>
<div class="grain" aria-hidden="true"></div>
<header class="masthead">
  <div class="wordmark">
    <span class="dot" aria-hidden="true"></span>
    <h1><a href="/" style="color:inherit;text-decoration:none">DETECTOR</a></h1>
    <span class="unit">vineai&nbsp;·&nbsp;synthetic&nbsp;media&nbsp;forensics</span>
  </div>
  <nav class="masthead-meta"><span class="stat-line">shared verdict · {kind} specimen</span></nav>
</header>
<main class="bench" style="grid-template-columns:1fr;max-width:860px">
  <section class="verdict" aria-label="Verdict">
    <div class="verdict-live">
      {specimen}
      <div class="gauge-wrap">
        <svg class="gauge" viewBox="0 0 200 110" aria-hidden="true">
          <defs><linearGradient id="arc-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="var(--human)"/><stop offset="50%" stop-color="#c8b46a"/><stop offset="100%" stop-color="var(--ai)"/>
          </linearGradient></defs>
          <path d="M15 100 A 85 85 0 0 1 185 100" fill="none" stroke="var(--hairline)" stroke-width="8" stroke-linecap="round"/>
          <path d="M15 100 A 85 85 0 0 1 185 100" fill="none" stroke="url(#arc-grad)" stroke-width="8" stroke-linecap="round"
                pathLength="100" stroke-dasharray="100" stroke-dashoffset="{dashoffset}"/>
          <g transform="rotate({needle} 100 100)">
            <line x1="100" y1="100" x2="100" y2="26" stroke="var(--bone)" stroke-width="2.5"/>
            <circle cx="100" cy="100" r="5" fill="var(--bone)"/>
          </g>
        </svg>
        <div class="gauge-poles mono-dim" aria-hidden="true"><span>HUMAN</span><span>SYNTHETIC</span></div>
      </div>
      <div class="reading">
        <div class="pct">{pct}<span class="pct-sign">%</span></div>
        <div class="pct-caption mono-dim">likelihood AI-generated</div>
        <div class="stamp {stamp_cls}">{stamp}</div>
        <div class="conf mono-dim">confidence: {confidence} · analyzed {when}</div>
      </div>
      <div class="evidence">
        <h3 class="evidence-head">EVIDENCE <span class="mono-dim">· {n_live} live signals</span></h3>
        <ol class="signals">{signal_rows}</ol>
      </div>
      <div class="feedback" style="text-align:center">
        <h3>Is it AI? Run your own analysis.</h3>
        <p class="mono-dim">Text · documents · images · video — free, open-source, self-hosted on open-weight models.</p>
        <p style="margin-top:14px"><a class="analyze" style="display:inline-block;text-decoration:none;padding:14px 26px" href="/"><span class="btn-label">TRY DETECTOR</span></a></p>
      </div>
    </div>
  </section>
</main>
<footer class="colophon">
  <span>{privacy_note}</span>
  <a href="https://github.com/bryanvine/ai-detector" rel="noopener">source ↗</a>
</footer>
</body>
</html>"""


def share_page(analysis: dict, base: str) -> str:
    pct = float(analysis["percent"] or 0)
    label, _, cls = stamp_for(pct)
    kind = html.escape(analysis.get("kind", "content"))
    confidence = html.escape(str(analysis.get("confidence") or "—"))

    rows = []
    signals = analysis.get("signals") or []
    live = [s for s in signals if s.get("score") is not None]
    order = sorted(live, key=lambda s: -s["score"]) + [s for s in signals if s.get("score") is None]
    for i, s in enumerate(order):
        name = html.escape(s.get("label") or s.get("name") or "")
        detail = html.escape(s.get("detail") or "")
        if s.get("score") is None:
            meter = '<span class="sig-na">n/a</span>'
        else:
            fill = "var(--ai)" if s["score"] >= 0.5 else "var(--human)"
            meter = (f'<span class="sig-meter"><b style="--v:{s["score"]:.3f};'
                     f'--fill:{fill}"></b></span>')
        rows.append(f'<li style="--i:{i}"><span class="sig-name">{name}</span>'
                    f'{meter}<span class="sig-detail">{detail}</span></li>')

    import datetime
    when = datetime.datetime.fromtimestamp(
        analysis.get("created_at", 0), datetime.timezone.utc
    ).strftime("%Y-%m-%d")

    # Only IMAGE specimens are ever rendered on a shared verdict; submitted
    # text, documents and video stay private.
    if analysis.get("kind") == "image":
        specimen = (f'<div style="text-align:center;margin-bottom:26px">'
                    f'<img src="/content/{analysis["id"]}" alt="analyzed image" '
                    f'style="max-width:100%;max-height:380px;border:1px solid '
                    f'var(--hairline);object-fit:contain"></div>')
        privacy_note = "Image specimens are shown on shared verdicts; text, documents and video never are."
    else:
        specimen = ""
        privacy_note = "The submitted content is not shown on shared verdicts — only the analysis."

    pct_str = f"{pct:.0f}" if pct == int(pct) else f"{pct:.1f}"
    return _PAGE.format(
        specimen=specimen, privacy_note=privacy_note,
        title=f"DETECTOR verdict: {pct_str}% likely AI-generated ({label.lower()})",
        desc=(f"Forensic analysis of a {kind} specimen — {len(live)} independent "
              f"signals, confidence {confidence}. Run your own at detector.vineai.tech."),
        base=base, id=analysis["id"], kind=kind, pct=pct_str,
        dashoffset=f"{100 - pct:.1f}", needle=f"{-90 + pct * 1.8:.1f}",
        stamp=label, stamp_cls=cls, confidence=confidence, when=when,
        n_live=len(live), signal_rows="".join(rows),
    )


def load_analysis(conn, analysis_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, created_at, kind, percent, confidence, signals_json, status "
        "FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["signals"] = json.loads(d.pop("signals_json") or "[]")
    return d
