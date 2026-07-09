"""Deterministic image forensics: provenance metadata + signal-level statistics.

Signal families, strongest first:

* Provenance (near-definitive when present): C2PA/JUMBF content-credential
  manifests, the IPTC `trainedAlgorithmicMedia` DigitalSourceType (embedded by
  DALL·E 3 and others), Stable Diffusion / ComfyUI / generator PNG chunks and
  software tags. Absence proves nothing — most pipelines strip metadata — but
  presence is close to a confession.
* Camera evidence: rich EXIF (make/model/exposure) argues for a real capture.
  Weak on its own (screenshots strip it, and it can be forged) so it earns a
  modest negative weight.
* Spectral statistics: generators leave frequency-domain fingerprints —
  depressed/irregular high-frequency energy and periodic peaks from upsampling.
  We measure the radial power-spectrum slope, its high-band residual, and peak
  prominence. Deliberately low-weighted: these vary with content and recompression.
* Noise residual: camera sensor noise is everywhere and inhomogeneous;
  diffusion output is over-smooth with unnaturally uniform residual statistics.
"""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image, ImageFilter

from ..ensemble import Signal, sig_score

Image.MAX_IMAGE_PIXELS = 80_000_000

# Long/specific byte sequences safe to search across the whole file.
_STRUCTURAL_MARKERS = [
    (rb"trainedAlgorithmicMedia", "IPTC DigitalSourceType=trainedAlgorithmicMedia"),
    (rb"c2pa", "C2PA content-credentials manifest"),
    (rb"jumb", "JUMBF metadata box (content credentials)"),
]
# Generator names: short/common enough that they'd false-match random bytes in
# compressed pixel data, so they are only searched in extracted metadata text.
_NAME_MARKERS = [
    (rb"DALL[-\xc2\xb7]E", "DALL·E tag"),
    (rb"Midjourney", "Midjourney tag"),
    (rb"Stable ?Diffusion", "Stable Diffusion tag"),
    (rb"ComfyUI", "ComfyUI tag"),
    (rb"NovelAI", "NovelAI tag"),
    (rb"Adobe Firefly", "Adobe Firefly tag"),
    (rb"\bGrok\b|\bxAI\b", "Grok/xAI tag"),
    (rb"Google AI|Imagen|SynthID", "Google Imagen/SynthID tag"),
]


def _metadata_blob(data: bytes, img: Image.Image) -> bytes:
    """Concatenate the parts of the file that are actually metadata/text."""
    parts: list[bytes] = [data[:2048]]  # header region (before pixel data)
    for value in img.info.values():
        if isinstance(value, str):
            parts.append(value.encode("utf-8", "ignore"))
        elif isinstance(value, bytes):
            parts.append(value)
    try:
        for value in img.getexif().values():
            parts.append(str(value).encode("utf-8", "ignore"))
    except Exception:
        pass
    for seg in getattr(img, "applist", []) or []:  # JPEG APP/COM segments
        try:
            parts.append(seg[1] if isinstance(seg[1], bytes) else bytes(seg[1]))
        except Exception:
            continue
    start = data.find(b"<x:xmpmeta")  # XMP packet, wherever it sits
    if start != -1:
        end = data.find(b"</x:xmpmeta>", start)
        parts.append(data[start:end + 12] if end != -1 else data[start:start + 65536])
    return b"\x00".join(parts)

_SD_PNG_KEYS = {"parameters", "prompt", "workflow", "sd-metadata", "Comment"}
_CAMERA_EXIF_TAGS = {271, 272, 33434, 33437, 34855, 36867, 37386}  # Make, Model, Exposure, FNumber, ISO, DateTimeOriginal, FocalLength


def provenance(data: bytes, img: Image.Image) -> list[Signal]:
    signals: list[Signal] = []

    found: list[str] = []
    for pattern, label in _STRUCTURAL_MARKERS:
        if re.search(pattern, data[:8_000_000]):  # case-sensitive, exact
            found.append(label)
    meta_blob = _metadata_blob(data, img)
    for pattern, label in _NAME_MARKERS:
        if re.search(pattern, meta_blob, re.IGNORECASE):
            found.append(label)

    png_keys = sorted(set(img.info) & _SD_PNG_KEYS) if img.format == "PNG" else []
    gen_chunk = None
    for k in png_keys:
        value = str(img.info.get(k, ""))[:200]
        if k == "parameters" or "Steps:" in value or '"seed"' in value.lower():
            gen_chunk = f"PNG '{k}' chunk carries generation parameters"
            found.append(gen_chunk)
            break

    if found:
        # A manifest/marker is close to definitive. C2PA alone means "signed
        # provenance exists" (cameras use it too), so require an AI-specific
        # marker for the top score.
        ai_specific = [f for f in found if "C2PA" not in f and "JUMBF" not in f]
        score = 0.97 if ai_specific else 0.75
        signals.append(Signal(
            "provenance", "Generator fingerprints", score, 3.0 if ai_specific else 1.0,
            "; ".join(found[:3]), {"markers": found},
        ))
    else:
        signals.append(Signal(
            "provenance", "Generator fingerprints", None, 0,
            "No generator metadata found (absence proves nothing — most sites strip it)",
        ))

    # Camera EXIF: evidence *against* AI generation.
    try:
        exif = img.getexif()
        cam_tags = _CAMERA_EXIF_TAGS & set(exif.keys())
        if exif and len(cam_tags) >= 2 and not found:
            make_model = " ".join(
                str(exif.get(t, "")).strip() for t in (271, 272)
            ).strip()
            signals.append(Signal(
                "exif", "Camera EXIF present", 0.12, 1.2,
                f"Camera metadata: {make_model or f'{len(cam_tags)} capture tags'}",
                {"camera_tags": len(cam_tags), "make_model": make_model},
            ))
        else:
            signals.append(Signal(
                "exif", "Camera EXIF", None, 0,
                "No camera EXIF (normal for web images — weak evidence either way)",
            ))
    except Exception:
        signals.append(Signal("exif", "Camera EXIF", None, 0, "EXIF unreadable"))

    return signals


def _gray(img: Image.Image, max_side: int = 1024) -> np.ndarray:
    g = img.convert("L")
    if max(g.size) > max_side:
        ratio = max_side / max(g.size)
        g = g.resize((max(1, int(g.width * ratio)), max(1, int(g.height * ratio))),
                     Image.LANCZOS)
    return np.asarray(g, dtype=np.float64)


def spectral(img: Image.Image) -> Signal:
    a = _gray(img)
    h, w = a.shape
    if min(h, w) < 128:
        return Signal("spectral", "Frequency spectrum", None, 0, "Image too small")
    a = a - a.mean()
    window = np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fftshift(np.fft.fft2(a * window))) ** 2

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)  # normalized 0..~0.7
    bins = np.linspace(0.01, 0.5, 60)
    idx = np.digitize(r.ravel(), bins)
    p = power.ravel()
    radial = np.array([
        p[idx == i].mean() if np.any(idx == i) else np.nan
        for i in range(1, len(bins))
    ])
    freqs = (bins[:-1] + bins[1:]) / 2
    ok = ~np.isnan(radial) & (radial > 0)
    if ok.sum() < 12:  # degenerate spectrum (flat/synthetic fill)
        return Signal("spectral", "Frequency spectrum", None, 0,
                      "Spectrum too degenerate to fit")
    lf, lp = np.log(freqs[ok]), np.log(radial[ok])

    slope, intercept = np.polyfit(lf, lp, 1)
    fit = slope * lf + intercept
    resid = lp - fit
    hi = lf > np.log(0.25)
    hi_resid = float(resid[hi].mean()) if np.any(hi) else 0.0
    # Periodic upsampler peaks: max residual spike in the upper band.
    peak = float(resid[hi].max() - np.median(resid[hi])) if np.any(hi) else 0.0

    # Natural photos: smooth ~1/f^2 decay (slope ≈ -2, small residuals).
    # Diffusion/GAN output: high-band energy deficit or periodic spikes.
    deviation = abs(hi_resid) * 0.8 + max(peak - 0.8, 0.0) * 0.5 + max(-slope - 3.2, 0.0) * 0.4
    score = sig_score(deviation, center=0.55, scale=0.22)
    return Signal(
        "spectral", "Frequency spectrum", score, 0.5,
        f"Spectrum slope {slope:.2f}, high-band residual {hi_resid:+.2f}, peak {peak:.2f}",
        {"slope": round(float(slope), 3), "hi_resid": round(hi_resid, 3),
         "peak": round(peak, 3)},
    )


def noise(img: Image.Image) -> Signal:
    a = _gray(img)
    if min(a.shape) < 96:
        return Signal("noise", "Sensor-noise residual", None, 0, "Image too small")
    med = np.asarray(
        Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
        dtype=np.float64,
    )
    residual = a - med

    bs = 32
    h, w = residual.shape
    stds = [
        float(residual[y:y + bs, x:x + bs].std())
        for y in range(0, h - bs + 1, bs)
        for x in range(0, w - bs + 1, bs)
    ]
    stds = np.array(stds)
    med_std = float(np.median(stds))
    cv = float(stds.std() / stds.mean()) if stds.mean() > 0 else 0.0

    # Real captures: residual energy everywhere (median block std ~1.5-6) with
    # spatial variation. Synthetic: very clean blocks and/or uniform statistics.
    s_energy = sig_score(med_std, center=1.15, scale=0.35, invert=True)
    s_uniform = sig_score(cv, center=0.65, scale=0.18, invert=True)
    score = 0.6 * s_energy + 0.4 * s_uniform
    return Signal(
        "noise", "Sensor-noise residual", score, 0.6,
        f"Median block residual {med_std:.2f}, spatial variation {cv:.2f} "
        "(clean+uniform = synthetic-like)",
        {"median_block_std": round(med_std, 3), "cv": round(cv, 3),
         "blocks": len(stds)},
    )


def analyze(data: bytes, img: Image.Image) -> list[Signal]:
    signals = provenance(data, img)
    for fn in (spectral, noise):
        try:
            signals.append(fn(img))
        except Exception as exc:
            signals.append(Signal(fn.__name__, fn.__name__, None, 0,
                                  f"Failed: {type(exc).__name__}"))
    return signals
