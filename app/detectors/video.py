"""Video AI-generation detection (v1).

Signal families:

* Container provenance — ffprobe metadata: generator names (Sora, Veo, Runway,
  Kling, Wan, …), C2PA/JUMBF manifests in the file bytes, AIGC tags. Near-
  definitive when present.
* Camera metadata — com.apple.quicktime.make/model, Android capture tags, GPS:
  evidence of a real capture (weak — re-encodes strip it).
* Frame classifier ensemble — sample ~1 fps (capped), run the image classifier
  (GPU-batched when VRAM is free), aggregate with a trimmed mean + "fraction of
  frames called AI".
* Frame forensics — spectral + noise stats on a small frame subset. Video
  compression erodes these, so the weight is low.
* Temporal noise coherence — on short bursts of consecutive frames: real sensor
  noise is temporally uncorrelated, so adjacent-frame noise residuals decorrelate
  once the static texture component is removed; generated/denoised video keeps
  residuals coherent. Scored conservatively (v1 calibration is thin).

Analysis is synchronous and CPU/GPU-bound — the caller runs it in a worker
thread behind an async job (see main.py).
"""
from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .. import config
from ..ensemble import Signal, combine, sig_score
from . import image_forensics, image_ml

log = logging.getLogger(__name__)

_GEN_MARKERS = [
    (rb"\bSora\b|OpenAI", "OpenAI/Sora tag"),
    (rb"\bVeo\b|Google DeepMind", "Google Veo tag"),
    (rb"Runway|Gen-\d", "Runway tag"),
    (rb"Kling|Kuaishou", "Kling tag"),
    (rb"Pika", "Pika tag"),
    (rb"Luma|Dream Machine", "Luma tag"),
    (rb"Hailuo|MiniMax", "Hailuo/MiniMax tag"),
    (rb"Wan2|Wan-AI|WanVideo", "Wan tag"),
    (rb"Hunyuan", "HunyuanVideo tag"),
    (rb"Mochi|CogVideo|LTX-Video|Stable Video", "open video generator tag"),
    (rb"AIGC|AI[- ]generated", "AIGC tag"),
]
_CAMERA_TAG_KEYS = (
    "com.apple.quicktime.make", "com.apple.quicktime.model",
    "com.apple.quicktime.location.iso6709", "com.android.version",
    "com.android.capture.fps", "location",
)


class VideoError(Exception):
    pass


def _ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise VideoError(f"Not a decodable video ({proc.stderr.strip()[:120]})")
    return json.loads(proc.stdout or "{}")


def _extract_sampled(path: Path, out_dir: Path, seconds: int, max_frames: int) -> list[Path]:
    """~1 fps sampled frames from the first `seconds`, longest side <=1024."""
    pattern = out_dir / "s%03d.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-t", str(seconds), "-i", str(path),
         "-vf", "fps=1,scale='min(1024,iw)':-2", "-frames:v", str(max_frames),
         "-q:v", "3", str(pattern)],
        capture_output=True, timeout=300, check=True,
    )
    return sorted(out_dir.glob("s*.jpg"))


def _extract_bursts(path: Path, out_dir: Path, duration: float,
                    n_bursts: int = 3, burst_len: int = 5) -> list[list[Path]]:
    """Short runs of consecutive native-rate frames for temporal analysis."""
    bursts = []
    usable = max(min(duration, config.VIDEO_ANALYZE_SECONDS) - 1.0, 0.1)
    for i in range(n_bursts):
        t = usable * (0.15 + 0.7 * i / max(n_bursts - 1, 1))
        pattern = out_dir / f"b{i}_%02d.png"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
                 "-vf", "scale='min(640,iw)':-2", "-frames:v", str(burst_len),
                 str(pattern)],
                capture_output=True, timeout=120, check=True,
            )
        except subprocess.SubprocessError:
            continue
        frames = sorted(out_dir.glob(f"b{i}_*.png"))
        if len(frames) >= 3:
            bursts.append(frames)
    return bursts


def _provenance_signals(path: Path, probe: dict) -> list[Signal]:
    tags: dict = {}
    for src in [probe.get("format", {})] + probe.get("streams", []):
        tags.update({str(k).lower(): str(v) for k, v in (src.get("tags") or {}).items()})
    tag_blob = ("\n".join(f"{k}={v}" for k, v in tags.items())).encode("utf-8", "ignore")

    # C2PA/JUMBF + generator names: metadata boxes live near the start/end of
    # the container — scan head and tail, not the (huge) sample data.
    with path.open("rb") as fh:
        head = fh.read(4_000_000)
        fh.seek(max(path.stat().st_size - 2_000_000, 0))
        tail = fh.read()
    found = []
    if re.search(rb"c2pa|jumb", head + tail):
        found.append("C2PA/JUMBF manifest present")
    if re.search(rb"trainedAlgorithmicMedia", head + tail):
        found.append("IPTC trainedAlgorithmicMedia marker")
    for pattern, label in _GEN_MARKERS:
        if re.search(pattern, tag_blob, re.IGNORECASE):
            found.append(label)

    signals = []
    if found:
        ai_specific = [f for f in found if "C2PA" not in f]
        signals.append(Signal(
            "provenance", "Generator fingerprints",
            0.97 if ai_specific else 0.75, 3.0 if ai_specific else 1.0,
            "; ".join(found[:3]), {"markers": found},
        ))
    else:
        signals.append(Signal(
            "provenance", "Generator fingerprints", None, 0,
            "No generator metadata (most re-encodes strip it — absence proves nothing)",
        ))

    cam_hits = {k: v for k, v in tags.items()
                if any(k.endswith(ck) or k == ck for ck in _CAMERA_TAG_KEYS)}
    encoder = tags.get("encoder", "") or tags.get("handler_name", "")
    if cam_hits and not found:
        make_model = " ".join(v for k, v in sorted(cam_hits.items())
                              if k.endswith(("make", "model")))
        signals.append(Signal(
            "camera_meta", "Capture-device metadata", 0.12, 1.2,
            f"Camera tags present: {make_model or ', '.join(list(cam_hits)[:2])}",
            {"tags": cam_hits},
        ))
    else:
        signals.append(Signal(
            "camera_meta", "Capture-device metadata", None, 0,
            f"No capture-device tags (encoder: {encoder[:40] or 'unknown'})",
            {"encoder": encoder[:80]},
        ))
    return signals


def _frame_classifier_signal(frames: list[Path]) -> Signal:
    if not frames:
        return Signal("frame_classifier", "Frame classifier", None, 0, "No frames")
    images = [Image.open(f) for f in frames]
    try:
        probs = [p for p in image_ml.classify_batch(images) if p is not None]
    except Exception as exc:
        return Signal("frame_classifier", "Frame classifier", None, 0,
                      f"Classifier failed ({type(exc).__name__})")
    finally:
        for im in images:
            im.close()
    if not probs:
        return Signal("frame_classifier", "Frame classifier", None, 0,
                      "Classifier unavailable")
    arr = np.sort(np.array(probs))
    trim = max(len(arr) // 10, 0)
    tmean = float(arr[trim:len(arr) - trim].mean()) if len(arr) > 2 * trim else float(arr.mean())
    frac_ai = float((arr > 0.5).mean())
    # Blend level and breadth — then CLAMP hard. Validation showed the image
    # classifier is out-of-domain on compressed video frames (unanimously wrong
    # in both directions on some clips), so it is never allowed to be certain.
    score = min(max(0.7 * tmean + 0.3 * frac_ai, 0.15), 0.85)
    return Signal(
        "frame_classifier", "Frame classifier ensemble", score, 0.5,
        f"{len(arr)} frames: trimmed-mean P(AI) {tmean:.0%}, {frac_ai:.0%} of frames >50% "
        "(image model on video frames — capped, low trust)",
        {"frames": len(arr), "trimmed_mean": round(tmean, 4),
         "frac_ai": round(frac_ai, 4)},
    )


def _frame_forensics_signal(frames: list[Path]) -> Signal:
    subset = frames[:: max(len(frames) // 6, 1)][:6]
    spec_scores, noise_scores = [], []
    for f in subset:
        try:
            img = Image.open(f)
            s = image_forensics.spectral(img)
            n = image_forensics.noise(img)
            if s.score is not None:
                spec_scores.append(s.score)
            if n.score is not None:
                noise_scores.append(n.score)
        except Exception:
            continue
    if not spec_scores and not noise_scores:
        return Signal("frame_forensics", "Frame forensics", None, 0, "Unavailable")
    parts = spec_scores + noise_scores
    score = float(np.mean(parts))
    return Signal(
        "frame_forensics", "Frame forensics (spectral+noise)", score, 0.4,
        f"Mean over {len(subset)} frames — compression-degraded, low weight",
        {"spectral": [round(s, 3) for s in spec_scores],
         "noise": [round(s, 3) for s in noise_scores]},
    )


def _residual(gray: np.ndarray) -> np.ndarray:
    med = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
        dtype=np.float64,
    )
    return gray - med


def _temporal_signal(bursts: list[list[Path]]) -> Signal:
    """Adjacent-frame noise-residual correlation, static-texture compensated.

    For consecutive frames A,B,C: texture leakage in the residual is shared by
    all three, sensor noise only by none. corr(resA-resB, resB-resC) of the
    *differences* isolates the per-frame noise behaviour: camera noise gives a
    strong negative correlation (~-0.5, the shared-B term), while denoised /
    generated video (little per-frame noise) drifts toward 0.
    """
    corrs = []
    noise_levels = []
    for burst in bursts:
        grays = []
        for f in burst:
            img = Image.open(f).convert("L")
            grays.append(np.asarray(img, dtype=np.float64))
        if len({g.shape for g in grays}) != 1:
            continue
        residuals = [_residual(g) for g in grays]
        noise_levels.extend(float(r.std()) for r in residuals)
        diffs = [residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)]
        for i in range(len(diffs) - 1):
            a, b = diffs[i].ravel(), diffs[i + 1].ravel()
            denom = a.std() * b.std()
            if denom > 1e-9:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    if not corrs:
        return Signal("temporal", "Temporal noise coherence", None, 0,
                      "Could not extract frame bursts")
    mean_corr = float(np.mean(corrs))
    mean_noise = float(np.mean(noise_levels)) if noise_levels else 0.0
    # Camera video: mean_corr near -0.5 and healthy noise floor. Generated or
    # aggressively denoised video: corr toward 0 and/or a very low noise floor.
    # Caveat from validation: modern phone footage is temporally denoised and
    # also lands near corr 0, and some generators ADD per-frame grain — so this
    # separates "noise-preserving capture" from everything else, not
    # camera-vs-AI directly. Low weight accordingly.
    s_corr = sig_score(mean_corr, center=-0.32, scale=0.10)
    s_floor = sig_score(mean_noise, center=1.0, scale=0.4, invert=True)
    score = 0.65 * s_corr + 0.35 * s_floor
    return Signal(
        "temporal", "Temporal noise coherence", score, 0.4,
        f"Adjacent-residual corr {mean_corr:+.2f} (camera ≈ -0.5), "
        f"noise floor {mean_noise:.2f}",
        {"mean_corr": round(mean_corr, 4), "noise_floor": round(mean_noise, 3),
         "pairs": len(corrs)},
    )


def _context_signals(probe: dict, duration: float) -> list[Signal]:
    """Circumstantial evidence — deliberately tiny weights, honest labels."""
    signals = []
    # NB: audio/length are weak-and-weakening priors — Veo-3-era generators
    # produce audio and long-form output (confirmed by a labeled miss on
    # 2026-07-09), so their "human" pull is kept close to neutral.
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    signals.append(Signal(
        "audio", "Audio track", 0.38 if has_audio else 0.62, 0.35,
        "Audio present (mild capture evidence — newer generators emit audio too)"
        if has_audio
        else "No audio track (most generator output is silent)",
        {"has_audio": has_audio},
    ))
    if duration <= 8:
        signals.append(Signal(
            "duration", "Clip length", 0.60, 0.25,
            f"{duration:.1f}s — short clips are the typical generator output length",
            {"duration_s": duration},
        ))
    elif duration >= 20:
        signals.append(Signal(
            "duration", "Clip length", 0.42, 0.25,
            f"{duration:.1f}s — long for a generator, but extensions make this weak evidence",
            {"duration_s": duration},
        ))
    else:
        signals.append(Signal("duration", "Clip length", None, 0,
                              f"{duration:.1f}s — uninformative"))
    return signals


def analyze(path: Path) -> dict:
    probe = _ffprobe(path)
    fmt = probe.get("format", {})
    duration = float(fmt.get("duration") or 0)
    vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise VideoError("File contains no video stream")
    if duration < 1.0:
        raise VideoError("Video too short to analyze (<1s)")

    signals = _provenance_signals(path, probe)
    signals.extend(_context_signals(probe, duration))
    with tempfile.TemporaryDirectory(prefix="aidet-video-") as td:
        tmp = Path(td)
        try:
            frames = _extract_sampled(
                path, tmp, config.VIDEO_ANALYZE_SECONDS, config.VIDEO_MAX_SAMPLED_FRAMES
            )
        except subprocess.SubprocessError as exc:
            raise VideoError(f"Frame extraction failed: {exc}") from exc
        signals.append(_frame_classifier_signal(frames))
        signals.append(_frame_forensics_signal(frames))
        signals.append(_temporal_signal(_extract_bursts(path, tmp, duration)))

    result = combine(signals)
    v = vstreams[0]
    result["video"] = {
        "duration_s": round(duration, 1),
        "codec": v.get("codec_name"),
        "resolution": f"{v.get('width')}x{v.get('height')}",
        "analyzed_seconds": min(duration, config.VIDEO_ANALYZE_SECONDS),
    }
    return result
