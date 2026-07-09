"""Learned image classifier (HuggingFace image-classification pipeline).

The default model is a Swin transformer fine-tuned to separate diffusion output
from photographs/human art. It's the strongest single image signal we can run
locally, but like every 2023-era detector it lags the newest generators, so it
shares the ensemble with the forensics signals instead of speaking alone.

Device policy — the RTX 5060 is shared with arch-router and training jobs, so
we are strictly a guest on it:
  * before each run, check free VRAM via nvidia-smi (no CUDA context needed);
    >= GPU_MIN_FREE_MB -> run on GPU, else CPU;
  * CUDA OOM mid-run -> move back to CPU and retry there;
  * after GPU_IDLE_EVICT_S without use, evict the model to CPU and release the
    CUDA cache so training can claim the whole card.
Weights stay fp32 everywhere (~350MB — the pipeline doesn't cast inputs for a
half model, and fp32 on GPU is still ~30x faster than CPU here).

Loaded lazily on first request; if the model can't load (offline, bad model id,
torch missing) the signal reports unavailable and the rest of the pipeline works.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time

from PIL import Image

from .. import config
from ..ensemble import Signal

log = logging.getLogger(__name__)

_lock = threading.RLock()
_pipe = None
_load_error: str | None = None
_device = "cpu"
_last_gpu_use = 0.0
_evictor_started = False

_AI_LABELS = {"artificial", "ai", "fake", "generated", "synthetic", "deepfake"}
_HUMAN_LABELS = {"human", "real", "authentic", "photo", "natural"}


def _gpu_free_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _get_pipe():
    global _pipe, _load_error
    if _pipe is None and _load_error is None:
        try:
            from transformers import pipeline

            log.info("loading image classifier %s ...", config.IMAGE_ML_MODEL)
            _pipe = pipeline(
                "image-classification", model=config.IMAGE_ML_MODEL, device=-1
            )
            log.info("image classifier ready (cpu)")
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            log.error("image classifier failed to load: %s", _load_error)
    return _pipe


def _set_device(dev: str) -> None:
    """Move the model (lock must be held). Falls back to CPU on any failure."""
    global _device
    if _pipe is None or dev == _device:
        return
    import torch

    try:
        if dev == "cuda":
            _pipe.model.to("cuda")
        else:
            _pipe.model.to("cpu")
            torch.cuda.empty_cache()
        _pipe.device = torch.device(dev)
        _device = dev
        log.info("image classifier moved to %s", dev)
    except Exception as exc:
        log.warning("device move to %s failed (%s); staying on cpu", dev, exc)
        _pipe.model.to("cpu")
        _pipe.device = torch.device("cpu")
        _device = "cpu"


def _start_evictor() -> None:
    global _evictor_started
    if _evictor_started:
        return
    _evictor_started = True

    def loop():
        while True:
            time.sleep(60)
            with _lock:
                if (_device == "cuda"
                        and time.time() - _last_gpu_use > config.GPU_IDLE_EVICT_S):
                    log.info("evicting idle classifier from GPU")
                    _set_device("cpu")

    threading.Thread(target=loop, daemon=True, name="gpu-evictor").start()


def _ai_prob(results: list[dict]) -> float | None:
    for r in results:
        label = r["label"].lower()
        if label in _AI_LABELS:
            return float(r["score"])
        if label in _HUMAN_LABELS:
            return 1.0 - float(r["score"])
    top = results[0]
    return float(top["score"]) if "art" in top["label"].lower() else None


def _infer(images: list[Image.Image], batch_size: int) -> list[list[dict]]:
    """Run the pipeline with VRAM gating + OOM fallback. Lock must be held."""
    global _last_gpu_use
    want = "cuda" if _gpu_free_mb() >= config.GPU_MIN_FREE_MB else "cpu"
    _set_device(want)
    try:
        out = _pipe(images, batch_size=batch_size)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and _device == "cuda":
            log.warning("CUDA OOM mid-inference — falling back to CPU")
            _set_device("cpu")
            out = _pipe(images, batch_size=batch_size)
        else:
            raise
    if _device == "cuda":
        _last_gpu_use = time.time()
        _start_evictor()
    # pipeline returns list-of-dicts for a single image, list-of-lists for many
    return out if isinstance(out[0], list) else [out]


def classify_batch(images: list[Image.Image], batch_size: int = 8) -> list[float | None]:
    """P(AI) per image, or None where labels are unrecognized. Raises on hard failure."""
    if not config.IMAGE_ML_ENABLED or not images:
        return [None] * len(images)
    with _lock:
        if _get_pipe() is None:
            return [None] * len(images)
        results = _infer([im.convert("RGB") for im in images], batch_size)
    return [_ai_prob(r) for r in results]


def classify(img: Image.Image) -> Signal:
    if not config.IMAGE_ML_ENABLED:
        return Signal("classifier", "ML classifier", None, 0, "Disabled")
    with _lock:
        if _get_pipe() is None:
            return Signal("classifier", "ML classifier", None, 0,
                          f"Model unavailable ({_load_error})")
        results = _infer([img.convert("RGB")], batch_size=1)[0]
        device = _device
    ai_prob = _ai_prob(results)
    if ai_prob is None:
        return Signal("classifier", "ML classifier", None, 0,
                      f"Unrecognized labels: {[r['label'] for r in results][:4]}")
    model_short = config.IMAGE_ML_MODEL.split("/")[-1]
    return Signal(
        "classifier", f"ML classifier ({model_short})", ai_prob, 1.5,
        f"Classifier P(AI) = {ai_prob:.0%}",
        {"model": config.IMAGE_ML_MODEL, "device": device,
         "labels": {r["label"]: round(float(r["score"]), 4) for r in results}},
    )
