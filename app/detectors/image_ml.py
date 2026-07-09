"""Learned image classifier (HuggingFace image-classification pipeline, CPU).

The default model is a Swin transformer fine-tuned to separate diffusion output
from photographs/human art. It's the strongest single image signal we can run
locally, but like every 2023-era detector it lags the newest generators, so it
shares the ensemble with the forensics signals instead of speaking alone.

Loaded lazily in a worker thread on first request; if the model can't load
(offline, bad model id, torch missing) the signal reports unavailable and the
rest of the pipeline still works.
"""
from __future__ import annotations

import logging
import threading

from PIL import Image

from .. import config
from ..ensemble import Signal

log = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe = None
_load_error: str | None = None

_AI_LABELS = {"artificial", "ai", "fake", "generated", "synthetic", "deepfake"}
_HUMAN_LABELS = {"human", "real", "authentic", "photo", "natural"}


def _get_pipe():
    global _pipe, _load_error
    with _lock:
        if _pipe is None and _load_error is None:
            try:
                from transformers import pipeline

                log.info("loading image classifier %s ...", config.IMAGE_ML_MODEL)
                _pipe = pipeline(
                    "image-classification", model=config.IMAGE_ML_MODEL, device=-1
                )
                log.info("image classifier ready")
            except Exception as exc:
                _load_error = f"{type(exc).__name__}: {exc}"
                log.error("image classifier failed to load: %s", _load_error)
        return _pipe


def classify(img: Image.Image) -> Signal:
    if not config.IMAGE_ML_ENABLED:
        return Signal("classifier", "ML classifier", None, 0, "Disabled")
    pipe = _get_pipe()
    if pipe is None:
        return Signal("classifier", "ML classifier", None, 0,
                      f"Model unavailable ({_load_error})")

    results = pipe(img.convert("RGB"))
    ai_prob = None
    for r in results:
        label = r["label"].lower()
        if label in _AI_LABELS:
            ai_prob = float(r["score"])
            break
        if label in _HUMAN_LABELS:
            ai_prob = 1.0 - float(r["score"])
            break
    if ai_prob is None:  # unknown label set — take top label at face value
        top = results[0]
        ai_prob = float(top["score"]) if "art" in top["label"].lower() else None
    if ai_prob is None:
        return Signal("classifier", "ML classifier", None, 0,
                      f"Unrecognized labels: {[r['label'] for r in results][:4]}")

    model_short = config.IMAGE_ML_MODEL.split("/")[-1]
    return Signal(
        "classifier", f"ML classifier ({model_short})", ai_prob, 1.5,
        f"Classifier P(AI) = {ai_prob:.0%}",
        {"model": config.IMAGE_ML_MODEL,
         "labels": {r["label"]: round(float(r["score"]), 4) for r in results}},
    )
