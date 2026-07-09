"""Learned classifiers (HuggingFace image-classification pipelines).

Two slots:
  * the IMAGE model (default Organika/sdxl-detector) — still-image analysis;
  * an optional VIDEO-FRAME model (config.VIDEO_FRAME_MODEL) — a checkpoint
    fine-tuned on video frames by training/train.py. Until it's configured,
    video frames fall back to the image model (capped upstream in video.py,
    since it's out-of-domain there).

Device policy — the GPU may be shared with other workloads, so we are strictly
a guest on it: free-VRAM gate before each run (nvidia-smi, no
CUDA context), CUDA OOM falls back to CPU, and models evict themselves to CPU
after GPU_IDLE_EVICT_S without use. Weights stay fp32 (~350MB; the pipeline
doesn't cast inputs for half models, and fp32 on GPU is still ~30x CPU here).

Loaded lazily; a failed load degrades that signal, never the pipeline.
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

_lock = threading.RLock()          # one lock: serializes all classifier use/moves
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


class _Classifier:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipe = None
        self.load_error: str | None = None
        self.device = "cpu"
        self.last_gpu_use = 0.0

    def _load(self):
        if self.pipe is None and self.load_error is None:
            try:
                from transformers import pipeline

                log.info("loading classifier %s ...", self.model_id)
                self.pipe = pipeline("image-classification",
                                     model=self.model_id, device=-1)
                log.info("classifier %s ready (cpu)", self.model_id)
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                log.error("classifier %s failed to load: %s",
                          self.model_id, self.load_error)
        return self.pipe

    def _set_device(self, dev: str) -> None:
        if self.pipe is None or dev == self.device:
            return
        import torch

        try:
            self.pipe.model.to(dev)
            if dev == "cpu":
                torch.cuda.empty_cache()
            self.pipe.device = torch.device(dev)
            self.device = dev
            log.info("classifier %s moved to %s", self.model_id, dev)
        except Exception as exc:
            log.warning("device move to %s failed (%s); staying on cpu", dev, exc)
            self.pipe.model.to("cpu")
            self.pipe.device = torch.device("cpu")
            self.device = "cpu"

    def infer(self, images: list[Image.Image], batch_size: int) -> list[list[dict]] | None:
        """VRAM-gated inference with OOM fallback. Returns None if unavailable."""
        with _lock:
            if self._load() is None:
                return None
            want = "cuda" if _gpu_free_mb() >= config.GPU_MIN_FREE_MB else "cpu"
            self._set_device(want)
            try:
                out = self.pipe(images, batch_size=batch_size)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.device == "cuda":
                    log.warning("CUDA OOM mid-inference — falling back to CPU")
                    self._set_device("cpu")
                    out = self.pipe(images, batch_size=batch_size)
                else:
                    raise
            if self.device == "cuda":
                self.last_gpu_use = time.time()
                _start_evictor()
            return out if isinstance(out[0], list) else [out]


_image_clf = _Classifier(config.IMAGE_ML_MODEL)
_video_clf = (_Classifier(config.VIDEO_FRAME_MODEL)
              if config.VIDEO_FRAME_MODEL else None)


def _start_evictor() -> None:
    global _evictor_started
    if _evictor_started:
        return
    _evictor_started = True

    def loop():
        while True:
            time.sleep(60)
            with _lock:
                for clf in (_image_clf, _video_clf):
                    if (clf and clf.device == "cuda"
                            and time.time() - clf.last_gpu_use > config.GPU_IDLE_EVICT_S):
                        log.info("evicting idle %s from GPU", clf.model_id)
                        clf._set_device("cpu")

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


def video_model_active() -> bool:
    """True when a dedicated (in-domain) video-frame model is configured and loadable."""
    return _video_clf is not None and _video_clf.load_error is None


def classify_batch(images: list[Image.Image], batch_size: int = 8,
                   video: bool = False) -> list[float | None]:
    """P(AI) per image; None where labels are unrecognized/model unavailable."""
    if not config.IMAGE_ML_ENABLED or not images:
        return [None] * len(images)
    clf = _video_clf if (video and _video_clf) else _image_clf
    results = clf.infer([im.convert("RGB") for im in images], batch_size)
    if results is None:
        return [None] * len(images)
    return [_ai_prob(r) for r in results]


def classify(img: Image.Image) -> Signal:
    if not config.IMAGE_ML_ENABLED:
        return Signal("classifier", "ML classifier", None, 0, "Disabled")
    results = _image_clf.infer([img.convert("RGB")], batch_size=1)
    if results is None:
        return Signal("classifier", "ML classifier", None, 0,
                      f"Model unavailable ({_image_clf.load_error})")
    ai_prob = _ai_prob(results[0])
    if ai_prob is None:
        return Signal("classifier", "ML classifier", None, 0,
                      f"Unrecognized labels: {[r['label'] for r in results[0]][:4]}")
    model_short = config.IMAGE_ML_MODEL.split("/")[-1]
    return Signal(
        "classifier", f"ML classifier ({model_short})", ai_prob, 1.5,
        f"Classifier P(AI) = {ai_prob:.0%}",
        {"model": config.IMAGE_ML_MODEL, "device": _image_clf.device,
         "labels": {r["label"]: round(float(r["score"]), 4) for r in results[0]}},
    )
