#!/usr/bin/env python3
"""Evaluate a trained frame classifier the way production uses it.

Two modes:
  * --manifest (default): score the val split, break down per generator,
    frame- and clip-level.
  * --clips <dir-or-file...>: score arbitrary video files end-to-end through
    the SAME ffmpeg sampling as the app (useful for /tmp/vidtest controls and
    the feedback ledger).

  .venv/bin/python training/evaluate.py --model models/video-frame-detector
  .venv/bin/python training/evaluate.py --model ... --clips /tmp/vidtest data/uploads/xyz.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "training" / "frames"
sys.path.insert(0, str(ROOT))
from training.train import auc, gpu_free_mb  # noqa: E402


def load(model_dir: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()
    ai_idx = int(model.config.label2id.get("artificial", 1))
    return processor, model, ai_idx


@torch.no_grad()
def score_images(images, processor, model, ai_idx, device, batch=16):
    probs = []
    for i in range(0, len(images), batch):
        px = processor(images=images[i:i + batch], return_tensors="pt")["pixel_values"]
        logits = model(pixel_values=px.to(device)).logits
        probs.extend(torch.softmax(logits.float(), -1)[:, ai_idx].cpu().tolist())
    return probs


def eval_manifest(args, processor, model, ai_idx, device):
    rows = [json.loads(l) for l in (FRAMES / "manifest.jsonl").open()
            if json.loads(l)["split"] == "val"]
    by_clip, clip_meta = defaultdict(list), {}
    ys, ps, gens = [], [], []
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        images = [Image.open(FRAMES / r["path"]).convert("RGB") for r in chunk]
        probs = score_images(images, processor, model, ai_idx, device)
        for r, p in zip(chunk, probs):
            ys.append(r["label"]); ps.append(p); gens.append(r["generator"])
            by_clip[r["clip_id"]].append(p)
            clip_meta[r["clip_id"]] = (r["label"], r["generator"])
    ys, ps = np.array(ys), np.array(ps)
    print(f"\nVAL frames n={len(ys)}: acc {((ps > .5) == ys).mean():.3f} auc {auc(ys, ps):.3f}")
    cl = np.array([clip_meta[c][0] for c in by_clip])
    cp = np.array([np.mean(v) for v in by_clip.values()])
    print(f"VAL clips  n={len(cl)}: acc {((cp > .5) == cl).mean():.3f} auc {auc(cl, cp):.3f}\n")
    print(f"{'generator':<14}{'n_clips':>8}{'clip_acc':>10}{'mean_P(AI)':>12}")
    per_gen = defaultdict(list)
    for c in by_clip:
        label, gen = clip_meta[c]
        per_gen[gen].append((label, float(np.mean(by_clip[c]))))
    for gen, items in sorted(per_gen.items()):
        labels = np.array([x[0] for x in items]); scores = np.array([x[1] for x in items])
        acc = ((scores > .5) == labels).mean()
        print(f"{gen:<14}{len(items):>8}{acc:>10.3f}{scores.mean():>12.3f}")


def eval_clips(paths, processor, model, ai_idx, device):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*")) if p.is_dir() else [p]
    files = [f for f in files if f.suffix.lower() in
             (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".ogv")]
    print(f"\n{'clip':<44}{'frames':>7}{'mean_P(AI)':>12}{'frac>0.5':>10}")
    for f in files:
        with tempfile.TemporaryDirectory() as td:
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-t", "90", "-i", str(f),
                     "-vf", "fps=1,scale='min(1024,iw)':-2", "-frames:v", "48",
                     "-q:v", "3", f"{td}/f%03d.jpg"],
                    capture_output=True, timeout=300, check=True)
            except subprocess.SubprocessError:
                print(f"{f.name:<44}  decode failed")
                continue
            frames = [Image.open(x).convert("RGB") for x in sorted(Path(td).glob("*.jpg"))]
            if not frames:
                continue
            probs = np.array(score_images(frames, processor, model, ai_idx, device))
            print(f"{f.name[:43]:<44}{len(probs):>7}{probs.mean():>12.3f}"
                  f"{(probs > .5).mean():>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "video-frame-detector"))
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()
                       or gpu_free_mb() < 2500) else "cuda"
    print(f"device: {device} | model: {args.model}")
    processor, model, ai_idx = load(args.model, device)
    if args.clips:
        eval_clips(args.clips, processor, model, ai_idx, device)
    else:
        eval_manifest(args, processor, model, ai_idx, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
