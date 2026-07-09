#!/usr/bin/env python3
"""Fine-tune a frame classifier (human vs AI-generated) on the video-frame corpus.

Plain PyTorch loop (no accelerate/torchvision deps). Respects the shared-GPU
policy: uses CUDA only if enough VRAM is free at launch, else refuses (or run
--cpu). Saves the best checkpoint by CLIP-LEVEL val accuracy — production
aggregates frames per clip, so that's the metric that matters.

  .venv/bin/python training/train.py --smoke          # 2-batch CPU sanity run
  .venv/bin/python training/train.py                  # real training (GPU when free)
  .venv/bin/python training/train.py --holdout-generator sora   # generalization test

Labels: 0=human, 1=artificial (matches the app's label sets).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "training" / "frames"
DEFAULT_OUT = ROOT / "models" / "video-frame-detector"
DEFAULT_BASE = "microsoft/swin-tiny-patch4-window7-224"


def gpu_free_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (no sklearn dependency)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], processor, augment: bool):
        self.rows, self.processor, self.augment = rows, processor, augment

    def __len__(self):
        return len(self.rows)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.7:  # random resized crop, scale 0.7-1.0
            w, h = img.size
            s = random.uniform(0.7, 1.0)
            cw, ch = int(w * s), int(h * s)
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch))
        if random.random() < 0.5:  # JPEG re-quality: mimic platform re-encodes
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(40, 92))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return img

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(FRAMES / row["path"]).convert("RGB")
        if self.augment:
            img = self._augment(img)
        pixel = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return pixel, row["label"], row["clip_id"]


def collate(batch):
    px = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch])
    clips = [b[2] for b in batch]
    return px, y, clips


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps, clip_scores, clip_labels = [], [], defaultdict(list), {}
    for px, y, clips in loader:
        logits = model(pixel_values=px.to(device)).logits
        prob_ai = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        ys.extend(y.tolist())
        ps.extend(prob_ai.tolist())
        for cid, label, p in zip(clips, y.tolist(), prob_ai.tolist()):
            clip_scores[cid].append(p)
            clip_labels[cid] = label
    ys, ps = np.array(ys), np.array(ps)
    cl = np.array([clip_labels[c] for c in clip_scores])
    cp = np.array([float(np.mean(v)) for v in clip_scores.values()])
    return {
        "frame_acc": float(((ps > 0.5) == ys).mean()),
        "frame_auc": auc(ys, ps),
        "clip_acc": float(((cp > 0.5) == cl).mean()),
        "clip_auc": auc(cl, cp),
        "n_frames": len(ys), "n_clips": len(cl),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-free-mb", type=int, default=3000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--holdout-generator", default=None,
                    help="exclude this generator from training; report on it separately")
    ap.add_argument("--smoke", action="store_true",
                    help="2 batches on CPU — verifies the loop, discards nothing useful")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    manifest = FRAMES / "manifest.jsonl"
    if not manifest.exists():
        sys.exit("no manifest — run fetch_data.py then build_frames.py first")
    rows = [json.loads(l) for l in manifest.open()]
    holdout_rows = []
    if args.holdout_generator:
        holdout_rows = [r for r in rows if r["generator"] == args.holdout_generator]
        rows = [r for r in rows if r["generator"] != args.holdout_generator]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if args.smoke:
        random.seed(0)
        train_rows = random.sample(train_rows, min(8, len(train_rows)))
        val_rows = random.sample(val_rows, min(8, len(val_rows))) or train_rows
        args.epochs, args.batch, args.cpu = 1, 4, True
    print(f"train {len(train_rows)} frames / val {len(val_rows)} frames"
          + (f" / holdout[{args.holdout_generator}] {len(holdout_rows)}" if holdout_rows else ""))
    if not train_rows:
        sys.exit("no training frames")

    if args.cpu:
        device = "cpu"
    else:
        free = gpu_free_mb()
        if not torch.cuda.is_available() or free < args.min_free_mb:
            sys.exit(f"GPU gate: {free}MB free < {args.min_free_mb}MB required "
                     "(other workloads own the card). Re-run when free, or --cpu.")
        device = "cuda"

    processor = AutoImageProcessor.from_pretrained(args.base)
    model = AutoModelForImageClassification.from_pretrained(
        args.base, num_labels=2,
        id2label={0: "human", 1: "artificial"},
        label2id={"human": 0, "artificial": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    # class-balanced sampling
    n_pos = sum(r["label"] for r in train_rows)
    w = {1: len(train_rows) / max(n_pos, 1), 0: len(train_rows) / max(len(train_rows) - n_pos, 1)}
    sampler = WeightedRandomSampler([w[r["label"]] for r in train_rows],
                                    num_samples=len(train_rows), replacement=True)
    train_dl = DataLoader(FrameDataset(train_rows, processor, augment=True),
                          batch_size=args.batch, sampler=sampler,
                          num_workers=0 if args.smoke else 4, collate_fn=collate)
    val_dl = DataLoader(FrameDataset(val_rows, processor, augment=False),
                        batch_size=args.batch, shuffle=False,
                        num_workers=0 if args.smoke else 4, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(len(train_dl) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best_clip_acc, step = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for bi, (px, y, _) in enumerate(train_dl):
            if args.smoke and bi >= 2:
                break
            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=device == "cuda"):
                out = model(pixel_values=px.to(device),
                            labels=y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(out.loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"  e{epoch} step {step}/{total_steps} loss {out.loss.item():.4f}")

        metrics = evaluate(model, val_dl, device)
        metrics.update({"epoch": epoch, "seconds": round(time.time() - t0, 1)})
        print("val:", json.dumps(metrics))
        with log_path.open("a") as fh:
            fh.write(json.dumps(metrics) + "\n")
        if metrics["clip_acc"] > best_clip_acc:
            best_clip_acc = metrics["clip_acc"]
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            (out_dir / "eval_report.json").write_text(json.dumps(metrics, indent=2))
            print(f"  saved best (clip_acc {best_clip_acc:.3f}) -> {out_dir}")

    if holdout_rows:
        hold_dl = DataLoader(FrameDataset(holdout_rows, processor, augment=False),
                             batch_size=args.batch, shuffle=False,
                             num_workers=0, collate_fn=collate)
        print(f"holdout[{args.holdout_generator}]:",
              json.dumps(evaluate(model, hold_dl, device)))
    print("done. NOTE: the app only uses this model once VIDEO_FRAME_MODEL "
          "points at it in .env — evaluate first (training/evaluate.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
