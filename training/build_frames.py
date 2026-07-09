#!/usr/bin/env python3
"""Extract training frames from the clip corpus and write a manifest.

Frames mirror the inference pipeline (ffmpeg, longest side <=1024, JPEG).
The train/val split is BY CLIP (stable hash of clip id) so frames from one
video never straddle the split — frame-level splits leak trivially.

Real clips are longer, so they contribute more frames per clip than the
short AI clips; residual class imbalance is handled by the sampler in train.py.

  data/training/frames/<label>/<clip_id>__NN.jpg
  data/training/frames/manifest.jsonl   {path,label,clip_id,generator,split}

Usage: .venv/bin/python training/build_frames.py [--ai-frames 8] [--real-frames 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data" / "training" / "clips"
FRAMES = ROOT / "data" / "training" / "frames"
VAL_FRACTION = 0.15


def _duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _extract(path: Path, out_stub: Path, n_frames: int) -> list[Path]:
    existing = sorted(out_stub.parent.glob(out_stub.name + "__*.jpg"))
    if existing:
        return existing
    dur = _duration(path)
    if dur < 0.5:
        return []
    fps = n_frames / max(dur - 0.2, 0.5)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path),
             "-vf", f"fps={fps:.4f},scale='min(1024,iw)':-2",
             "-frames:v", str(n_frames), "-q:v", "3",
             str(out_stub) + "__%02d.jpg"],
            capture_output=True, timeout=600, check=True,
        )
    except subprocess.SubprocessError as exc:
        print(f"  ! {path.name}: {exc}", file=sys.stderr)
        return []
    return sorted(out_stub.parent.glob(out_stub.name + "__*.jpg"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-frames", type=int, default=8)
    ap.add_argument("--real-frames", type=int, default=16)
    args = ap.parse_args()

    manifest_path = FRAMES / "manifest.jsonl"
    FRAMES.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {"ai": 0, "real": 0}

    for side, label in (("ai", 1), ("real", 0)):
        for clip in sorted((CLIPS / side).rglob("*")):
            if not clip.is_file() or clip.suffix.lower() not in (
                ".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".ogv"
            ):
                continue
            generator = clip.parent.name  # sora/pika/... or wikimedia/feedback
            clip_id = f"{side}-{generator}-{clip.stem}"[:120]
            split = ("val" if int(hashlib.md5(clip_id.encode()).hexdigest(), 16)
                     % 1000 < VAL_FRACTION * 1000 else "train")
            out_dir = FRAMES / side
            out_dir.mkdir(parents=True, exist_ok=True)
            n = args.ai_frames if side == "ai" else args.real_frames
            frames = _extract(clip, out_dir / clip_id, n)
            for f in frames:
                rows.append({"path": str(f.relative_to(FRAMES)), "label": label,
                             "clip_id": clip_id, "generator": generator,
                             "split": split})
            counts[side] += len(frames)

    with manifest_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    by_gen: dict[str, int] = {}
    for r in rows:
        by_gen[r["generator"]] = by_gen.get(r["generator"], 0) + 1
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"frames: {counts} | val: {n_val}/{len(rows)}")
    print("by generator:", json.dumps(by_gen, indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
