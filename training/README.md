# Video-frame classifier training

Replaces the out-of-domain image classifier for video frames with a model
fine-tuned on actual (compressed) video frames. Built 2026-07-09; **not yet
trained** — the app keeps using the capped image-model fallback until
`VIDEO_FRAME_MODEL` is set in `.env`.

## Runbook

```bash
# 1. Assemble corpus (~700 generator-labeled AI clips from Rapidata +
#    ~100 real Wikimedia clips + portal feedback ledger; a few GB)
.venv/bin/python training/fetch_data.py

# 2. Extract frames (clip-level train/val split, manifest.jsonl)
.venv/bin/python training/build_frames.py

# 3. Train (Swin-tiny base). GPU-gated: refuses to start unless >=3GB VRAM
#    free so it never fights other tenants of the card. ~20-40 min GPU.
.venv/bin/python training/train.py            # add --cpu for overnight CPU
.venv/bin/python training/train.py --holdout-generator sora   # generalization check

# 4. Evaluate — per-generator clip accuracy on val, plus real end-to-end
#    scoring of control clips and the feedback ledger
.venv/bin/python training/evaluate.py
.venv/bin/python training/evaluate.py --clips /tmp/vidtest data/uploads

# 5. Only if eval is convincing: activate in .env and restart the app
#    VIDEO_FRAME_MODEL=./models/video-frame-detector
```

Smoke test any time (2 batches, CPU, no GPU touch):
`.venv/bin/python training/train.py --smoke`

## Design notes

- **Split by clip, never by frame** — frames from one video are near-duplicates;
  a frame-level split inflates val accuracy dishonestly.
- **Save/select by clip-level accuracy** — production averages frame probs per
  clip, so that's the metric that matters.
- Augmentations mimic platform laundering: random crop, h-flip, JPEG
  re-quality 40–92.
- Labels: `0=human, 1=artificial` (matches the app's label mapping).
- Class imbalance (AI clips outnumber real) handled by a weighted sampler;
  fetch more Wikimedia/real footage over time — and the portal feedback
  ledger folds in automatically on the next fetch_data run.
- `--holdout-generator X` trains without one generator and reports on it —
  the honest measure of "will it catch the next Sora".
