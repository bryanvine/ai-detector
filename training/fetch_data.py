#!/usr/bin/env python3
"""Assemble the labeled video corpus for frame-classifier training.

Sources:
  * AI clips  — Rapidata text-2-video preference datasets on HuggingFace:
                713+ mp4s named {prompt}_{generator}_{id}.mp4 (sora, pika,
                hunyuan, ray2, alpha, ...). Generator label is kept.
  * Real clips — Wikimedia Commons video search across everyday-footage topics
                (camera-origin, freely licensed).
  * Feedback ledger — videos submitted to the portal whose ground truth a user
                supplied (data/detector.db); the highest-value examples.

Layout produced:
  data/training/clips/ai/<generator>/<file>.mp4
  data/training/clips/real/<source>/<file>
Idempotent: existing files are skipped.

Usage:
  .venv/bin/python training/fetch_data.py                 # full fetch
  .venv/bin/python training/fetch_data.py --ai-limit 5 --real-limit 3   # smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data" / "training" / "clips"

RAPIDATA_DATASETS = [
    "Rapidata/text-2-video-human-preferences",
]
WIKIMEDIA_TOPICS = [
    "street market", "dog park", "cooking kitchen", "city traffic",
    "birds feeding", "waterfall hiking", "train station", "festival crowd",
    "cat playing", "beach waves", "workshop tools", "snow winter",
]
MAX_CLIP_BYTES = 80 * 1024 * 1024


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-detector-training/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-detector-training/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        if tmp.stat().st_size < 50_000:
            tmp.unlink()
            return False
        tmp.rename(dest)
        return True
    except Exception as exc:
        print(f"  ! {dest.name}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False


def fetch_rapidata(limit: int | None) -> int:
    got = 0
    for repo in RAPIDATA_DATASETS:
        try:
            tree = json.loads(_get(
                f"https://huggingface.co/api/datasets/{repo}/tree/main/Videos?recursive=false"
            ))
        except Exception as exc:
            print(f"! could not list {repo}: {exc}", file=sys.stderr)
            continue
        files = [f["path"] for f in tree
                 if isinstance(f, dict) and f["path"].endswith(".mp4")]
        print(f"{repo}: {len(files)} clips listed")
        for path in files:
            name = Path(path).name
            parts = name.split("_")
            generator = parts[1] if len(parts) >= 3 else "unknown"
            dest = CLIPS / "ai" / generator / name
            url = (f"https://huggingface.co/datasets/{repo}/resolve/main/"
                   + urllib.parse.quote(path))
            if _download(url, dest):
                got += 1
                if got % 25 == 0:
                    print(f"  {got} AI clips downloaded ...")
            if limit and got >= limit:
                return got
    return got


def fetch_wikimedia(limit: int | None) -> int:
    got = 0
    seen: set[str] = set()
    per_topic = max(2, (limit or 120) // len(WIKIMEDIA_TOPICS) + 1)
    for topic in WIKIMEDIA_TOPICS:
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"filetype:video {topic}", "gsrnamespace": 6,
            "gsrlimit": per_topic * 3, "prop": "videoinfo", "viprop": "url|size",
        })
        try:
            data = json.loads(_get(f"https://commons.wikimedia.org/w/api.php?{params}"))
        except Exception as exc:
            print(f"! wikimedia search '{topic}' failed: {exc}", file=sys.stderr)
            continue
        topic_got = 0
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            vi = (page.get("videoinfo") or [{}])[0]
            url, size = vi.get("url"), vi.get("size") or 0
            if not url or url in seen or size > MAX_CLIP_BYTES or size < 300_000:
                continue
            seen.add(url)
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])[:120]
            if _download(url, CLIPS / "real" / "wikimedia" / name):
                got += 1
                topic_got += 1
                print(f"  real[{topic}] {name[:60]}")
            if topic_got >= per_topic or (limit and got >= limit):
                break
        if limit and got >= limit:
            break
    return got


def fetch_feedback_ledger() -> int:
    """Videos from the portal with user-supplied ground truth — copy with label."""
    import sqlite3

    db_path = ROOT / "data" / "detector.db"
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT a.id, a.content_path, f.ground_truth FROM analyses a
           JOIN feedback f ON f.analysis_id = a.id
           WHERE a.kind = 'video' AND f.ground_truth IN ('ai', 'human')"""
    ).fetchall()
    got = 0
    for analysis_id, content_path, truth in rows:
        src = ROOT / "data" / content_path
        if not src.exists():
            continue
        side = "ai" if truth == "ai" else "real"
        dest = CLIPS / side / "feedback" / f"{analysis_id}{src.suffix}"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            got += 1
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-limit", type=int, default=None)
    ap.add_argument("--real-limit", type=int, default=None)
    ap.add_argument("--skip-ai", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    if not args.skip_ai:
        print(f"AI clips downloaded: {fetch_rapidata(args.ai_limit)}")
    if not args.skip_real:
        print(f"Real clips downloaded: {fetch_wikimedia(args.real_limit)}")
    print(f"Feedback-ledger clips copied: {fetch_feedback_ledger()}")

    counts = {}
    for side in ("ai", "real"):
        for d in sorted((CLIPS / side).glob("*")) if (CLIPS / side).exists() else []:
            counts[f"{side}/{d.name}"] = sum(1 for _ in d.iterdir())
    print("\ncorpus:", json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
