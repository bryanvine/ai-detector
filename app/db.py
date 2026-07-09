"""SQLite store for analyses + human feedback.

Every analysis is persisted with its full signal breakdown and the raw content
(text inline, files under data/uploads/), so that user-supplied ground truth
in `feedback` yields labelled (content, signals, verdict) triples — the
training set for future weight re-fits or RL fine-tuning. Export via
/api/export.jsonl.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

from . import config

_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id           TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    kind         TEXT NOT NULL,             -- text | document | image
    filename     TEXT,
    content_sha256 TEXT NOT NULL,
    content_text TEXT,                      -- inline for text/document
    content_path TEXT,                      -- data/uploads/<id>.<ext> for files
    percent      REAL,
    confidence   TEXT,
    signals_json TEXT NOT NULL,
    models_json  TEXT NOT NULL,
    duration_ms  INTEGER,
    client_ip    TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id           TEXT PRIMARY KEY,
    analysis_id  TEXT NOT NULL REFERENCES analyses(id),
    created_at   REAL NOT NULL,
    ground_truth TEXT NOT NULL,             -- ai | human | mixed | unsure
    source_hint  TEXT,                      -- e.g. "GPT-5", "my own writing"
    comment      TEXT,
    client_ip    TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON feedback(analysis_id);
"""


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DATA_DIR / "detector.db", timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


def init() -> None:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = _conn()
    conn.executescript(_SCHEMA)
    # Async-job support (video): analyses gain a lifecycle. Older DBs migrate.
    for ddl in (
        "ALTER TABLE analyses ADD COLUMN status TEXT NOT NULL DEFAULT 'done'",
        "ALTER TABLE analyses ADD COLUMN error TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists


def update_analysis(analysis_id: str, *, result: dict | None, status: str,
                    error: str | None = None, duration_ms: int | None = None) -> None:
    conn = _conn()
    with conn:
        conn.execute(
            """UPDATE analyses SET status = ?, error = ?, percent = ?,
                      confidence = ?, signals_json = ?, duration_ms = ?
               WHERE id = ?""",
            (status, error,
             (result or {}).get("percent"), (result or {}).get("confidence"),
             json.dumps((result or {}).get("signals", [])), duration_ms,
             analysis_id),
        )


def insert_analysis(
    *, analysis_id: str, kind: str, filename: str | None, sha256: str,
    content_text: str | None, content_path: str | None, result: dict,
    models: dict, duration_ms: int, client_ip: str, status: str = "done",
) -> None:
    conn = _conn()
    with conn:
        conn.execute(
            "INSERT INTO analyses (id, created_at, kind, filename, content_sha256,"
            " content_text, content_path, percent, confidence, signals_json,"
            " models_json, duration_ms, client_ip, status)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (analysis_id, time.time(), kind, filename, sha256, content_text,
             content_path, result.get("percent"), result.get("confidence"),
             json.dumps(result.get("signals", [])), json.dumps(models),
             duration_ms, client_ip, status),
        )


def insert_feedback(
    *, analysis_id: str, ground_truth: str, source_hint: str | None,
    comment: str | None, client_ip: str,
) -> str:
    row = _conn().execute(
        "SELECT id FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    if row is None:
        raise KeyError(analysis_id)
    fid = uuid.uuid4().hex[:16]
    conn = _conn()
    with conn:
        conn.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
            (fid, analysis_id, time.time(), ground_truth,
             (source_hint or "")[:200], (comment or "")[:2000], client_ip),
        )
    return fid


def stats() -> dict:
    conn = _conn()
    n_analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    n_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    by_kind = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM analyses GROUP BY kind").fetchall())
    return {"analyses": n_analyses, "feedback": n_feedback, "by_kind": by_kind}


def export_rows():
    """Labelled rows for training: every analysis joined with its feedback."""
    conn = _conn()
    cursor = conn.execute(
        """SELECT a.id, a.created_at, a.kind, a.filename, a.content_sha256,
                  a.content_text, a.content_path, a.percent, a.confidence,
                  a.signals_json, a.models_json,
                  f.ground_truth, f.source_hint, f.comment,
                  f.created_at AS feedback_at
           FROM analyses a LEFT JOIN feedback f ON f.analysis_id = a.id
           ORDER BY a.created_at"""
    )
    for row in cursor:
        d = dict(row)
        d["signals"] = json.loads(d.pop("signals_json"))
        d["models"] = json.loads(d.pop("models_json"))
        yield d
