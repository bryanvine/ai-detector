"""ai-detector web portal — FastAPI app.

POST /api/analyze/text      {"text": ...}                -> verdict
POST /api/analyze/file      multipart file (doc | image) -> verdict
POST /api/feedback          ground-truth from the user   -> stored for training
GET  /api/analysis/{id}     re-fetch a past verdict
GET  /api/stats             counters for the footer
GET  /api/export.jsonl      labelled data dump (Bearer EXPORT_TOKEN)
GET  /api/health
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db, ensemble
from .detectors import document, image_forensics, image_ml, text_llm, text_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("ai-detector")

app = FastAPI(title="ai-detector", docs_url=None, redoc_url=None)
db.init()

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp", "image/tiff"}

# ---------------------------------------------------------------- rate limit
_hits: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def check_rate(request: Request) -> None:
    ip = client_ip(request)
    now = time.time()
    dq = _hits[ip]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= config.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Rate limit exceeded — try again in a minute.")
    dq.append(now)
    if len(_hits) > 10_000:  # don't let the map grow unbounded
        for key in [k for k, v in _hits.items() if not v]:
            _hits.pop(key, None)


# ---------------------------------------------------------------- analyzers
def _models_used(kind: str) -> dict:
    if kind == "image":
        return {"classifier": config.IMAGE_ML_MODEL if config.IMAGE_ML_ENABLED else None}
    return {
        "scoring": config.SCORING_MODEL,
        "secondary": config.SECONDARY_SCORING_MODEL,
        "judge": config.JUDGE_MODEL,
    }


async def analyze_text_content(text: str) -> dict:
    signals = text_stats.analyze(text)
    signals.extend(await text_llm.analyze(text))
    return ensemble.combine(signals)


async def analyze_image_content(data: bytes) -> dict:
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"Could not decode image: {exc}") from exc

    loop = asyncio.get_running_loop()
    forensics_task = loop.run_in_executor(None, image_forensics.analyze, data, img)
    ml_task = loop.run_in_executor(None, image_ml.classify, img)
    signals = await forensics_task
    signals.append(await ml_task)
    result = ensemble.combine(signals)
    result["image"] = {"format": img.format, "size": list(img.size)}
    return result


def _store(analysis_id: str, kind: str, filename: str | None, data: bytes,
           text: str | None, result: dict, started: float, ip: str) -> None:
    content_path = None
    if kind != "text":
        ext = Path(filename or "file").suffix[:10] or ".bin"
        path = config.UPLOAD_DIR / f"{analysis_id}{ext}"
        path.write_bytes(data)
        content_path = str(path.relative_to(config.DATA_DIR))
    db.insert_analysis(
        analysis_id=analysis_id, kind=kind, filename=filename,
        sha256=hashlib.sha256(data).hexdigest(),
        content_text=text, content_path=content_path, result=result,
        models=_models_used(kind), duration_ms=int((time.time() - started) * 1000),
        client_ip=ip,
    )


# ---------------------------------------------------------------- routes
class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=config.MAX_TEXT_CHARS)


@app.post("/api/analyze/text")
async def analyze_text(body: TextIn, request: Request):
    check_rate(request)
    text = body.text.strip()
    if len(text) < config.MIN_TEXT_CHARS:
        raise HTTPException(
            400, f"Need at least {config.MIN_TEXT_CHARS} characters for a meaningful verdict."
        )
    started = time.time()
    result = await analyze_text_content(text)
    analysis_id = uuid.uuid4().hex[:16]
    _store(analysis_id, "text", None, text.encode(), text, result, started,
           client_ip(request))
    return {"id": analysis_id, "kind": "text", **result}


@app.post("/api/analyze/file")
async def analyze_file(file: UploadFile, request: Request):
    check_rate(request)
    data = await file.read()
    if len(data) > config.MAX_FILE_BYTES:
        raise HTTPException(413, "File too large (25 MB max).")
    if not data:
        raise HTTPException(400, "Empty file.")
    started = time.time()
    filename = file.filename or "upload"
    ctype = (file.content_type or "").lower()

    if ctype in IMAGE_TYPES or filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
    ):
        kind = "image"
        result = await analyze_image_content(data)
        text = None
    else:
        kind = "document"
        try:
            text, meta_signal = document.extract(filename, data)
        except document.ExtractionError as exc:
            raise HTTPException(400, str(exc)) from exc
        result = await analyze_text_content(text)
        result["signals"].insert(0, meta_signal.as_dict())
        result["document"] = {"chars": len(text)}

    analysis_id = uuid.uuid4().hex[:16]
    _store(analysis_id, kind, filename, data, text, result, started,
           client_ip(request))
    return {"id": analysis_id, "kind": kind, "filename": filename, **result}


class FeedbackIn(BaseModel):
    analysis_id: str = Field(max_length=32)
    ground_truth: str = Field(pattern="^(ai|human|mixed|unsure)$")
    source_hint: str | None = Field(default=None, max_length=200)
    comment: str | None = Field(default=None, max_length=2000)


@app.post("/api/feedback")
async def feedback(body: FeedbackIn, request: Request):
    try:
        fid = db.insert_feedback(
            analysis_id=body.analysis_id, ground_truth=body.ground_truth,
            source_hint=body.source_hint, comment=body.comment,
            client_ip=client_ip(request),
        )
    except KeyError:
        raise HTTPException(404, "Unknown analysis id.")
    return {"ok": True, "feedback_id": fid, "thanks": True}


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    import json as _json

    row = db._conn().execute(
        "SELECT id, created_at, kind, filename, percent, confidence, signals_json "
        "FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Unknown analysis id.")
    d = dict(row)
    d["signals"] = _json.loads(d.pop("signals_json"))
    return d


@app.get("/api/stats")
async def stats():
    return db.stats()


@app.get("/api/export.jsonl")
async def export(request: Request):
    import json as _json

    if not config.EXPORT_TOKEN:
        raise HTTPException(403, "Export disabled (set EXPORT_TOKEN).")
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {config.EXPORT_TOKEN}":
        raise HTTPException(401, "Bad export token.")

    def gen():
        for row in db.export_rows():
            yield _json.dumps(row, default=str) + "\n"

    return StreamingResponse(gen(), media_type="application/jsonl")


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "ai-detector"}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal error — try again."}, status_code=500)


# Static frontend (must be mounted last so /api wins).
_static = Path(__file__).resolve().parent / "static"


@app.get("/")
async def index():
    return FileResponse(_static / "index.html")


app.mount("/", StaticFiles(directory=_static), name="static")
