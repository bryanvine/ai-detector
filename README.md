# ai-detector

**https://detector.vineai.tech** — forensic portal that estimates the probability a piece of
content (text, document, or image) was AI-generated, with a per-signal evidence breakdown,
and collects ground-truth feedback for future retraining.

Runs entirely on open-weight models via the local [ai-router](../ai-router) LiteLLM gateway
(`qwen3-30b-tq` primary, `openai/gpt-oss-120b` secondary) plus deterministic statistics —
no external detector APIs.

## How it works

Every analyzer emits independent **signals** (each an estimate of P(AI) with a weight);
they're fused as a weighted mean in logit space so decisive evidence dominates and a lone
confident-but-wrong signal can't saturate the verdict. Weights are hand-calibrated priors
(see `scripts/calibrate.py`), stored with every analysis so they can be re-fit from
accumulated feedback.

### Text (and documents, after extraction)

| Signal | How |
|---|---|
| Perplexity | vLLM `prompt_logprobs` via the gateway: every input token's logprob under `qwen3-30b-tq`. AI text sits in the model's high-probability manifold. |
| Token rank profile | Exact rank of each token (GLTR-style): AI text is mostly the model's #1 pick; tokens ranked >100 are "human surprises". |
| Predictability evenness | Windowed logprob variance — humans spike and dip, AI stays flat. |
| Cross-model ratio | Binoculars-lite: entropy ratio between qwen3 and gpt-oss-120b (both-unsurprised = machine text). |
| LLM judge | `qwen3-30b-tq` with a forensic rubric returns probability + named indicators. |
| Stylometry | Sentence-rhythm burstiness, windowed type/token ratio, AI-register lexicon rate, listicle scaffolding, trigram recycling. Deterministic, low weight. |

Documents (PDF/DOCX/TXT/MD) are text-extracted (`pypdf`, `python-docx`); producer/creator
metadata is surfaced as context but carries no weight (export tools say nothing about authorship).

### Images

| Signal | How |
|---|---|
| Provenance | C2PA/JUMBF manifests, IPTC `trainedAlgorithmicMedia` DigitalSourceType (DALL·E 3 etc.), Stable Diffusion / ComfyUI PNG parameter chunks, generator names in metadata segments. Near-definitive when present. |
| Camera EXIF | Rich capture metadata (make/model/exposure) argues for a real photo. Weak — most pipelines strip it. |
| Frequency spectrum | Radial power-spectrum slope + high-band residual + periodic upsampler peaks (FFT). Low weight. |
| Sensor noise | Median-filter residual energy and spatial uniformity — diffusion output is over-smooth and unnaturally uniform. |
| ML classifier | `Organika/sdxl-detector` (Swin) on CPU via transformers. Strongest single signal; lags newest generators, hence the ensemble. |

**Not implemented (ideas for later):** VLM judge once a vision model lands on the router;
DIRE (diffusion-reconstruction error, needs GPU + SD weights); PRNU camera-fingerprint
matching; OCR of in-image gibberish text; SynthID and other proprietary watermarks
(closed); face/hand anatomical-anomaly heuristics.

## Feedback → training data

Every analysis (content, full signal breakdown, verdict, model versions) is stored in
SQLite; the portal asks users whether they *know* the ground truth (`human / ai / mixed /
unsure` + source hint). `GET /api/export.jsonl` (Bearer `EXPORT_TOKEN`) dumps labelled
rows for RL / re-calibration.

## API

```
POST /api/analyze/text     {"text": "..."}           → {id, percent, confidence, signals[]}
POST /api/analyze/file     multipart file            → same (document or image)
POST /api/feedback         {analysis_id, ground_truth, source_hint?, comment?}
GET  /api/analysis/{id}    re-fetch a verdict
GET  /api/stats            counters
GET  /api/export.jsonl     labelled dataset (auth)
GET  /api/health
```

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env       # set ROUTER_API_KEY (mint via LiteLLM /key/generate)
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 27000
```

Deployment (this server): **27000–27999 port block**, systemd unit
`scripts/ai-detector.service` → `127.0.0.1:27000`, published through the shared
Cloudflare tunnel (`scripts/setup_dns.py` + `scripts/setup_tunnel_remote.py`).
Per-IP rate limiting protects the GPU backends (`RATE_LIMIT_PER_MIN`).

## Calibration & honesty

`scripts/calibrate.py` runs live known-AI generations and known-human corpora through the
pipeline and prints per-signal metrics — used to set the sigmoid centers in
`app/detectors/`. Current behavior on the calibration set: AI samples 53–83%, human
samples 14–36%. Known failure modes, shown honestly in the UI:

- Adversarial "write like a human, add typos" prompts can drop under 50%.
- Canonical pre-LLM text the scorer has memorized (e.g. *On the Origin of Species*) reads
  as low-perplexity; rhythm signals and the judge usually pull it back.
- Heavily edited AI text and non-English text reduce signal quality.
- Scores near 50% mean *uncertain*, not "half AI".
