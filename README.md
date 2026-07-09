# ai-detector

**Live demo: https://detector.vineai.tech**

A self-hostable web portal that estimates the probability a piece of content —
text, document, image, or video — was AI-generated, shows the per-signal
evidence behind every verdict, and collects ground-truth feedback to improve
itself over time.

Runs entirely on **open-weight models** behind any OpenAI-compatible endpoint
(vLLM, LiteLLM, Open WebUI, llama.cpp server, ...) plus deterministic
statistics. No external detector APIs; nothing leaves your infrastructure.

## How it works

Every analyzer emits independent **signals** (each an estimate of P(AI) with a
weight); they're fused as a weighted mean in logit space so decisive evidence
dominates and a lone confident-but-wrong signal can't saturate the verdict.
Weights are hand-calibrated priors (see `scripts/calibrate.py`), stored with
every analysis so they can be re-fit from accumulated feedback.

### Text (and documents, after extraction)

| Signal | How |
|---|---|
| Perplexity | vLLM `prompt_logprobs`: every input token's logprob under an open-weight scoring model. AI text sits in the model's high-probability manifold. |
| Token rank profile | Exact rank of each token (GLTR-style): AI text is mostly the model's #1 pick; tokens ranked >100 are "human surprises". |
| Predictability evenness | Windowed logprob variance — humans spike and dip, AI stays flat. |
| Cross-model ratio | Binoculars-lite: entropy ratio between two unrelated models (both-unsurprised = machine text). |
| LLM judge | A chat model with a forensic rubric returns probability + named indicators. |
| Stylometry | Sentence-rhythm burstiness, type/token diversity, AI-register lexicon, em-dash density, structural-emoji decoration, listicle scaffolding, trigram recycling. Deterministic, low weight. |

Documents (PDF/DOCX/TXT/MD) are text-extracted (`pypdf`, `python-docx`);
producer/creator metadata is surfaced as context but carries no weight.

### Images

| Signal | How |
|---|---|
| Provenance | C2PA/JUMBF manifests, IPTC `trainedAlgorithmicMedia` DigitalSourceType, Stable Diffusion / ComfyUI PNG parameter chunks, generator names in metadata segments. Near-definitive when present. |
| Camera EXIF | Rich capture metadata argues for a real photo. Weak — most pipelines strip it. |
| Frequency spectrum | Radial power-spectrum slope + high-band residual + periodic upsampler peaks (FFT). Low weight. |
| Sensor noise | Median-filter residual energy and spatial uniformity — diffusion output is over-smooth and unnaturally uniform. |
| ML classifier | A HuggingFace image-classification model (default `Organika/sdxl-detector`) on CPU, or GPU when free. |

### Video (provenance-strong, content-humble)

Analyzed asynchronously (`status: processing` → poll `/api/analysis/{id}`).

| Signal | How |
|---|---|
| Container provenance | ffprobe tags + byte scan: C2PA/JUMBF manifests (Sora embeds them), `trainedAlgorithmicMedia`, generator names. Near-definitive. |
| Capture metadata | QuickTime/Android make/model/GPS tags. |
| Frame classifier ensemble | Image classifier over sampled frames. **Capped and low-weighted** until you train the in-domain model (below) — image models are out-of-domain on compressed video frames. |
| Temporal noise coherence | Adjacent-frame noise-residual correlation on consecutive-frame bursts. |
| Audio / duration | Weak circumstantial priors. |

Without provenance, content-only video verdicts deliberately hover mid-range
with low confidence rather than committing on unreliable signals.
`training/` contains a complete pipeline (corpus assembly from public
generator-labeled datasets + your own feedback ledger, clip-level splits,
fine-tuning, per-generator evaluation) to train a proper video-frame
classifier; point `VIDEO_FRAME_MODEL` at the result to let the frame signal
speak at full weight.

## Feedback → training data

Every analysis (content, full signal breakdown, verdict, model versions) is
stored in SQLite; the portal asks users whether they *know* the ground truth
(`human / ai / mixed / unsure` + source hint). `GET /api/export.jsonl`
(Bearer `EXPORT_TOKEN`) dumps labelled rows for recalibration or fine-tuning.

## API

```
POST /api/analyze/text     {"text": "..."}           → {id, percent, confidence, signals[]}
POST /api/analyze/file     multipart file            → same (document | image | video async)
POST /api/feedback         {analysis_id, ground_truth, source_hint?, comment?}
GET  /api/analysis/{id}    re-fetch a verdict / poll a video job
GET  /api/stats            counters
GET  /api/export.jsonl     labelled dataset (auth)
GET  /api/health

GET  /r/{id}               shareable verdict permalink (server-rendered, OG tags)
GET  /og/{id}.png          social-unfurl card image (gauge + percent + stamp)
GET  /content/{id}         the analyzed specimen — served for IMAGES only;
                           submitted text, documents and video are never exposed
```

Analysis ids are unguessable (64-bit); a verdict is only visible to someone
given its link. Ground-truth feedback is accepted **once** per analysis
(enforced by a unique index; duplicates get HTTP 409).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # add --extra-index-url https://download.pytorch.org/whl/cpu for CPU-only torch
cp .env.example .env                              # point it at your LLM endpoint
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Then put it behind whatever you normally use (nginx, Caddy, a Cloudflare
Tunnel, ...). The app binds localhost and trusts `CF-Connecting-IP` for rate
limiting when proxied through Cloudflare.

**LLM endpoint requirements** (`ROUTER_BASE_URL`, OpenAI-compatible):

- The **judge** signal needs only `/chat/completions` — any backend works.
- The **perplexity/rank** signals need `/completions` with
  `echo=true, max_tokens=0, logprobs=N` returning vLLM-style `prompt_logprobs`
  — vLLM supports this natively (directly or through LiteLLM / Open WebUI's
  OpenAI-compatible passthrough). On backends without it those signals
  gracefully report "unavailable" and the rest of the ensemble still works.
- `ffmpeg`/`ffprobe` must be on PATH for video analysis.

**GPU (optional).** Everything runs CPU-only. If CUDA torch is installed, the
classifiers use the GPU only when at least `GPU_MIN_FREE_MB` VRAM is free — so
the app can share a card with other workloads — fall back to CPU on OOM, and
evict themselves after `GPU_IDLE_EVICT_S` idle.

## Calibration & honesty

`scripts/calibrate.py` runs live known-AI generations and known-human corpora
through the pipeline and prints per-signal metrics — used to set the sigmoid
centers in `app/detectors/`. Known failure modes, shown honestly in the UI:

- Adversarial "write like a human, add typos" prompts can drop under 50%.
- Canonical pre-LLM text the scorer has memorized reads as low-perplexity;
  rhythm signals and the judge usually pull it back.
- Heavily edited AI text and non-English text reduce signal quality.
- Content-only video detection is genuinely hard; provenance is the strong
  signal until you train the frame classifier.
- Scores near 50% mean *uncertain*, not "half AI".

## License

MIT
