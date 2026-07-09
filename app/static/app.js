/* DETECTOR portal — vanilla JS. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = { mode: "text", file: null, analysisId: null, truth: null, busy: false };

const SCAN_LINES = {
  text: [
    "tokenizing specimen…",
    "scoring tokens under qwen3-30b-tq…",
    "cross-examining with gpt-oss-120b…",
    "measuring burstiness + rank profile…",
    "consulting the judge…",
    "fusing signals…",
  ],
  document: [
    "extracting text from document…",
    "reading producer metadata…",
    "scoring tokens under qwen3-30b-tq…",
    "cross-examining with gpt-oss-120b…",
    "consulting the judge…",
    "fusing signals…",
  ],
  image: [
    "parsing container metadata…",
    "searching for C2PA / generator fingerprints…",
    "computing frequency spectrum…",
    "measuring sensor-noise residual…",
    "running ML classifier…",
    "fusing signals…",
  ],
  video: [
    "probing container metadata…",
    "searching for generator fingerprints…",
    "extracting sampled frames…",
    "classifying frames (GPU when free)…",
    "measuring temporal noise coherence…",
    "running frame forensics…",
    "fusing signals — long videos take a minute…",
  ],
};

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.mode = tab.dataset.mode;
    state.file = null;
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    ["text", "document", "image", "video"].forEach((m) =>
      $(`panel-${m}`).classList.toggle("hidden", m !== state.mode)
    );
    $("file-chip").classList.add("hidden");
    $("image-preview").classList.add("hidden");
    refreshButton();
  });
});

/* ---------------- inputs ---------------- */
const textInput = $("text-input");
textInput.addEventListener("input", () => {
  $("char-count").textContent = `${textInput.value.length.toLocaleString()} chars`;
  refreshButton();
});

function wireDropzone(zoneId, inputId) {
  const zone = $(zoneId);
  const input = $(inputId);
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  ["dragover", "dragenter"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("armed"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, () => zone.classList.remove("armed"))
  );
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) takeFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) takeFile(input.files[0]);
  });
}
wireDropzone("drop-document", "file-document");
wireDropzone("drop-image", "file-image");
wireDropzone("drop-video", "file-video");

function takeFile(file) {
  const maxMb = state.mode === "video" ? 100 : 25;
  if (file.size > maxMb * 1024 * 1024)
    return showError(`File too large — ${maxMb} MB max.`);
  state.file = file;
  $("file-name").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  $("file-chip").classList.remove("hidden");
  if (state.mode === "image") {
    const preview = $("image-preview");
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
  }
  hideError();
  refreshButton();
}

$("file-clear").addEventListener("click", () => {
  state.file = null;
  $("file-chip").classList.add("hidden");
  $("image-preview").classList.add("hidden");
  refreshButton();
});

function refreshButton() {
  const btn = $("analyze-btn");
  let ready = false, sub = "awaiting specimen";
  if (state.mode === "text") {
    const n = textInput.value.trim().length;
    ready = n >= 120 && n <= 60000;
    sub = n === 0 ? "awaiting specimen"
        : n < 120 ? `${120 - n} more characters needed`
        : `${n.toLocaleString()} characters ready`;
  } else if (state.file) {
    ready = true;
    sub = `${state.file.name} ready`;
  }
  btn.disabled = !ready || state.busy;
  $("btn-sub").textContent = state.busy ? "analyzing…" : sub;
}

/* ---------------- analysis ---------------- */
$("analyze-btn").addEventListener("click", runAnalysis);

let scanTimer = null;
function startScanner() {
  const lines = SCAN_LINES[state.mode];
  let i = 0;
  $("scanner").classList.remove("hidden");
  $("scan-status").textContent = lines[0];
  scanTimer = setInterval(() => {
    i = Math.min(i + 1, lines.length - 1);
    $("scan-status").textContent = lines[i];
  }, 2200);
}
function stopScanner() {
  clearInterval(scanTimer);
  $("scanner").classList.add("hidden");
}

async function runAnalysis() {
  hideError();
  state.busy = true;
  refreshButton();
  startScanner();
  const t0 = performance.now();
  try {
    let resp;
    if (state.mode === "text") {
      resp = await fetch("/api/analyze/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput.value.trim() }),
      });
    } else {
      const form = new FormData();
      form.append("file", state.file);
      resp = await fetch("/api/analyze/file", { method: "POST", body: form });
    }
    let data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `Analysis failed (${resp.status})`);
    if (data.status === "processing") {
      localStorage.setItem("aidet-pending", JSON.stringify({ id: data.id, t: Date.now() }));
      data = await pollJob(data.id);
    }
    renderVerdict(data, performance.now() - t0);
    loadStats();
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    state.busy = false;
    stopScanner();
    refreshButton();
  }
}

/* Poll until the server reaches a terminal state. No client-side deadline:
   background tabs get their timers throttled, so a fixed timeout falsely
   reported "timed out" on jobs that had long since finished. The pending id
   is persisted so a reload (or reopened phone) recovers the verdict. */
async function pollJob(id) {
  const t0 = Date.now();
  let misses = 0;
  for (;;) {
    await new Promise((r) => setTimeout(r, 2500));
    let data;
    try {
      const resp = await fetch(`/api/analysis/${id}`);
      if (resp.status === 404) throw new Error("Analysis not found — it may have been cleaned up.");
      if (!resp.ok) throw { transient: true };
      data = await resp.json();
      misses = 0;
    } catch (err) {
      if (err instanceof Error) throw err;
      if (++misses >= 8) throw new Error("Connection lost while waiting — reload to resume.");
      continue;
    }
    if (data.status === "error") {
      localStorage.removeItem("aidet-pending");
      throw new Error(data.error || "Video analysis failed.");
    }
    if (data.status === "done" || data.status === undefined) {
      localStorage.removeItem("aidet-pending");
      return data;
    }
    const s = Math.round((Date.now() - t0) / 1000);
    $("scan-status").textContent =
      `${$("scan-status").textContent.split(" · ")[0]} · ${s}s elapsed`;
  }
}

/* Recover a verdict if the tab was closed/asleep while a job ran. */
(async function resumePending() {
  try {
    const pending = JSON.parse(localStorage.getItem("aidet-pending") || "null");
    if (!pending || Date.now() - pending.t > 30 * 60 * 1000) return;
    state.mode = "video";
    state.busy = true;
    refreshButton();
    startScanner();
    $("scan-status").textContent = "reconnecting to your running analysis…";
    const data = await pollJob(pending.id);
    renderVerdict(data, Date.now() - pending.t);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    state.busy = false;
    stopScanner();
    state.mode = "text";
    refreshButton();
  }
})();

/* ---------------- verdict rendering ---------------- */
function stampFor(pct) {
  if (pct >= 78) return ["LIKELY SYNTHETIC", "ai"];
  if (pct >= 58) return ["LEANING SYNTHETIC", "ai"];
  if (pct > 42) return ["INCONCLUSIVE", "mid"];
  if (pct > 22) return ["LEANING HUMAN", "human"];
  return ["LIKELY HUMAN", "human"];
}

function renderVerdict(data, elapsedMs) {
  state.analysisId = data.id;
  state.truth = null;
  $("verdict-idle").classList.add("hidden");
  $("verdict-live").classList.remove("hidden");

  const pct = data.percent ?? 50;

  // gauge
  $("gauge-arc").style.strokeDashoffset = String(100 - pct);
  $("needle").setAttribute("transform", `rotate(${-90 + (pct / 100) * 180} 100 100)`);

  // counting number
  animateNumber($("pct-num"), pct);

  const [label, cls] = stampFor(pct);
  const stamp = $("stamp");
  stamp.textContent = label;
  stamp.className = `stamp ${cls}`;

  $("conf").textContent = data.confidence || "—";
  $("took").textContent = `${(elapsedMs / 1000).toFixed(1)}s`;

  // evidence rows
  const list = $("signal-list");
  list.innerHTML = "";
  const signals = data.signals || [];
  const live = signals.filter((s) => s.score !== null);
  $("evidence-count").textContent = `· ${live.length} live / ${signals.length} signals`;
  signals
    .slice()
    .sort((a, b) => (b.score === null ? -1 : b.score) - (a.score === null ? -1 : a.score))
    .sort((a, b) => (a.score === null) - (b.score === null))
    .forEach((s, i) => {
      const li = document.createElement("li");
      li.style.setProperty("--i", i);
      const name = document.createElement("span");
      name.className = "sig-name";
      name.textContent = s.label || s.name;
      li.appendChild(name);
      if (s.score === null) {
        const na = document.createElement("span");
        na.className = "sig-na";
        na.textContent = "n/a";
        li.appendChild(na);
      } else {
        const meter = document.createElement("span");
        meter.className = "sig-meter";
        const fill = document.createElement("b");
        fill.style.setProperty("--v", s.score);
        fill.style.setProperty("--fill", s.score >= 0.5 ? "var(--ai)" : "var(--human)");
        meter.appendChild(fill);
        li.appendChild(meter);
      }
      const detail = document.createElement("span");
      detail.className = "sig-detail";
      detail.textContent = s.detail || "";
      li.appendChild(detail);
      list.appendChild(li);
    });

  // reset feedback widget
  $("feedback-done").classList.add("hidden");
  $("feedback-extra").classList.add("hidden");
  $("stamp-buttons").classList.remove("hidden");
  $("feedback-box").querySelectorAll("button").forEach((b) => b.classList.remove("picked"));
  $("source-hint").value = "";

  $("verdict-live").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function animateNumber(el, target) {
  const dur = 1000, t0 = performance.now();
  const step = (t) => {
    const k = Math.min((t - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = (target * eased).toFixed(target < 10 ? 1 : 0);
    if (k < 1) requestAnimationFrame(step);
    else el.textContent = String(target);
  };
  requestAnimationFrame(step);
}

/* ---------------- feedback ---------------- */
$("stamp-buttons").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-truth]");
  if (!btn) return;
  state.truth = btn.dataset.truth;
  $("stamp-buttons").querySelectorAll("button").forEach((b) =>
    b.classList.toggle("picked", b === btn)
  );
  $("feedback-extra").classList.remove("hidden");
  $("source-hint").focus();
});

$("feedback-send").addEventListener("click", async () => {
  if (!state.analysisId || !state.truth) return;
  try {
    const resp = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        ground_truth: state.truth,
        source_hint: $("source-hint").value.trim() || null,
      }),
    });
    if (!resp.ok) throw new Error("feedback failed");
    $("feedback-extra").classList.add("hidden");
    $("stamp-buttons").classList.add("hidden");
    $("feedback-done").classList.remove("hidden");
    loadStats();
  } catch {
    showError("Could not record feedback — try again.");
  }
});

/* ---------------- misc ---------------- */
function showError(msg) {
  const box = $("error-box");
  box.textContent = msg;
  box.classList.remove("hidden");
}
function hideError() { $("error-box").classList.add("hidden"); }

async function loadStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    $("stat-line").textContent =
      `${s.analyses.toLocaleString()} analyses · ${s.feedback.toLocaleString()} ground-truth labels`;
  } catch { /* footer stat is cosmetic */ }
}
loadStats();
refreshButton();
