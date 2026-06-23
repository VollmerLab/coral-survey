/* Coral Survey Review — canvas + SAM interaction */

const API = "";  // same origin

let state = {
  sessionId: null,
  corals: [],
  idx: 0,
  imgEl: null,          // HTMLImageElement
  imgW: 0, imgH: 0,    // natural dimensions
  canvasW: 0, canvasH: 0,
  scale: 1,
  offsetX: 0, offsetY: 0,
  currentMaskData: null, // ImageData for current mask
  promptPoints: [],      // [{x, y, label}]
  qualityFlag: "ok",
  loading: false,
};

const canvas = document.getElementById("main-canvas");
const ctx = canvas.getContext("2d");

// ── Session loader ──────────────────────────────────────────────

export async function loadSessions() {
  const resp = await fetch(`${API}/api/sessions`);
  const sessions = await resp.json();
  const sel = document.getElementById("session-select");
  sel.innerHTML = '<option value="">— select session —</option>';
  for (const s of sessions) {
    const opt = document.createElement("option");
    opt.value = s.id;
    const confirmed = s.confirmed ?? 0;
    const total = s.total ?? 0;
    opt.textContent = `${s.name} (${confirmed}/${total})`;
    sel.appendChild(opt);
  }
}

export async function selectSession(sessionId) {
  state.sessionId = sessionId;
  if (!sessionId) { clearCanvas(); return; }
  const resp = await fetch(`${API}/api/sessions/${sessionId}`);
  const data = await resp.json();
  state.corals = data.corals || [];
  state.idx = 0;
  renderSidebar();
  if (state.corals.length > 0) loadCoral(0);
}

// ── Sidebar ─────────────────────────────────────────────────────

function renderSidebar() {
  const list = document.getElementById("coral-list");
  list.innerHTML = "";
  state.corals.forEach((c, i) => {
    const item = document.createElement("div");
    item.className = "coral-item" + (i === state.idx ? " active" : "");
    item.innerHTML = `
      <div class="status-dot ${c.status}"></div>
      <div class="coral-name">${c.genotype_id || "—"} <span style="color:var(--text-dim);font-size:0.75rem">${c.species || ""}</span></div>
      <div class="coral-idx">${i + 1}</div>
    `;
    item.onclick = () => loadCoral(i);
    list.appendChild(item);
  });
  document.getElementById("progress-label").textContent =
    `${state.corals.filter(c => c.status === "confirmed").length} / ${state.corals.length} confirmed`;
}

// ── Image + mask loading ─────────────────────────────────────────

async function loadCoral(idx) {
  state.idx = idx;
  const coral = state.corals[idx];
  if (!coral) return;

  state.promptPoints = [];
  state.currentMaskData = null;
  clearRightPanel();
  document.getElementById("geno-input").value = coral.genotype_id || "";
  document.getElementById("species-input").value = coral.species || "";
  setQuality("ok");

  document.querySelectorAll(".coral-item").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
  });

  document.getElementById("btn-confirm").disabled = false;
  document.getElementById("btn-skip").disabled = false;

  // Load thumbnail
  const imgUrl = `${API}/api/coral/${coral.id}/thumb`;
  const img = new Image();
  img.onload = () => {
    state.imgEl = img;
    state.imgW = img.naturalWidth;
    state.imgH = img.naturalHeight;
    fitCanvas();
    drawScene();
    loadAutoMask(coral.id);
  };
  img.src = imgUrl;
}

function fitCanvas() {
  const wrap = document.getElementById("canvas-wrap");
  const maxW = wrap.clientWidth;
  const maxH = wrap.clientHeight;
  const scaleW = maxW / state.imgW;
  const scaleH = maxH / state.imgH;
  state.scale = Math.min(scaleW, scaleH, 1);
  state.canvasW = Math.round(state.imgW * state.scale);
  state.canvasH = Math.round(state.imgH * state.scale);
  canvas.width = state.canvasW;
  canvas.height = state.canvasH;
  state.offsetX = 0;
  state.offsetY = 0;
}

async function loadAutoMask(coralId) {
  setLoading(true);
  try {
    const resp = await fetch(`${API}/api/coral/${coralId}/auto_mask`);
    const data = await resp.json();
    if (data.masks && data.masks.length > 0) {
      await applyMaskB64(data.masks[0].mask_b64);
    }
  } finally {
    setLoading(false);
  }
}

async function applyMaskB64(b64) {
  const blob = await (await fetch("data:image/png;base64," + b64)).blob();
  const bmp = await createImageBitmap(blob);
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = bmp.width;
  tmpCanvas.height = bmp.height;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(bmp, 0, 0);
  state.currentMaskData = tmpCtx.getImageData(0, 0, bmp.width, bmp.height);
  drawScene();
  return b64;
}

// ── Drawing ──────────────────────────────────────────────────────

function drawScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.imgEl) return;
  ctx.drawImage(state.imgEl, 0, 0, state.canvasW, state.canvasH);

  if (state.currentMaskData) {
    drawMaskOverlay();
  }

  // Draw prompt points
  for (const pt of state.promptPoints) {
    ctx.beginPath();
    ctx.arc(pt.x * state.scale, pt.y * state.scale, 6, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 1 ? "rgba(46,204,113,0.9)" : "rgba(231,76,60,0.9)";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function drawMaskOverlay() {
  const md = state.currentMaskData;
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = md.width;
  tmpCanvas.height = md.height;
  const tmpCtx = tmpCanvas.getContext("2d");
  const out = tmpCtx.createImageData(md.width, md.height);
  for (let i = 0; i < md.data.length; i += 4) {
    const v = md.data[i];
    if (v > 127) {
      out.data[i]   = 83;
      out.data[i+1] = 192;
      out.data[i+2] = 240;
      out.data[i+3] = 120;
    }
  }
  tmpCtx.putImageData(out, 0, 0);
  ctx.save();
  ctx.drawImage(tmpCanvas, 0, 0, state.canvasW, state.canvasH);
  ctx.restore();
}

// ── Canvas interaction ────────────────────────────────────────────

canvas.addEventListener("click", async (e) => {
  if (state.loading) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const imgX = Math.round(cx / state.scale);
  const imgY = Math.round(cy / state.scale);
  const label = e.shiftKey ? 0 : 1;  // shift-click = background
  state.promptPoints.push({ x: imgX, y: imgY, label });
  drawScene();
  await sendPrompt();
});

canvas.addEventListener("contextmenu", async (e) => {
  e.preventDefault();
  if (state.loading || state.promptPoints.length === 0) return;
  state.promptPoints.pop();
  drawScene();
  if (state.promptPoints.length > 0) await sendPrompt();
  else { state.currentMaskData = null; drawScene(); }
});

async function sendPrompt() {
  const coral = state.corals[state.idx];
  if (!coral || state.promptPoints.length === 0) return;
  setLoading(true);
  try {
    const resp = await fetch(`${API}/api/coral/${coral.id}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        points: state.promptPoints.map(p => [p.x, p.y]),
        labels: state.promptPoints.map(p => p.label),
      }),
    });
    const data = await resp.json();
    if (data.mask_b64) await applyMaskB64(data.mask_b64);
  } finally {
    setLoading(false);
  }
}

// ── Confirm / Skip ────────────────────────────────────────────────

export async function confirmCoral() {
  const coral = state.corals[state.idx];
  if (!coral || state.loading) return;
  if (!state.currentMaskData) {
    alert("No mask — click on the coral first, or use auto-mask.");
    return;
  }
  setLoading(true);
  try {
    const b64 = await maskDataToB64(state.currentMaskData);
    const resp = await fetch(`${API}/api/coral/${coral.id}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mask_b64: b64,
        quality_flag: state.qualityFlag,
        notes: document.getElementById("notes-input").value,
        genotype_id: document.getElementById("geno-input").value,
        species: document.getElementById("species-input").value,
        scale_mm_px: null,
        whibal_correction: null,
      }),
    });
    const result = await resp.json();
    coral.status = "confirmed";
    updateMetrics(result);
    renderSidebar();
    advanceNext();
  } finally {
    setLoading(false);
  }
}

export async function skipCoral() {
  const coral = state.corals[state.idx];
  if (!coral || state.loading) return;
  await fetch(`${API}/api/coral/${coral.id}/skip`, { method: "POST" });
  coral.status = "skipped";
  renderSidebar();
  advanceNext();
}

function advanceNext() {
  const next = state.corals.findIndex((c, i) => i > state.idx && c.status === "pending");
  if (next !== -1) loadCoral(next);
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  state.imgEl = null;
  state.currentMaskData = null;
  state.promptPoints = [];
}

export function clearPoints() {
  state.promptPoints = [];
  if (state.corals.length && state.corals[state.idx]) {
    loadAutoMask(state.corals[state.idx].id);
  } else {
    state.currentMaskData = null;
    drawScene();
  }
}

// ── Quality flag ──────────────────────────────────────────────────

function setQuality(flag) {
  state.qualityFlag = flag;
  document.querySelectorAll(".quality-btn").forEach(btn => {
    btn.className = "quality-btn" + (btn.dataset.flag === flag ? ` selected-${flag === "ok" ? "ok" : flag === "uncertain" ? "warn" : "bad"}` : "");
  });
}

// ── Metrics display ───────────────────────────────────────────────

function updateMetrics(result) {
  document.getElementById("m-l").textContent  = result.l_mean?.toFixed(1)  ?? "—";
  document.getElementById("m-a").textContent  = result.a_mean?.toFixed(1)  ?? "—";
  document.getElementById("m-b").textContent  = result.b_mean?.toFixed(1)  ?? "—";
  document.getElementById("m-area").textContent = result.area_cm2?.toFixed(2) ?? "—";
}

function clearRightPanel() {
  ["m-l","m-a","m-b","m-area"].forEach(id => {
    document.getElementById(id).textContent = "—";
  });
  document.getElementById("notes-input").value = "";
}

// ── Utility ───────────────────────────────────────────────────────

async function maskDataToB64(imgData) {
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = imgData.width;
  tmpCanvas.height = imgData.height;
  const tmpCtx = tmpCanvas.getContext("2d");
  // convert overlay back to binary PNG
  const out = tmpCtx.createImageData(imgData.width, imgData.height);
  for (let i = 0; i < imgData.data.length; i += 4) {
    const v = imgData.data[i+3] > 64 ? 255 : 0;
    out.data[i] = v; out.data[i+1] = v; out.data[i+2] = v; out.data[i+3] = 255;
  }
  tmpCtx.putImageData(out, 0, 0);
  return tmpCanvas.toDataURL("image/png").split(",")[1];
}

function setLoading(v) {
  state.loading = v;
  document.getElementById("loading-spinner").classList.toggle("hidden", !v);
}

// ── Import modal ──────────────────────────────────────────────────

export async function importSession() {
  const folder = document.getElementById("import-folder").value.trim();
  const name   = document.getElementById("import-name").value.trim();
  const site   = document.getElementById("import-site").value.trim();
  if (!folder || !name) { alert("Folder path and session name are required."); return; }
  const resp = await fetch(`${API}/api/sessions/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder_path: folder, name, site }),
  });
  const data = await resp.json();
  if (!resp.ok) { alert("Import failed: " + JSON.stringify(data)); return; }
  alert(`Imported ${data.corals_imported} photos. Session ID: ${data.session_id}`);
  closeModal("import-modal");
  await loadSessions();
  document.getElementById("session-select").value = data.session_id;
  selectSession(data.session_id);
}

// ── Training tab ──────────────────────────────────────────────────

export async function refreshTrainingStatus() {
  const resp = await fetch(`${API}/api/training/status`);
  const data = await resp.json();
  document.getElementById("t-confirmed").textContent = data.confirmed_pairs;
  document.getElementById("t-running").textContent = data.training_running ? "Running" : "Idle";
}

export async function startTraining() {
  const resp = await fetch(`${API}/api/training/start`, { method: "POST" });
  const data = await resp.json();
  alert(data.status);
  streamLog();
}

export async function exportDataset() {
  const resp = await fetch(`${API}/api/training/export`, { method: "POST",
    headers: {"Content-Type":"application/json"}, body: JSON.stringify({}) });
  const data = await resp.json();
  alert(`Exported ${data.exported} pairs to ${data.output_dir}`);
}

function streamLog() {
  const box = document.getElementById("log-box");
  box.textContent = "";
  const es = new EventSource(`${API}/api/training/log`);
  es.onmessage = (e) => {
    if (e.data === "[done]") { es.close(); return; }
    box.textContent += e.data + "\n";
    box.scrollTop = box.scrollHeight;
  };
}

// ── Modal helpers ─────────────────────────────────────────────────

export function openModal(id) { document.getElementById(id).classList.add("open"); }
export function closeModal(id) { document.getElementById(id).classList.remove("open"); }

// ── Resize handler ────────────────────────────────────────────────

window.addEventListener("resize", () => {
  if (state.imgEl) { fitCanvas(); drawScene(); }
});

// Export state for inline handlers
window.App = {
  loadSessions, selectSession, confirmCoral, skipCoral, clearPoints,
  setQuality, importSession, openModal, closeModal,
  refreshTrainingStatus, startTraining, exportDataset,
};
