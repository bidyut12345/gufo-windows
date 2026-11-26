#!/usr/bin/env python3
"""Minimal WebUI and reverse proxy for Gufo's OpenAI-compatible Image API.

Serves a self-contained image generation and editing UI on --port (default 7860)
and proxies /v1/*, /sdapi/*, /ready, /health, and /props to the Gufo server
(--backend, default http://127.0.0.1:8080). Uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gufo Image Studio</title>
<style>
:root {
  --bg: #0f1115;
  --panel: #171a21;
  --panel-2: #1e232d;
  --border: #2a3140;
  --fg: #eef2f7;
  --muted: #98a2b3;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --ok: #22c55e;
  --warn: #f59e0b;
  --err: #ef4444;
  --radius: 12px;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f4f5f7;
    --panel: #ffffff;
    --panel-2: #f8fafc;
    --border: #e2e8f0;
    --fg: #0f172a;
    --muted: #64748b;
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 24px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 17px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 11px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: var(--panel-2);
  border: 1px solid var(--border);
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--warn);
}
.dot.ok { background: var(--ok); }
.dot.err { background: var(--err); }
main {
  max-width: 1320px;
  margin: 0 auto;
  padding: 20px 24px 40px;
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 20px;
  align-items: start;
}
@media (max-width: 960px) {
  main { grid-template-columns: 1fr; }
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
}
.tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 4px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 16px;
}
.tab {
  border: 0;
  background: transparent;
  color: var(--muted);
  font: inherit;
  font-weight: 600;
  padding: 8px 12px;
  border-radius: 7px;
  cursor: pointer;
}
.tab.active {
  background: var(--accent);
  color: #fff;
}
label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  margin: 12px 0 5px;
}
label:first-child { margin-top: 0; }
input, textarea, select {
  width: 100%;
  font: inherit;
  color: var(--fg);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
}
textarea {
  min-height: 92px;
  resize: vertical;
}
.row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.row-3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
}
.hint {
  font-size: 11px;
  color: var(--muted);
  margin-top: 4px;
}
.dropzone {
  border: 2px dashed var(--border);
  border-radius: 10px;
  padding: 16px;
  text-align: center;
  background: var(--panel-2);
  cursor: pointer;
  transition: border-color .15s;
}
.dropzone.drag { border-color: var(--accent); }
.thumbs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
  gap: 8px;
  margin-top: 10px;
}
.thumb {
  position: relative;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg);
  aspect-ratio: 1;
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.thumb .idx {
  position: absolute;
  left: 4px;
  top: 4px;
  background: rgba(0,0,0,.72);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 999px;
}
.thumb .rm {
  position: absolute;
  right: 4px;
  top: 4px;
  width: 20px;
  height: 20px;
  border: 0;
  border-radius: 50%;
  background: rgba(0,0,0,.72);
  color: #fff;
  cursor: pointer;
  font-size: 12px;
  line-height: 20px;
  padding: 0;
}
.actions {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}
.btn {
  flex: 1;
  border: 0;
  border-radius: 9px;
  padding: 11px 16px;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  background: var(--accent);
  color: #fff;
}
.btn:hover:not(:disabled) { background: var(--accent-hover); }
.btn:disabled { opacity: .55; cursor: not-allowed; }
.btn.secondary {
  flex: 0 0 auto;
  background: var(--panel-2);
  color: var(--fg);
  border: 1px solid var(--border);
}
.status-bar {
  margin-top: 12px;
  font-size: 13px;
  color: var(--muted);
  min-height: 20px;
}
.status-bar.error { color: var(--err); }
.gallery-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.gallery-header h2 {
  margin: 0;
  font-size: 16px;
}
.empty {
  text-align: center;
  padding: 64px 20px;
  color: var(--muted);
  border: 1px dashed var(--border);
  border-radius: 10px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 14px;
}
.result {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.result img {
  width: 100%;
  display: block;
  background: #000;
  cursor: zoom-in;
}
.result-body {
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.result-prompt {
  font-size: 12px;
  color: var(--fg);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.result-meta {
  font-size: 11px;
  color: var(--muted);
}
.result-actions {
  display: flex;
  gap: 8px;
}
.result-actions button, .result-actions a {
  flex: 1;
  text-align: center;
  text-decoration: none;
  font-size: 12px;
  font-weight: 600;
  padding: 6px 8px;
  border-radius: 7px;
  background: var(--panel);
  color: var(--fg);
  border: 1px solid var(--border);
  cursor: pointer;
}
dialog {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--panel);
  color: var(--fg);
  padding: 10px;
  max-width: 92vw;
  max-height: 92vh;
}
dialog::backdrop { background: rgba(0,0,0,.78); }
dialog img { max-width: 88vw; max-height: 82vh; display: block; border-radius: 8px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>Gufo Image Studio</span>
  </div>
  <div class="badge" id="healthBadge">
    <span class="dot" id="healthDot"></span>
    <span id="healthText">Checking server...</span>
  </div>
</header>

<main>
  <section class="card">
    <div class="tabs">
      <button class="tab active" id="tabGen" type="button">Generate</button>
      <button class="tab" id="tabEdit" type="button">Edit</button>
    </div>

    <form id="studioForm">
      <div id="editSection" style="display:none;">
        <label>Reference Images (up to 10 PNG/JPEG, referred to in upload order)</label>
        <div class="dropzone" id="dropzone">
          <div>Drop images here or click to browse</div>
          <div class="hint">In your prompt, refer to images as "first image", "second image", etc.</div>
          <input id="fileInput" type="file" accept="image/png,image/jpeg" multiple hidden>
        </div>
        <div class="thumbs" id="thumbs"></div>
      </div>

      <div id="genInitSection">
        <label for="initFile">Optional Start Image (img2img)</label>
        <input id="initFile" type="file" accept="image/png,image/jpeg">
      </div>

      <label for="prompt">Prompt</label>
      <textarea id="prompt" placeholder="A red ceramic teapot on a wooden table, warm morning light" required></textarea>

      <label for="negativePrompt">Negative Prompt (used when Guidance Scale &gt; 1)</label>
      <input id="negativePrompt" type="text" placeholder="Optional">

      <div class="row">
        <div>
          <label for="sizePreset">Size</label>
          <select id="sizePreset">
            <option value="auto">auto (1024x1024 or ref ratio)</option>
            <option value="1024x1024" selected>1024x1024 (1:1)</option>
            <option value="1344x768">1344x768 (16:9)</option>
            <option value="768x1344">768x1344 (9:16)</option>
            <option value="1280x720">1280x720 (HD)</option>
            <option value="720x1280">720x1280 (Portrait)</option>
            <option value="1536x1024">1536x1024 (3:2)</option>
            <option value="1024x1536">1024x1536 (2:3)</option>
            <option value="2048x2048">2048x2048 (2K 1:1)</option>
            <option value="2560x1440">2560x1440 (16:9 QHD)</option>
            <option value="2720x1536">2720x1536 (16:9 Max 4MP)</option>
            <option value="custom">Custom...</option>
          </select>
        </div>
        <div id="customSizeWrap" style="display:none;">
          <label for="customSize">Custom WxH (mult. of 32)</label>
          <input id="customSize" type="text" value="1024x1024" placeholder="1024x1024">
        </div>
        <div id="stepsWrap">
          <label for="steps">Steps (2.1 = 40, Turbo = 8)</label>
          <input id="steps" type="number" min="2" max="100" value="8" placeholder="auto">
        </div>
      </div>

      <div class="row-3">
        <div>
          <label for="guidance">Guidance</label>
          <input id="guidance" type="number" min="0" max="100" step="0.5" value="1">
        </div>
        <div>
          <label for="strength">Strength</label>
          <input id="strength" type="number" min="0.01" max="1" step="0.05" value="1">
        </div>
        <div>
          <label for="count">Count (n)</label>
          <input id="count" type="number" min="1" max="10" value="1">
        </div>
      </div>

      <div class="row">
        <div>
          <label for="seed">Seed (empty = random)</label>
          <input id="seed" type="number" min="0" placeholder="Random">
        </div>
        <div>
          <label for="background">Background</label>
          <select id="background">
            <option value="auto" selected>auto</option>
            <option value="opaque">opaque</option>
            <option value="transparent">transparent</option>
          </select>
        </div>
      </div>

      <label for="modelName">Model ID</label>
      <input id="modelName" type="text" value="DEFAULT_MODEL_PLACEHOLDER">

      <div id="loraSection" style="display:none; margin-top:12px;">
        <label>LoRA Adapters</label>
        <div id="loraList" style="display:flex; flex-direction:column; gap:6px;"></div>
      </div>

      <div class="actions">
        <button class="btn" id="runBtn" type="submit">Generate Image</button>
      </div>
      <div class="status-bar" id="statusBar"></div>
    </form>
  </section>

  <section class="card">
    <div class="gallery-header">
      <h2>Results</h2>
      <button class="btn secondary" id="clearBtn" type="button">Clear</button>
    </div>
    <div class="empty" id="emptyState">Generated and edited images will appear here.</div>
    <div class="grid" id="gallery"></div>
  </section>
</main>

<dialog id="lightbox">
  <img id="lightboxImg" alt="Preview">
</dialog>

<script>
"use strict";
const $ = (id) => document.getElementById(id);

let mode = "generate";
let editFiles = [];
let busy = false;

function setMode(next) {
  mode = next;
  $("tabGen").classList.toggle("active", mode === "generate");
  $("tabEdit").classList.toggle("active", mode === "edit");
  $("editSection").style.display = mode === "edit" ? "block" : "none";
  $("genInitSection").style.display = mode === "generate" ? "block" : "none";
  $("runBtn").textContent = busy
    ? (mode === "edit" ? "Editing..." : "Generating...")
    : (mode === "edit" ? "Edit Image" : "Generate Image");
}

$("tabGen").onclick = () => setMode("generate");
$("tabEdit").onclick = () => setMode("edit");

$("sizePreset").onchange = () => {
  $("customSizeWrap").style.display = $("sizePreset").value === "custom" ? "block" : "none";
};

function renderThumbs() {
  const wrap = $("thumbs");
  wrap.innerHTML = "";
  editFiles.forEach((file, idx) => {
    const div = document.createElement("div");
    div.className = "thumb";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src);
    const badge = document.createElement("span");
    badge.className = "idx";
    badge.textContent = String(idx + 1);
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "rm";
    rm.textContent = "x";
    rm.onclick = (e) => {
      e.stopPropagation();
      editFiles.splice(idx, 1);
      renderThumbs();
    };
    div.append(img, badge, rm);
    wrap.appendChild(div);
  });
}

function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    if (editFiles.length >= 10) break;
    editFiles.push(f);
  }
  renderThumbs();
}

const dz = $("dropzone");
dz.onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => {
  addFiles(e.target.files);
  e.target.value = "";
};
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave = () => dz.classList.remove("drag");
dz.ondrop = (e) => {
  e.preventDefault();
  dz.classList.remove("drag");
  addFiles(e.dataTransfer.files);
};

function syncDefaultSteps(modelId) {
  if (!$("steps").dataset.touched) {
    $("steps").value = /turbo/i.test(modelId || "") ? "8" : "40";
  }
}
syncDefaultSteps($("modelName").value);

let lorasLoaded = false;

async function loadLoras() {
  if (lorasLoaded) return;
  try {
    const r = await fetch("/sdapi/v1/loras");
    if (!r.ok) return;
    const list = await r.json();
    lorasLoaded = true;
    const wrap = $("loraList");
    wrap.innerHTML = "";
    if (!Array.isArray(list) || list.length === 0) {
      $("loraSection").style.display = "none";
      return;
    }
    $("loraSection").style.display = "block";
    for (const item of list) {
      const name = item.name || item.alias;
      if (!name) continue;
      const row = document.createElement("div");
      row.style.cssText = "display:flex; align-items:center; gap:8px; background:var(--panel-2); border:1px solid var(--border); border-radius:8px; padding:6px 10px;";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.style.width = "auto";
      cb.dataset.loraName = name;
      cb.checked = true;
      const lbl = document.createElement("span");
      lbl.textContent = name;
      lbl.style.cssText = "flex:1; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";
      const sc = document.createElement("input");
      sc.type = "number";
      sc.step = "0.05";
      sc.min = "-10";
      sc.max = "10";
      sc.value = "0.8";
      sc.style.width = "72px";
      sc.title = "LoRA scale";
      row.append(cb, lbl, sc);
      wrap.appendChild(row);
    }
  } catch {}
}

function selectedLoras() {
  if (!lorasLoaded) return null;
  const rows = $("loraList").children;
  if (!rows.length) return null;
  const out = [];
  for (const row of rows) {
    const cb = row.querySelector("input[type=checkbox]");
    const sc = row.querySelector("input[type=number]");
    if (cb && cb.checked) {
      out.push({ name: cb.dataset.loraName, scale: parseFloat(sc.value) || 0.8 });
    }
  }
  return out;
}

async function pollHealth() {
  try {
    const r = await fetch("/ready");
    if (r.ok) {
      $("healthDot").className = "dot ok";
      loadLoras();
      let modelInfo = "";
      try {
        const mr = await fetch("/v1/models");
        if (mr.ok) {
          const mj = await mr.json();
          if (mj.data && mj.data.length > 0) {
            const id = mj.data[0].id;
            modelInfo = id;
            if (!$("modelName").dataset.touched) {
              $("modelName").value = id;
              syncDefaultSteps(id);
            }
          }
        }
      } catch (_) {}
      $("healthText").textContent = modelInfo ? `Ready: ${modelInfo}` : "Server ready";
    } else {
      $("healthDot").className = "dot";
      $("healthText").textContent = "Loading model...";
    }
  } catch (_) {
    $("healthDot").className = "dot err";
    $("healthText").textContent = "Server offline";
  }
}
$("steps").oninput = () => { $("steps").dataset.touched = "1"; };
$("modelName").oninput = () => { $("modelName").dataset.touched = "1"; };
pollHealth();
setInterval(pollHealth, 3000);

function getSizeValue() {
  const p = $("sizePreset").value;
  return p === "custom" ? $("customSize").value.trim() : p;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function dataUrlToFile(dataUrl, name) {
  const [meta, b64] = dataUrl.split(",");
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new File([arr], name, { type: "image/png" });
}

function addResult(b64, promptText, metaText) {
  $("emptyState").style.display = "none";
  const url = `data:image/png;base64,${b64}`;
  const card = document.createElement("div");
  card.className = "result";

  const img = document.createElement("img");
  img.src = url;
  img.alt = promptText;
  img.onclick = () => {
    $("lightboxImg").src = url;
    $("lightbox").showModal();
  };

  const body = document.createElement("div");
  body.className = "result-body";

  const p = document.createElement("div");
  p.className = "result-prompt";
  p.textContent = promptText;

  const meta = document.createElement("div");
  meta.className = "result-meta";
  meta.textContent = metaText;

  const actions = document.createElement("div");
  actions.className = "result-actions";

  const dl = document.createElement("a");
  dl.href = url;
  dl.download = `gufo-${Date.now()}.png`;
  dl.textContent = "Download";

  const toEdit = document.createElement("button");
  toEdit.type = "button";
  toEdit.textContent = "Send to Edit";
  toEdit.onclick = () => {
    addFiles([dataUrlToFile(url, `generated-${Date.now()}.png`)]);
    setMode("edit");
  };

  actions.append(dl, toEdit);
  body.append(p, meta, actions);
  card.append(img, body);
  $("gallery").prepend(card);
}

$("lightbox").onclick = () => $("lightbox").close();
$("clearBtn").onclick = () => {
  $("gallery").innerHTML = "";
  $("emptyState").style.display = "block";
};

$("studioForm").onsubmit = async (e) => {
  e.preventDefault();
  if (busy) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) return;

  const model = $("modelName").value.trim() || "qwen-image-2.1-turbo";
  const size = getSizeValue();
  const stepsStr = $("steps").value.trim();
  const guidance = parseFloat($("guidance").value || "1");
  const strength = parseFloat($("strength").value || "1");
  const count = parseInt($("count").value || "1", 10);
  const seedStr = $("seed").value.trim();
  const neg = $("negativePrompt").value.trim();
  const bg = $("background").value;

  if (mode === "edit" && editFiles.length === 0) {
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = "Add at least one reference image to edit.";
    return;
  }

  busy = true;
  $("runBtn").disabled = true;
  setMode(mode);
  $("statusBar").className = "status-bar";
  $("statusBar").textContent = mode === "edit" ? "Running image edit..." : "Generating image...";
  const t0 = performance.now();

  try {
    let resp;
    if (mode === "generate") {
      const payload = {
        model,
        prompt,
        size,
        n: count,
        background: bg,
        guidance_scale: guidance,
      };
      if (stepsStr) payload.steps = parseInt(stepsStr, 10);
      if (seedStr) payload.seed = parseInt(seedStr, 10);
      if (neg) payload.negative_prompt = neg;
      const loras = selectedLoras();
      if (loras !== null) payload.loras = loras;
      const initInput = $("initFile").files[0];
      if (initInput) {
        payload.image = await fileToDataUrl(initInput);
        payload.strength = strength;
      }
      resp = await fetch("/v1/images/generations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } else {
      const form = new FormData();
      form.append("model", model);
      form.append("prompt", prompt);
      form.append("size", size);
      form.append("n", String(count));
      form.append("background", bg);
      form.append("guidance_scale", String(guidance));
      form.append("strength", String(strength));
      if (stepsStr) form.append("steps", stepsStr);
      if (seedStr) form.append("seed", seedStr);
      if (neg) form.append("negative_prompt", neg);
      for (const f of editFiles) form.append("image[]", f);
      resp = await fetch("/v1/images/edits", { method: "POST", body: form });
    }

    const data = await resp.json();
    if (!resp.ok) {
      const msg = (data && data.error && data.error.message) || `HTTP ${resp.status}`;
      throw new Error(msg);
    }

    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    const items = data.data || [];
    const meta = `${mode} | ${size} | ${dt}s` + (seedStr ? ` | seed ${seedStr}` : "");
    for (const item of items) {
      if (item.b64_json) addResult(item.b64_json, prompt, meta);
    }
    $("statusBar").textContent = `Done in ${dt}s (${items.length} image${items.length === 1 ? "" : "s"}).`;
  } catch (err) {
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = `Error: ${err.message}`;
  } finally {
    busy = false;
    $("runBtn").disabled = false;
    setMode(mode);
  }
};
</script>
</body>
</html>
"""


def make_handler(backend_url: str, default_model: str):
    parsed = urllib.parse.urlsplit(backend_url)
    backend_host = parsed.hostname or "127.0.0.1"
    backend_port = parsed.port or (443 if parsed.scheme == "https" else 80)

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            # Keep console output focused on gufo server logs.
            pass

        def _send_html(self) -> None:
            body = HTML_PAGE.replace("DEFAULT_MODEL_PLACEHOLDER", default_model).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else None

            forward_headers = {}
            for k, v in self.headers.items():
                kl = k.lower()
                if kl in ("host", "connection", "keep-alive", "proxy-authenticate",
                          "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"):
                    continue
                forward_headers[k] = v
            forward_headers["Connection"] = "close"

            conn = http.client.HTTPConnection(backend_host, backend_port, timeout=3600)
            try:
                conn.request(self.command, self.path, body=body, headers=forward_headers)
                resp = conn.getresponse()
                resp_body = resp.read()
                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                if resp_body:
                    self.wfile.write(resp_body)
            except OSError as exc:
                err = json.dumps({"error": {"message": f"Backend unreachable: {exc}"}}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            finally:
                conn.close()

        def do_GET(self) -> None:
            if self.path == "/" or self.path == "/index.html":
                self._send_html()
            elif self.path == "/sw.js":
                # Unregister any stale service worker from an earlier app on this port.
                js = b"self.addEventListener('install',()=>self.skipWaiting());self.addEventListener('activate',e=>e.waitUntil(self.registration.unregister()));"
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(js)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(js)
            else:
                self._proxy()

        def do_POST(self) -> None:
            self._proxy()

    return ProxyHandler


def _watch_parent(pid: int) -> None:
    if pid <= 0 or os.name != "nt":
        return
    SYNCHRONIZE = 0x00100000
    INFINITE = 0xFFFFFFFF
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        os._exit(0)
    ctypes.windll.kernel32.WaitForSingleObject(handle, INFINITE)
    ctypes.windll.kernel32.CloseHandle(handle)
    os._exit(0)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Gufo Image Studio WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="qwen-image-2.1-turbo")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if args.parent_pid > 0:
        threading.Thread(target=_watch_parent, args=(args.parent_pid,), daemon=True).start()

    server = ReusableThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.backend, args.model),
    )
    print(f"Gufo Image WebUI listening on http://{args.host}:{args.port} -> {args.backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
