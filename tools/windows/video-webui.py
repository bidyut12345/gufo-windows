#!/usr/bin/env python3
"""Minimal WebUI and reverse proxy for Gufo's OpenAI-compatible Video API.

Serves a self-contained video generation UI on --port (default 8080) and
proxies /v1/*, /ready, and /health to the Gufo video server (--backend,
default http://127.0.0.1:8081). Uses only Python stdlib.
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
<title>Gufo Video Studio (MiniMax H3)</title>
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
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }
.dot.ok { background: var(--ok); }
.dot.err { background: var(--err); }
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px 24px 40px;
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 20px;
  align-items: start;
}
@media (max-width: 960px) { main { grid-template-columns: 1fr; } }
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
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
textarea { min-height: 104px; resize: vertical; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.actions { display: flex; gap: 10px; margin-top: 16px; }
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
.btn.danger { background: var(--err); flex: 0 0 auto; }
.btn.secondary {
  flex: 0 0 auto;
  background: var(--panel-2);
  color: var(--fg);
  border: 1px solid var(--border);
}
.progress-wrap {
  margin-top: 12px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  height: 10px;
  overflow: hidden;
  display: none;
}
.progress-bar {
  height: 100%;
  width: 0%;
  background: var(--accent);
  transition: width .25s;
}
.status-bar { margin-top: 10px; font-size: 13px; color: var(--muted); min-height: 20px; }
.status-bar.error { color: var(--err); }
.empty {
  text-align: center;
  padding: 64px 20px;
  color: var(--muted);
  border: 1px dashed var(--border);
  border-radius: 10px;
}
.video-list { display: flex; flex-direction: column; gap: 16px; }
.video-item {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
.video-item video {
  width: 100%;
  max-height: 540px;
  display: block;
  background: #000;
}
.video-meta {
  padding: 12px 14px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.video-prompt { font-size: 13px; font-weight: 500; }
.video-sub { font-size: 12px; color: var(--muted); }
.dl {
  text-decoration: none;
  font-size: 12px;
  font-weight: 600;
  padding: 6px 12px;
  border-radius: 7px;
  background: var(--panel);
  color: var(--fg);
  border: 1px solid var(--border);
}
.ref-list { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.ref-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.ref-item img {
  width: 40px;
  height: 40px;
  object-fit: cover;
  border-radius: 6px;
  flex-shrink: 0;
}
.ref-info { flex: 1; min-width: 0; }
.ref-tag { font-size: 11px; font-weight: 700; color: var(--accent); }
.ref-name {
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ref-btns { display: flex; gap: 4px; flex-shrink: 0; }
.mini-btn {
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--fg);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  cursor: pointer;
}
.mini-btn:disabled { opacity: .4; cursor: not-allowed; }
.mini-btn.del { color: var(--err); }
.tabs { display: flex; gap: 6px; margin-bottom: 14px; }
.tab {
  flex: 1;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--panel-2);
  color: var(--muted);
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.tab.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.video-preview {
  width: 100%;
  max-height: 180px;
  margin-top: 8px;
  border-radius: 8px;
  background: #000;
  display: none;
}
</style>
</head>
<body>
<header>
  <div class="brand">Gufo Video Studio — MiniMax H3</div>
  <div class="badge">
    <span class="dot" id="healthDot"></span>
    <span id="healthText">Checking server...</span>
  </div>
</header>

<main>
  <section class="card">
    <div class="tabs">
      <button type="button" class="tab active" id="tabGen">Create Video</button>
      <button type="button" class="tab" id="tabEdit">Video Edit / Character Swap</button>
    </div>
    <form id="videoForm">
      <div id="editSection" style="display:none;margin-bottom:12px;">
        <label for="editVideo">Source Video to Edit (&lt;Video 1&gt;, MP4/MOV/WebM)</label>
        <div style="display:flex;gap:8px;align-items:center;">
          <input id="editVideo" type="file" accept="video/mp4,video/webm,video/quicktime" style="flex:1;">
          <button class="btn secondary" id="clearEditVideoBtn" type="button" style="padding:7px 10px;font-size:12px;">Clear</button>
        </div>
        <video id="editVideoPreview" class="video-preview" controls muted></video>
        <div style="margin-top:8px;">
          <button class="btn secondary" id="fillSwapBtn" type="button" style="padding:6px 10px;font-size:12px;">Fill Character Swap Prompt Template</button>
        </div>
      </div>

      <label for="prompt">Prompt (video + audio description)</label>
      <textarea id="prompt" placeholder="A red fox walking through fresh snow in a quiet pine forest, soft crunching footsteps" required></textarea>

      <div class="row">
        <div>
          <label for="resolution">Resolution</label>
          <select id="resolution">
            <option value="512x512" selected>512x512 (1:1 Square)</option>
            <option value="768x768">768x768 (1:1 High)</option>
            <option value="1024x576">1024x576 (16:9 Widescreen)</option>
            <option value="1280x720">1280x720 (16:9 HD)</option>
            <option value="1344x768">1344x768 (16:9 Full Res)</option>
            <option value="576x1024">576x1024 (9:16 Portrait)</option>
            <option value="720x1280">720x1280 (9:16 HD Portrait)</option>
            <option value="768x1344">768x1344 (9:16 Full Portrait)</option>
            <option value="896x672">896x672 (4:3 Landscape)</option>
            <option value="672x896">672x896 (3:4 Portrait)</option>
            <option value="1024x768">1024x768 (4:3 High)</option>
            <option value="768x1024">768x1024 (3:4 High)</option>
            <option value="256x256">256x256 (Dev PPM check)</option>
          </select>
        </div>
        <div>
          <label for="steps">Steps</label>
          <select id="steps">
            <option value="50" selected>50 steps (Exact)</option>
            <option value="30">30 steps (Balanced)</option>
            <option value="20">20 steps (Fast)</option>
            <option value="10">10 steps (Draft)</option>
            <option value="5">5 steps (Dev)</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label for="seconds">Duration (seconds: 1, or 5–14.375)</label>
          <input id="seconds" type="number" min="1" max="14.375" step="0.125" value="5">
        </div>
        <div>
          <label for="seed">Seed (empty = 42)</label>
          <input id="seed" type="number" min="0" placeholder="42">
        </div>
      </div>

      <label for="modelName">Model ID</label>
      <input id="modelName" type="text" value="DEFAULT_MODEL_PLACEHOLDER">

      <div id="keyframeSection">
        <div class="row">
          <div>
            <label for="firstFrame">First Frame (FL2VA, optional)</label>
            <input id="firstFrame" type="file" accept="image/png,image/jpeg,image/webp">
          </div>
          <div>
            <label for="lastFrame">Last Frame (FL2VA, optional)</label>
            <input id="lastFrame" type="file" accept="image/png,image/jpeg,image/webp">
          </div>
        </div>
        <label style="display:flex;align-items:center;gap:8px;margin-top:8px;cursor:pointer;">
          <input id="matchFirstAspect" type="checkbox" style="width:auto;">
          <span>Match first/last frame aspect ratio (off = keep selected Resolution)</span>
        </label>
      </div>
      <label style="display:flex;align-items:center;gap:8px;margin-top:6px;cursor:pointer;">
        <input id="stepReuse" type="checkbox" style="width:auto;">
        <span>DiT step-reuse acceleration (~2x faster: reuse=2, 45/50 blocks)</span>
      </label>

      <label>Reference Images — Characters / Location / Objects (Ref2VA, up to 9)</label>
      <input id="refPicker" type="file" accept="image/png,image/jpeg,image/webp" multiple style="display:none;">
      <div style="display:flex;gap:8px;margin-top:4px;">
        <button class="btn secondary" id="addRefBtn" type="button" style="padding:7px 12px;font-size:12px;">+ Add Reference Image(s)</button>
        <button class="btn secondary" id="clearRefBtn" type="button" style="padding:7px 12px;font-size:12px;">Clear All</button>
      </div>
      <div class="ref-list" id="refList"></div>

      <div class="actions">
        <button class="btn" id="runBtn" type="submit">Generate Video</button>
        <button class="btn secondary" id="pauseBtn" type="button" style="display:none;">Pause</button>
        <button class="btn" id="resumeBtn" type="button" style="display:none;">Resume</button>
        <button class="btn danger" id="cancelBtn" type="button" style="display:none;">Cancel</button>
      </div>
      <div class="progress-wrap" id="progressWrap"><div class="progress-bar" id="progressBar"></div></div>
      <div class="status-bar" id="statusBar"></div>
    </form>
  </section>

  <section class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <h2 style="margin:0;font-size:16px;">Generated Videos</h2>
      <button class="btn secondary" id="clearBtn" type="button">Clear</button>
    </div>
    <div class="empty" id="emptyState">Generated MP4 videos will appear here.</div>
    <div class="video-list" id="gallery"></div>
  </section>
</main>

<script>
"use strict";
const $ = (id) => document.getElementById(id);

let activeJobId = null;
let pollTimer = null;
let elapsedTimer = null;
let jobStartMs = 0;
let lastProgress = 0;
let lastStatusText = "";
let refItems = []; // { file, url }
let activeTab = "gen";
let editVideoUrl = null;

function switchTab(tab) {
  activeTab = tab;
  $("tabGen").className = tab === "gen" ? "tab active" : "tab";
  $("tabEdit").className = tab === "edit" ? "tab active" : "tab";
  $("editSection").style.display = tab === "edit" ? "block" : "none";
  $("keyframeSection").style.display = tab === "gen" ? "block" : "none";
  $("runBtn").textContent = tab === "edit" ? "Edit Video" : "Generate Video";
}
$("tabGen").onclick = () => switchTab("gen");
$("tabEdit").onclick = () => switchTab("edit");

$("editVideo").onchange = (e) => {
  if (editVideoUrl) { URL.revokeObjectURL(editVideoUrl); editVideoUrl = null; }
  const f = e.target.files[0];
  if (f) {
    editVideoUrl = URL.createObjectURL(f);
    $("editVideoPreview").src = editVideoUrl;
    $("editVideoPreview").style.display = "block";
  } else {
    $("editVideoPreview").style.display = "none";
  }
};
$("clearEditVideoBtn").onclick = () => {
  $("editVideo").value = "";
  if (editVideoUrl) { URL.revokeObjectURL(editVideoUrl); editVideoUrl = null; }
  $("editVideoPreview").removeAttribute("src");
  $("editVideoPreview").style.display = "none";
};
$("fillSwapBtn").onclick = () => {
  $("prompt").value = "Replace the main person in <Video 1> with the character from <Picture 1>, keeping the exact camera movement, pose, lighting, and background.";
};

function formatDuration(ms) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function updateTimingStatus() {
  if (!activeJobId || !jobStartMs) return;
  const elapsedMs = Date.now() - jobStartMs;
  const elapsedStr = formatDuration(elapsedMs);
  let etaStr = "calculating...";
  if (lastProgress > 0 && lastProgress < 100) {
    const etaMs = (elapsedMs * (100 - lastProgress)) / lastProgress;
    etaStr = `~${formatDuration(etaMs)}`;
  }
  $("statusBar").textContent = `${lastStatusText} | Elapsed: ${elapsedStr} | ETA: ${etaStr}`;
}

function startElapsedTimer() {
  stopElapsedTimer();
  jobStartMs = Date.now();
  elapsedTimer = setInterval(updateTimingStatus, 1000);
}

function stopElapsedTimer() {
  if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

function renderRefList() {
  const container = $("refList");
  container.innerHTML = "";
  refItems.forEach((item, idx) => {
    const row = document.createElement("div");
    row.className = "ref-item";
    const img = document.createElement("img");
    img.src = item.url;
    const info = document.createElement("div");
    info.className = "ref-info";
    const tag = document.createElement("div");
    tag.className = "ref-tag";
    tag.textContent = `<Picture ${idx + 1}>`;
    const name = document.createElement("div");
    name.className = "ref-name";
    name.textContent = item.file.name;
    info.append(tag, name);

    const btns = document.createElement("div");
    btns.className = "ref-btns";
    const up = document.createElement("button");
    up.type = "button";
    up.className = "mini-btn";
    up.textContent = "↑";
    up.disabled = idx === 0;
    up.onclick = () => {
      [refItems[idx - 1], refItems[idx]] = [refItems[idx], refItems[idx - 1]];
      renderRefList();
    };
    const down = document.createElement("button");
    down.type = "button";
    down.className = "mini-btn";
    down.textContent = "↓";
    down.disabled = idx === refItems.length - 1;
    down.onclick = () => {
      [refItems[idx + 1], refItems[idx]] = [refItems[idx], refItems[idx + 1]];
      renderRefList();
    };
    const del = document.createElement("button");
    del.type = "button";
    del.className = "mini-btn del";
    del.textContent = "×";
    del.onclick = () => {
      URL.revokeObjectURL(item.url);
      refItems.splice(idx, 1);
      renderRefList();
    };
    btns.append(up, down, del);
    row.append(img, info, btns);
    container.appendChild(row);
  });
}

$("addRefBtn").onclick = () => $("refPicker").click();
$("refPicker").onchange = (e) => {
  const files = Array.from(e.target.files || []);
  for (const file of files) {
    if (refItems.length >= 9) break;
    refItems.push({ file, url: URL.createObjectURL(file) });
  }
  e.target.value = "";
  renderRefList();
};
$("clearRefBtn").onclick = () => {
  for (const item of refItems) URL.revokeObjectURL(item.url);
  refItems = [];
  renderRefList();
};

async function pollHealth() {
  try {
    const r = await fetch("/ready");
    if (r.ok) {
      $("healthDot").className = "dot ok";
      $("healthText").textContent = "Server ready";
    } else {
      $("healthDot").className = "dot";
      $("healthText").textContent = "Loading model...";
    }
  } catch (_) {
    $("healthDot").className = "dot err";
    $("healthText").textContent = "Server offline";
  }
}
pollHealth();
setInterval(pollHealth, 3000);

function setBusy(isBusy, isPaused = false) {
  $("runBtn").disabled = isBusy || isPaused;
  $("runBtn").textContent = isPaused ? "Paused" : (isBusy ? "Generating..." : "Generate Video");
  $("pauseBtn").style.display = (isBusy && !isPaused) ? "inline-block" : "none";
  $("resumeBtn").style.display = isPaused ? "inline-block" : "none";
  $("cancelBtn").style.display = (isBusy || isPaused) ? "inline-block" : "none";
  $("progressWrap").style.display = (isBusy || isPaused) ? "block" : "none";
}

function addVideoResult(job, promptText, preset) {
  $("emptyState").style.display = "none";
  const contentUrl = `/v1/videos/${encodeURIComponent(job.id)}/content`;
  const item = document.createElement("div");
  item.className = "video-item";

  if (job.output_format === "mp4") {
    const video = document.createElement("video");
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    video.src = contentUrl;
    item.appendChild(video);
  } else {
    const note = document.createElement("div");
    note.style.padding = "16px";
    note.textContent = `Dev output (${job.output_format}) ready for download.`;
    item.appendChild(note);
  }

  const meta = document.createElement("div");
  meta.className = "video-meta";
  const left = document.createElement("div");
  const p = document.createElement("div");
  p.className = "video-prompt";
  p.textContent = promptText;
  const sub = document.createElement("div");
  sub.className = "video-sub";
  sub.textContent = `${preset} | ${job.size} | ${job.seconds}s | id: ${job.id}`;
  left.append(p, sub);

  const dl = document.createElement("a");
  dl.className = "dl";
  dl.href = contentUrl;
  dl.download = `minimax-h3-${job.id}.${job.output_format || "mp4"}`;
  dl.textContent = `Download ${String(job.output_format || "mp4").toUpperCase()}`;

  meta.append(left, dl);
  item.appendChild(meta);
  $("gallery").prepend(item);
}

$("clearBtn").onclick = () => {
  $("gallery").innerHTML = "";
  $("emptyState").style.display = "block";
};

let currentPollFn = null;

$("pauseBtn").onclick = async () => {
  if (!activeJobId) return;
  try {
    $("pauseBtn").disabled = true;
    await fetch(`/v1/videos/${encodeURIComponent(activeJobId)}/pause`, { method: "POST" });
    $("statusBar").textContent = "Pausing at next checkpoint...";
  } catch (_) {
    $("pauseBtn").disabled = false;
  }
};

$("resumeBtn").onclick = async () => {
  if (!activeJobId) return;
  try {
    $("resumeBtn").disabled = true;
    const resp = await fetch(`/v1/videos/${encodeURIComponent(activeJobId)}/resume`, { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) throw new Error((data && data.error && data.error.message) || `HTTP ${resp.status}`);
    $("pauseBtn").disabled = false;
    $("resumeBtn").disabled = false;
    setBusy(true, false);
    startElapsedTimer();
    if (currentPollFn) currentPollFn();
  } catch (err) {
    $("resumeBtn").disabled = false;
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = `Resume error: ${err.message}`;
  }
};

$("cancelBtn").onclick = async () => {
  if (!activeJobId) return;
  try {
    await fetch(`/v1/videos/${encodeURIComponent(activeJobId)}`, { method: "DELETE" });
    $("statusBar").textContent = "Job cancelled.";
  } catch (_) {}
  if (pollTimer) clearTimeout(pollTimer);
  stopElapsedTimer();
  activeJobId = null;
  currentPollFn = null;
  $("pauseBtn").disabled = false;
  $("resumeBtn").disabled = false;
  setBusy(false, false);
};

$("videoForm").onsubmit = async (e) => {
  e.preventDefault();
  if (activeJobId) return;

  const prompt = $("prompt").value.trim();
  if (!prompt) return;

  const size = $("resolution").value;
  const steps = parseInt($("steps").value, 10);
  const isDev = size === "256x256";
  const preset = isDev ? "dev" : (size === "1344x768" ? "exact-1344x768" : "exact");
  const seconds = $("seconds").value.trim() || "5";
  const seedStr = $("seed").value.trim();
  const model = $("modelName").value.trim() || "minimax-h3";
  const firstFile = activeTab === "gen" ? $("firstFrame").files[0] : null;
  const lastFile = activeTab === "gen" ? $("lastFrame").files[0] : null;
  const editVideoFile = activeTab === "edit" ? $("editVideo").files[0] : null;
  if (activeTab === "edit" && !editVideoFile) {
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = "Error: Please attach a Source Video to edit.";
    return;
  }
  const refFiles = refItems.map((item) => item.file);
  const matchAspect = $("matchFirstAspect").checked;
  const stepReuse = $("stepReuse").checked;

  const gufoOpts = { preset, steps, lock_resolution: !matchAspect, step_reuse: stepReuse };
  if (isDev) gufoOpts.output_format = "ppm";
  if (seedStr) gufoOpts.seed = parseInt(seedStr, 10);

  setBusy(true);
  lastProgress = 0;
  lastStatusText = "Submitting video job...";
  startElapsedTimer();
  $("progressBar").style.width = "0%";
  $("statusBar").className = "status-bar";
  $("statusBar").textContent = lastStatusText;

  try {
    let resp;
    if (firstFile || lastFile || editVideoFile || refFiles.length > 0) {
      const form = new FormData();
      form.append("model", model);
      form.append("prompt", prompt);
      form.append("size", size);
      form.append("seconds", seconds);
      form.append("gufo", JSON.stringify(gufoOpts));
      if (firstFile) form.append("first_frame", firstFile);
      if (lastFile) form.append("last_frame", lastFile);
      if (editVideoFile) form.append("reference_video_0", editVideoFile);
      refFiles.forEach((f, idx) => form.append(`reference_${idx}`, f));
      resp = await fetch("/v1/videos", { method: "POST", body: form });
    } else {
      resp = await fetch("/v1/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt, size, seconds: parseFloat(seconds), gufo: gufoOpts }),
      });
    }

    const job = await resp.json();
    if (!resp.ok) {
      throw new Error((job && job.error && job.error.message) || `HTTP ${resp.status}`);
    }

    activeJobId = job.id;
    const poll = async () => {
      if (!activeJobId) return;
      try {
        const sr = await fetch(`/v1/videos/${encodeURIComponent(activeJobId)}`);
        const sj = await sr.json();
        if (!sr.ok) throw new Error((sj && sj.error && sj.error.message) || `HTTP ${sr.status}`);
        const pct = typeof sj.progress === "number" ? sj.progress : 0;
        lastProgress = pct;
        $("progressBar").style.width = `${pct}%`;
        const stageInfo = sj.checkpoint_stage
          ? ` [${sj.checkpoint_stage}${sj.checkpoint_total ? ` ${sj.checkpoint_step}/${sj.checkpoint_total}` : ""}]`
          : "";
        lastStatusText = pct === 0 && sj.status === "in_progress"
          ? `Job ${sj.id}: in_progress (0%) — encoding prompt & loading weights`
          : `Job ${sj.id}: ${sj.status} (${pct}%)${stageInfo}`;
        updateTimingStatus();

        if (sj.status === "paused") {
          stopElapsedTimer();
          $("pauseBtn").disabled = false;
          setBusy(false, true);
          $("statusBar").textContent = `Paused${stageInfo} (${pct}%). Click Resume to continue.`;
          return;
        }
        if (sj.status === "completed") {
          const totalElapsed = formatDuration(Date.now() - jobStartMs);
          stopElapsedTimer();
          activeJobId = null;
          currentPollFn = null;
          setBusy(false, false);
          $("statusBar").textContent = `Completed (${sj.size}, ${sj.seconds}s) in ${totalElapsed}.`;
          addVideoResult(sj, prompt, `${size} / ${steps} steps${stepReuse ? " / reuse" : ""}`);
          return;
        }
        if (sj.status === "failed") {
          const msg = (sj.error && sj.error.message) || "Video generation failed";
          throw new Error(msg);
        }
        pollTimer = setTimeout(poll, 1500);
      } catch (err) {
        stopElapsedTimer();
        activeJobId = null;
        currentPollFn = null;
        setBusy(false);
        $("statusBar").className = "status-bar error";
        $("statusBar").textContent = `Error: ${err.message}`;
      }
    };
    currentPollFn = poll;
    poll();
  } catch (err) {
    stopElapsedTimer();
    activeJobId = null;
    setBusy(false);
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = `Error: ${err.message}`;
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

            conn = http.client.HTTPConnection(backend_host, backend_port, timeout=600)
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

        def do_DELETE(self) -> None:
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
    parser = argparse.ArgumentParser(description="Gufo Video Studio WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="minimax-h3")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if args.parent_pid > 0:
        threading.Thread(target=_watch_parent, args=(args.parent_pid,), daemon=True).start()

    server = ReusableThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.backend, args.model),
    )
    print(f"Gufo Video WebUI listening on http://{args.host}:{args.port} -> {args.backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
