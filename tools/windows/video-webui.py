#!/usr/bin/env python3
"""Minimal WebUI and reverse proxy for Gufo's OpenAI-compatible Video API.

Serves a self-contained video generation UI on --port (default 8080) and
proxies /v1/*, /ready, and /health to the Gufo video server (--backend,
default http://127.0.0.1:8081). Uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import glob
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
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
.char-list { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
.char-card {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.char-card.selected { border-color: var(--accent); }
.char-head { display: flex; align-items: center; gap: 8px; }
.char-head input[type="checkbox"] { width: auto; margin: 0; }
.char-upload {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px;
  background: var(--panel);
  border: 1px dashed var(--border);
  border-radius: 8px;
}
.char-thumb {
  width: 52px;
  height: 52px;
  object-fit: cover;
  border-radius: 6px;
  background: var(--panel-2);
  flex-shrink: 0;
  display: none;
}
.swap-opts {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  margin-top: 10px;
}
.swap-check {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 5px 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--fg);
  cursor: pointer;
}
.swap-check input[type="checkbox"] { width: auto; margin: 0; }
.small-note { font-size: 11px; color: var(--muted); }
.frame-picker {
  margin-top: 10px;
  padding: 10px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  display: none;
}
.frame-strip {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding-bottom: 6px;
  margin-bottom: 8px;
}
.frame-thumb {
  height: 48px;
  border-radius: 6px;
  border: 2px solid transparent;
  cursor: pointer;
  flex-shrink: 0;
}
.frame-thumb.active { border-color: var(--accent); }
.crop-stage {
  position: relative;
  display: inline-block;
  max-width: 100%;
  user-select: none;
  cursor: crosshair;
}
.crop-stage canvas {
  display: block;
  max-width: 100%;
  height: auto;
  border-radius: 8px;
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
      <button type="button" class="tab" id="tabEdit">Video Edit</button>
      <button type="button" class="tab" id="tabSwap">Character Swap</button>
    </div>
    <form id="videoForm">
      <div id="editSection" style="display:none;margin-bottom:12px;">
        <label for="editVideo">Source Video to Edit (&lt;Video 1&gt;, MP4/MOV/WebM)</label>
        <div style="display:flex;gap:8px;align-items:center;">
          <input id="editVideo" type="file" accept="video/mp4,video/webm,video/quicktime" style="flex:1;">
          <button class="btn secondary" id="clearEditVideoBtn" type="button" style="padding:7px 10px;font-size:12px;">Clear</button>
        </div>
        <video id="editVideoPreview" class="video-preview" controls muted></video>
      </div>

      <div id="swapSection" style="display:none;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
          <label for="swapVideo" style="margin:0;">1. Source Video (&lt;Video 1&gt;)</label>
          <span id="swapVideoMeta" class="small-note"></span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:5px;">
          <input id="swapVideo" type="file" accept="video/mp4,video/webm,video/quicktime" style="flex:1;">
          <button class="btn secondary" id="clearSwapVideoBtn" type="button" style="padding:7px 10px;font-size:12px;">Clear</button>
        </div>
        <video id="swapVideoPreview" class="video-preview" controls muted></video>
        <div id="swapVideoMatchBar" style="display:none;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px;">
          <label class="swap-check" style="margin:0;"><input id="swapMatchVideoRes" type="checkbox" checked><span>Use Reference Video Resolution (<span id="swapMatchedResLabel">-</span>)</span></label>
          <label class="swap-check" style="margin:0;"><input id="swapMatchVideoDur" type="checkbox" checked><span>Match Video Duration</span></label>
        </div>

        <div id="swapCharActions" style="display:none;flex-wrap:wrap;gap:8px;margin-top:10px;">
          <button class="btn" id="detectCharsBtn" type="button" style="padding:7px 12px;font-size:12px;">List Characters in Video</button>
          <button class="btn secondary" id="pickFromFrameBtn" type="button" style="padding:7px 12px;font-size:12px;">Select Characters from Frame</button>
          <button class="btn secondary" id="addManualCharBtn" type="button" style="padding:7px 12px;font-size:12px;">+ Add Character</button>
          <button class="btn secondary" id="clearCharsBtn" type="button" style="padding:7px 10px;font-size:12px;">Clear List</button>
        </div>
        <div id="detectStatus" class="small-note" style="margin-top:6px;"></div>

        <div id="framePicker" class="frame-picker">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
            <span class="small-note" style="font-weight:600;color:var(--fg);">Drag a box around each character to separate them</span>
            <button class="mini-btn" id="closeFramePickerBtn" type="button">Done</button>
          </div>
          <div class="frame-strip" id="frameStrip"></div>
          <div class="crop-stage" id="cropStage">
            <canvas id="cropCanvas"></canvas>
          </div>
          <div id="cropHint" class="small-note" style="margin-top:6px;">Draw a rectangle around one character at a time. Each selection adds a separate character card below.</div>
        </div>

        <details style="margin-top:8px;">
          <summary class="small-note" style="cursor:pointer;">Character Detection VLM Settings (optional)</summary>
          <div style="margin-top:6px;display:flex;flex-direction:column;gap:6px;">
            <input id="visionUrl" type="text" placeholder="VLM URL (e.g. http://127.0.0.1:8082/v1/chat/completions)" value="DEFAULT_VISION_URL_PLACEHOLDER">
            <input id="visionModel" type="text" placeholder="VLM Model ID (optional)" value="DEFAULT_VISION_MODEL_PLACEHOLDER">
            <div class="small-note">Uses Qwen3.8-Flash-Next VLM on-demand when available. You can also select characters manually on the frame.</div>
          </div>
        </details>

        <div class="char-list" id="swapCharList"></div>

        <div class="swap-opts">
          <label for="swapScope" style="margin-top:0;">3. Swap Mode &amp; Scene Fidelity</label>
          <select id="swapScope">
            <option value="full" selected>Full Character Swap (face, body, hair &amp; outfit)</option>
            <option value="identity">Face &amp; Identity Only (keep original clothing)</option>
            <option value="outfit">Character + Custom Outfit / Style</option>
          </select>

          <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:2px 10px;">
            <label class="swap-check"><input id="optKeepOthers" type="checkbox" checked><span>Keep other characters</span></label>
            <label class="swap-check"><input id="optKeepPose" type="checkbox" checked><span>Keep pose &amp; motion</span></label>
            <label class="swap-check"><input id="optKeepCamera" type="checkbox" checked><span>Keep camera &amp; framing</span></label>
            <label class="swap-check"><input id="optKeepLighting" type="checkbox" checked><span>Match lighting &amp; color</span></label>
            <label class="swap-check"><input id="optKeepBackground" type="checkbox" checked><span>Keep background &amp; props</span></label>
            <label class="swap-check"><input id="optKeepExpression" type="checkbox" checked><span>Keep facial expression</span></label>
            <label class="swap-check"><input id="optKeepOcclusion" type="checkbox" checked><span>Keep occlusions</span></label>
            <label class="swap-check"><input id="optHighRealism" type="checkbox" checked><span>Seamless realism</span></label>
          </div>

          <label for="swapAudioMode">Audio Output</label>
          <select id="swapAudioMode">
            <option value="silent" selected>Silent / No Audio (disable audio generation)</option>
            <option value="preserve">Preserve Original Video Audio</option>
            <option value="custom">Custom Audio / Voice Description</option>
          </select>
          <div id="swapAudioNoteWrap" style="display:none;">
            <label for="swapAudioNote">Custom Audio / Voice Description</label>
            <input id="swapAudioNote" type="text" placeholder="e.g. speaking in a calm warm voice">
          </div>

          <label for="swapExtraPrompt">Additional Swap Instructions (optional)</label>
          <input id="swapExtraPrompt" type="text" placeholder="e.g. wearing a tailored black coat, natural motion blur">

          <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:10px;">
            <label class="swap-check" style="margin:0;"><input id="autoSyncSwapPrompt" type="checkbox" checked><span>Auto-update Prompt below</span></label>
            <button class="btn secondary" id="buildSwapPromptBtn" type="button" style="padding:6px 10px;font-size:12px;">Rebuild Prompt</button>
          </div>
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
            <option value="480x832">480x832 (9:16 480p Portrait)</option>
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

      <div class="row">
        <div>
          <label for="cfgScale">CFG Scale (1.0 = off; 3–5 = stronger prompt, 2× slower)</label>
          <input id="cfgScale" type="number" min="1" max="20" step="0.5" value="1">
        </div>
        <div></div>
      </div>

      <label for="negativePrompt">Negative Prompt (used when CFG &gt; 1.0)</label>
      <textarea id="negativePrompt" style="min-height:54px;">blurry, out of focus, distorted, deformed, bad anatomy, extra limbs, low quality, worst quality, artifacts, watermark, text, overexposed, flickering</textarea>

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
      <div class="row" style="margin-top:6px;">
        <div>
          <label for="activeBlocks">Active DiT Blocks (3–50; lower saves VRAM &amp; speed on 8s+)</label>
          <input id="activeBlocks" type="number" min="3" max="50" step="1" value="50">
        </div>
        <div style="display:flex;align-items:flex-end;">
          <label style="display:flex;align-items:center;gap:8px;margin:0 0 8px;cursor:pointer;">
            <input id="stepReuse" type="checkbox" style="width:auto;">
            <span>Step reuse (~2× faster)</span>
          </label>
        </div>
      </div>

      <label>Reference Images — Characters / Location / Objects (Ref2VA, up to 9)</label>
      <input id="refPicker" type="file" accept="image/png,image/jpeg,image/webp" multiple style="display:none;">
      <div style="display:flex;gap:8px;margin-top:4px;">
        <button class="btn secondary" id="addRefBtn" type="button" style="padding:7px 12px;font-size:12px;">+ Add Reference Image(s)</button>
        <button class="btn secondary" id="clearRefBtn" type="button" style="padding:7px 12px;font-size:12px;">Clear All</button>
      </div>
      <div class="ref-list" id="refList"></div>

      <div id="loraSection" style="display:none; margin-top:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:5px;">
          <label id="loraLabel" style="margin:0;">LoRA Adapters</label>
          <input id="loraFilter" type="text" placeholder="Filter LoRAs..." style="width:160px; padding:4px 8px; font-size:12px;">
        </div>
        <div id="loraList" style="display:flex; flex-direction:column; gap:6px; max-height:200px; overflow-y:auto;"></div>
      </div>

      <div class="actions">
        <button class="btn" id="runBtn" type="button">Generate Video</button>
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
let swapVideoUrl = null;
let swapChars = []; // { id, selected, label, description, note, sourceThumbUrl, file, url }
let nextCharSeq = 1;
let lastExtractedFrames = [];
let pickerFrameIdx = 0;
let pickerTargetCharId = null;

function tabRunLabel() {
  if (activeTab === "edit") return "Edit Video";
  if (activeTab === "swap") return "Swap Characters";
  return "Generate Video";
}

function switchTab(tab) {
  activeTab = tab;
  $("tabGen").className = tab === "gen" ? "tab active" : "tab";
  $("tabEdit").className = tab === "edit" ? "tab active" : "tab";
  $("tabSwap").className = tab === "swap" ? "tab active" : "tab";
  $("editSection").style.display = tab === "edit" ? "block" : "none";
  $("swapSection").style.display = tab === "swap" ? "block" : "none";
  $("keyframeSection").style.display = tab === "gen" ? "block" : "none";
  if (!$("runBtn").disabled) {
    $("runBtn").textContent = tabRunLabel();
  }
}
$("tabGen").onclick = () => switchTab("gen");
$("tabEdit").onclick = () => switchTab("edit");
$("tabSwap").onclick = () => switchTab("swap");

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

function computeAlignedVideoSize(vw, vh) {
  if (!vw || !vh) return null;
  const maxW = vw >= vh ? 1344 : 768;
  const maxH = vw >= vh ? 768 : 1344;
  const scale = Math.min(1, maxW / vw, maxH / vh);
  const round32 = (v) => Math.max(256, Math.round((v * scale) / 32) * 32);
  return { w: round32(vw), h: round32(vh) };
}

function applyMatchedVideoSettings() {
  const videoEl = $("swapVideoPreview");
  if (!videoEl.src || !videoEl.videoWidth || !videoEl.videoHeight) return;
  const aligned = computeAlignedVideoSize(videoEl.videoWidth, videoEl.videoHeight);
  if (aligned) {
    const val = `${aligned.w}x${aligned.h}`;
    $("swapMatchedResLabel").textContent = val;
    let opt = document.getElementById("refVideoResOption");
    if (!opt) {
      opt = document.createElement("option");
      opt.id = "refVideoResOption";
      $("resolution").prepend(opt);
    }
    opt.value = val;
    opt.textContent = `${val} (Reference Video)`;
    if ($("swapMatchVideoRes").checked) {
      $("resolution").value = val;
    }
  }
  if ($("swapMatchVideoDur").checked && Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
    const dur = videoEl.duration;
    let snapped;
    if (dur < 3) {
      snapped = 1;
    } else {
      snapped = Math.min(14.375, Math.max(5, Math.round(dur * 8) / 8));
    }
    $("seconds").value = String(snapped);
  }
  const durStr = Number.isFinite(videoEl.duration) ? `${videoEl.duration.toFixed(1)}s` : "";
  $("swapVideoMeta").textContent = `${videoEl.videoWidth}×${videoEl.videoHeight}${durStr ? " · " + durStr : ""}`;
}

$("swapVideoPreview").onloadedmetadata = () => applyMatchedVideoSettings();
$("swapMatchVideoRes").onchange = () => applyMatchedVideoSettings();
$("swapMatchVideoDur").onchange = () => applyMatchedVideoSettings();

$("swapVideo").onchange = (e) => {
  if (swapVideoUrl) { URL.revokeObjectURL(swapVideoUrl); swapVideoUrl = null; }
  const f = e.target.files[0];
  if (f) {
    swapVideoUrl = URL.createObjectURL(f);
    $("swapVideoPreview").src = swapVideoUrl;
    $("swapVideoPreview").style.display = "block";
    $("swapVideoMatchBar").style.display = "flex";
    $("swapCharActions").style.display = "flex";
    $("detectStatus").textContent = "Video loaded. Click 'List Characters in Video' or 'Select Characters from Frame'.";
  } else {
    $("swapVideoPreview").style.display = "none";
    $("swapVideoMatchBar").style.display = "none";
    $("swapCharActions").style.display = "none";
    $("swapVideoMeta").textContent = "";
    $("detectStatus").textContent = "";
  }
};
$("clearSwapVideoBtn").onclick = () => {
  $("swapVideo").value = "";
  if (swapVideoUrl) { URL.revokeObjectURL(swapVideoUrl); swapVideoUrl = null; }
  $("swapVideoPreview").removeAttribute("src");
  $("swapVideoPreview").style.display = "none";
  $("swapVideoMatchBar").style.display = "none";
  $("swapCharActions").style.display = "none";
  $("swapVideoMeta").textContent = "";
  $("detectStatus").textContent = "";
};

function computeSwapPictureIndices() {
  let picIdx = 1;
  const map = new Map();
  for (const ch of swapChars) {
    if (ch.selected && ch.file && picIdx <= 9) {
      map.set(ch.id, picIdx++);
    }
  }
  return map;
}

function buildSwapPrompt(force = false) {
  if (!force && !$("autoSyncSwapPrompt").checked) return;
  const picMap = computeSwapPictureIndices();
  const replacements = [];
  const keptCharacters = [];
  for (const ch of swapChars) {
    const desc = (ch.description || ch.label || "the character").trim();
    if (!ch.selected) {
      if (desc) keptCharacters.push(desc);
      continue;
    }
    const picNum = picMap.get(ch.id);
    if (!picNum) continue;
    const scope = $("swapScope").value;
    let action;
    if (scope === "identity") {
      action = `replace only the face and identity of ${desc} in <Video 1> with the person from <Picture ${picNum}> while keeping their original clothing`;
    } else if (scope === "outfit") {
      action = `replace ${desc} in <Video 1> with the character from <Picture ${picNum}>`;
    } else {
      action = `replace ${desc} in <Video 1> with the character from <Picture ${picNum}> (face, body, hair, and outfit)`;
    }
    if (ch.note && ch.note.trim()) {
      action += ` (${ch.note.trim()})`;
    }
    replacements.push(action);
  }
  if (replacements.length === 0) return;

  const clauses = [];
  if ($("optKeepOthers").checked) {
    if (keptCharacters.length > 0) {
      clauses.push(`keep ${keptCharacters.join(" and ")} and all other people in <Video 1> completely unchanged and present in their exact positions`);
    } else {
      clauses.push("keep all other unswapped people in <Video 1> completely unchanged and present");
    }
  }
  if ($("optKeepPose").checked) clauses.push("keep the exact body pose, gestures, timing, and movement from <Video 1>");
  if ($("optKeepCamera").checked) clauses.push("preserve the exact camera movement, angle, and framing");
  if ($("optKeepLighting").checked) clauses.push("match the scene lighting, shadows, and color grading");
  if ($("optKeepBackground").checked) clauses.push("keep the exact background, props, and environment from <Video 1> unchanged");
  if ($("optKeepExpression").checked) clauses.push("match the original facial expressions and head turns");
  if ($("optKeepOcclusion").checked) clauses.push("preserve natural occlusions and object interactions");
  if ($("optHighRealism").checked) clauses.push("with photorealistic, seamless blending and consistent character identity across every frame");

  let promptText = `Edit <Video 1> directly without changing the scene, action, or composition: ${replacements.join("; ")}`;
  if (clauses.length > 0) {
    promptText += ", " + clauses.join(", ");
  }
  promptText += ".";

  const audioMode = $("swapAudioMode").value;
  if (audioMode === "silent") {
    promptText += " Silent video with no singing, speech, or music.";
  } else if (audioMode === "custom") {
    const audioNote = $("swapAudioNote").value.trim();
    if (audioNote) promptText += ` Audio: ${audioNote}.`;
  } else if (audioMode === "preserve") {
    promptText += " Preserve the original audio from <Video 1>.";
  }

  const extra = $("swapExtraPrompt").value.trim();
  if (extra) {
    promptText += ` ${extra}`;
  }

  $("prompt").value = promptText;
}

function renderSwapChars() {
  const container = $("swapCharList");
  container.innerHTML = "";
  const picMap = computeSwapPictureIndices();

  swapChars.forEach((ch, idx) => {
    const card = document.createElement("div");
    card.className = "char-card" + (ch.selected ? " selected" : "");

    const head = document.createElement("div");
    head.className = "char-head";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = ch.selected;
    cb.title = "Select this character to swap";
    cb.onchange = () => {
      ch.selected = cb.checked;
      renderSwapChars();
      buildSwapPrompt();
    };

    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.value = ch.label;
    labelInput.placeholder = `Character ${idx + 1} name/role`;
    labelInput.style.flex = "1";
    labelInput.oninput = () => {
      ch.label = labelInput.value;
      buildSwapPrompt();
    };

    const picNum = picMap.get(ch.id);
    const badge = document.createElement("span");
    badge.className = "ref-tag";
    badge.textContent = picNum ? `<Picture ${picNum}>` : (ch.selected ? "Needs image" : "Skipped");

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "mini-btn del";
    delBtn.textContent = "×";
    delBtn.title = "Remove character";
    delBtn.onclick = () => {
      if (ch.url) URL.revokeObjectURL(ch.url);
      swapChars = swapChars.filter((item) => item.id !== ch.id);
      renderSwapChars();
      buildSwapPrompt();
    };
    const cropBtn = document.createElement("button");
    cropBtn.type = "button";
    cropBtn.className = "mini-btn";
    cropBtn.textContent = "Crop";
    cropBtn.title = "Select/re-crop this character from a video frame";
    cropBtn.onclick = () => openFramePicker(ch.id);
    head.append(cb, labelInput, badge, cropBtn, delBtn);

    const sourceRow = document.createElement("div");
    sourceRow.style.cssText = "display:flex;align-items:center;gap:10px;";
    if (ch.sourceThumbUrl) {
      const srcWrap = document.createElement("div");
      srcWrap.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:2px;flex-shrink:0;";
      const srcImg = document.createElement("img");
      srcImg.className = "char-thumb";
      srcImg.style.display = "block";
      srcImg.src = ch.sourceThumbUrl;
      srcImg.title = "Character frame from source video";
      const srcCap = document.createElement("span");
      srcCap.className = "small-note";
      srcCap.textContent = "In video";
      srcWrap.append(srcImg, srcCap);
      sourceRow.appendChild(srcWrap);
    }

    const descInput = document.createElement("input");
    descInput.type = "text";
    descInput.style.flex = "1";
    descInput.value = ch.description;
    descInput.placeholder = "Describe this character in <Video 1> (e.g. man in blue jacket on the left)";
    descInput.oninput = () => {
      ch.description = descInput.value;
      buildSwapPrompt();
    };
    sourceRow.appendChild(descInput);

    const uploadBox = document.createElement("div");
    uploadBox.className = "char-upload";
    const thumb = document.createElement("img");
    thumb.className = "char-thumb";
    if (ch.url) {
      thumb.src = ch.url;
      thumb.style.display = "block";
    }
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/png,image/jpeg,image/webp";
    fileInput.style.display = "none";
    fileInput.onchange = (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      if (ch.url) URL.revokeObjectURL(ch.url);
      ch.file = f;
      ch.url = URL.createObjectURL(f);
      ch.selected = true;
      renderSwapChars();
      buildSwapPrompt();
    };

    const uploadInfo = document.createElement("div");
    uploadInfo.style.cssText = "flex:1;min-width:0;";
    const uploadTitle = document.createElement("div");
    uploadTitle.className = "ref-name";
    uploadTitle.textContent = ch.file ? `New character: ${ch.file.name}` : "Upload replacement character image";
    const uploadHint = document.createElement("div");
    uploadHint.className = "small-note";
    uploadHint.textContent = picNum ? `Mapped to <Picture ${picNum}>` : "PNG, JPEG, or WebP";
    uploadInfo.append(uploadTitle, uploadHint);

    const uploadBtn = document.createElement("button");
    uploadBtn.type = "button";
    uploadBtn.className = "mini-btn";
    uploadBtn.textContent = ch.file ? "Change" : "Upload Image";
    uploadBtn.onclick = () => fileInput.click();

    uploadBox.append(thumb, fileInput, uploadInfo, uploadBtn);
    if (ch.file) {
      const clrImgBtn = document.createElement("button");
      clrImgBtn.type = "button";
      clrImgBtn.className = "mini-btn del";
      clrImgBtn.textContent = "Clear";
      clrImgBtn.onclick = () => {
        if (ch.url) URL.revokeObjectURL(ch.url);
        ch.file = null;
        ch.url = null;
        renderSwapChars();
        buildSwapPrompt();
      };
      uploadBox.appendChild(clrImgBtn);
    }

    const noteInput = document.createElement("input");
    noteInput.type = "text";
    noteInput.value = ch.note || "";
    noteInput.placeholder = "Optional per-character swap note (e.g. wearing sunglasses, smiling)";
    noteInput.oninput = () => {
      ch.note = noteInput.value;
      buildSwapPrompt();
    };

    card.append(head, sourceRow, uploadBox, noteInput);
    container.appendChild(card);
  });
}

function addSwapCharacter(data = {}) {
  const id = `char_${nextCharSeq++}`;
  swapChars.push({
    id,
    selected: data.selected !== undefined ? data.selected : true,
    label: data.label || `Character ${swapChars.length + 1}`,
    description: data.description || "",
    note: data.note || "",
    sourceThumbUrl: data.sourceThumbUrl || null,
    file: null,
    url: null,
  });
  renderSwapChars();
}

$("addManualCharBtn").onclick = () => addSwapCharacter();
$("clearCharsBtn").onclick = () => {
  for (const ch of swapChars) {
    if (ch.url) URL.revokeObjectURL(ch.url);
  }
  swapChars = [];
  renderSwapChars();
  $("detectStatus").textContent = "Character list cleared.";
};

async function extractVideoFrames(videoEl, maxFrames = 4) {
  if (!videoEl.src) throw new Error("No video loaded.");
  if (videoEl.readyState < 1) {
    await new Promise((resolve, reject) => {
      const onLoaded = () => { cleanup(); resolve(); };
      const onErr = () => { cleanup(); reject(new Error("Could not decode video metadata.")); };
      const cleanup = () => {
        videoEl.removeEventListener("loadedmetadata", onLoaded);
        videoEl.removeEventListener("error", onErr);
      };
      videoEl.addEventListener("loadedmetadata", onLoaded);
      videoEl.addEventListener("error", onErr);
    });
  }
  const duration = Number.isFinite(videoEl.duration) && videoEl.duration > 0 ? videoEl.duration : 1;
  const vw = videoEl.videoWidth || 512;
  const vh = videoEl.videoHeight || 512;
  const scale = Math.min(1, 512 / Math.max(vw, vh));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(64, Math.round(vw * scale));
  canvas.height = Math.max(64, Math.round(vh * scale));
  const ctx = canvas.getContext("2d");
  const frames = [];
  const prevTime = videoEl.currentTime;
  for (let i = 0; i < maxFrames; i++) {
    const t = maxFrames === 1 ? 0 : (duration * (i + 0.5)) / maxFrames;
    await new Promise((resolve) => {
      const onSeek = () => { videoEl.removeEventListener("seeked", onSeek); resolve(); };
      videoEl.addEventListener("seeked", onSeek);
      videoEl.currentTime = Math.min(duration - 0.01, Math.max(0, t));
    });
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    frames.push(canvas.toDataURL("image/jpeg", 0.8));
  }
  try { videoEl.currentTime = prevTime; } catch (_) {}
  return frames;
}

function normalizeBbox(rawBbox) {
  if (!rawBbox) return null;
  let nums = null;
  if (Array.isArray(rawBbox) && rawBbox.length === 4) {
    nums = rawBbox.map((v) => Number(v));
  } else if (typeof rawBbox === "object") {
    const x1 = rawBbox.x1 ?? rawBbox.xmin ?? rawBbox.left;
    const y1 = rawBbox.y1 ?? rawBbox.ymin ?? rawBbox.top;
    const x2 = rawBbox.x2 ?? rawBbox.xmax ?? rawBbox.right;
    const y2 = rawBbox.y2 ?? rawBbox.ymax ?? rawBbox.bottom;
    if (x1 !== undefined && y1 !== undefined && x2 !== undefined && y2 !== undefined) {
      nums = [Number(x1), Number(y1), Number(x2), Number(y2)];
    }
  }
  if (!nums || nums.some((v) => !Number.isFinite(v))) return null;
  const maxVal = Math.max(...nums);
  const scale = maxVal <= 1.05 ? 1 : 1000;
  const [x1, y1, x2, y2] = nums.map((v) => Math.max(0, Math.min(1, v / scale)));
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

async function cropCharacterFromFrame(frameUrl, rawBbox) {
  if (!frameUrl) return null;
  const norm = normalizeBbox(rawBbox);
  if (!norm) return frameUrl;
  let [x1, y1, x2, y2] = norm;
  const padX = (x2 - x1) * 0.12;
  const padY = (y2 - y1) * 0.12;
  x1 = Math.max(0, x1 - padX);
  y1 = Math.max(0, y1 - padY);
  x2 = Math.min(1, x2 + padX);
  y2 = Math.min(1, y2 + padY);
  return await new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const sx = Math.floor(x1 * img.width);
      const sy = Math.floor(y1 * img.height);
      const sw = Math.max(16, Math.ceil((x2 - x1) * img.width));
      const sh = Math.max(16, Math.ceil((y2 - y1) * img.height));
      const canvas = document.createElement("canvas");
      canvas.width = sw;
      canvas.height = sh;
      canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = () => resolve(frameUrl);
    img.src = frameUrl;
  });
}

let pickerImg = null;
let pickerDragStart = null;
let pickerDragCurr = null;

function drawPickerCanvas() {
  const canvas = $("cropCanvas");
  if (!pickerImg) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(pickerImg, 0, 0, canvas.width, canvas.height);
  if (pickerDragStart && pickerDragCurr) {
    const x = Math.min(pickerDragStart.x, pickerDragCurr.x);
    const y = Math.min(pickerDragStart.y, pickerDragCurr.y);
    const w = Math.abs(pickerDragCurr.x - pickerDragStart.x);
    const h = Math.abs(pickerDragCurr.y - pickerDragStart.y);
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(59, 130, 246, 0.2)";
    ctx.fillRect(x, y, w, h);
  }
}

function selectPickerFrame(idx) {
  if (!lastExtractedFrames.length) return;
  pickerFrameIdx = Math.max(0, Math.min(lastExtractedFrames.length - 1, idx));
  const strip = $("frameStrip");
  Array.from(strip.children).forEach((el, i) => {
    el.className = i === pickerFrameIdx ? "frame-thumb active" : "frame-thumb";
  });
  const img = new Image();
  img.onload = () => {
    pickerImg = img;
    const canvas = $("cropCanvas");
    canvas.width = img.width;
    canvas.height = img.height;
    pickerDragStart = null;
    pickerDragCurr = null;
    drawPickerCanvas();
  };
  img.src = lastExtractedFrames[pickerFrameIdx];
}

async function openFramePicker(targetCharId = null) {
  const videoEl = $("swapVideoPreview");
  if (!videoEl.src) {
    $("detectStatus").textContent = "Please load a source video first.";
    return;
  }
  pickerTargetCharId = targetCharId;
  if (!lastExtractedFrames.length) {
    lastExtractedFrames = await extractVideoFrames(videoEl, 6);
  }
  const strip = $("frameStrip");
  strip.innerHTML = "";
  lastExtractedFrames.forEach((url, idx) => {
    const t = document.createElement("img");
    t.src = url;
    t.className = idx === 0 ? "frame-thumb active" : "frame-thumb";
    t.onclick = () => selectPickerFrame(idx);
    strip.appendChild(t);
  });
  $("cropHint").textContent = targetCharId
    ? "Drag a box around the character to update its thumbnail."
    : "Draw a rectangle around one character at a time. Each selection adds a separate character card below.";
  $("framePicker").style.display = "block";
  selectPickerFrame(0);
}

$("pickFromFrameBtn").onclick = () => openFramePicker(null);
$("closeFramePickerBtn").onclick = () => {
  $("framePicker").style.display = "none";
  pickerTargetCharId = null;
};

function canvasCoords(canvas, evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((evt.clientX - rect.left) / rect.width) * canvas.width,
    y: ((evt.clientY - rect.top) / rect.height) * canvas.height,
  };
}

$("cropCanvas").addEventListener("mousedown", (evt) => {
  if (!pickerImg) return;
  pickerDragStart = canvasCoords($("cropCanvas"), evt);
  pickerDragCurr = pickerDragStart;
  drawPickerCanvas();
});
window.addEventListener("mousemove", (evt) => {
  if (!pickerDragStart) return;
  pickerDragCurr = canvasCoords($("cropCanvas"), evt);
  drawPickerCanvas();
});
window.addEventListener("mouseup", async () => {
  if (!pickerDragStart || !pickerDragCurr || !pickerImg) return;
  const x1 = Math.min(pickerDragStart.x, pickerDragCurr.x);
  const y1 = Math.min(pickerDragStart.y, pickerDragCurr.y);
  const x2 = Math.max(pickerDragStart.x, pickerDragCurr.x);
  const y2 = Math.max(pickerDragStart.y, pickerDragCurr.y);
  pickerDragStart = null;
  pickerDragCurr = null;
  drawPickerCanvas();
  if (x2 - x1 < 12 || y2 - y1 < 12) return;

  const cw = $("cropCanvas").width;
  const ch = $("cropCanvas").height;
  const bbox = [x1 / cw, y1 / ch, x2 / cw, y2 / ch];
  const thumbUrl = await cropCharacterFromFrame(lastExtractedFrames[pickerFrameIdx], bbox);
  const cx = (x1 + x2) / (2 * cw);
  const posLabel = cx < 0.35 ? "on the left" : (cx > 0.65 ? "on the right" : "in the center");

  if (pickerTargetCharId) {
    const target = swapChars.find((c) => c.id === pickerTargetCharId);
    if (target) {
      target.sourceThumbUrl = thumbUrl;
      renderSwapChars();
    }
    $("framePicker").style.display = "none";
    pickerTargetCharId = null;
  } else {
    addSwapCharacter({
      selected: true,
      label: `Character ${swapChars.length + 1}`,
      description: `the person ${posLabel}`,
      sourceThumbUrl: thumbUrl,
    });
    buildSwapPrompt();
  }
});

$("detectCharsBtn").onclick = async () => {
  const videoEl = $("swapVideoPreview");
  if (!videoEl.src) {
    $("detectStatus").textContent = "Please load a source video first.";
    return;
  }
  $("detectCharsBtn").disabled = true;
  $("detectStatus").textContent = "Extracting video frames and separating characters...";
  try {
    const frames = await extractVideoFrames(videoEl, 6);
    lastExtractedFrames = frames;
    const resp = await fetch("/api/detect-characters", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frames,
        vision_url: $("visionUrl").value.trim(),
        vision_model: $("visionModel").value.trim(),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    const detected = Array.isArray(data.characters) ? data.characters : [];
    if (detected.length === 0) {
      $("detectStatus").textContent = "No characters auto-detected. Drag boxes on the frame below to separate each character.";
      openFramePicker(null);
      return;
    }
    for (const ch of swapChars) { if (ch.url) URL.revokeObjectURL(ch.url); }
    swapChars = [];
    for (const item of detected) {
      const fIdx = Number.isInteger(item.frame_index) && item.frame_index >= 0 && item.frame_index < frames.length
        ? item.frame_index
        : 0;
      const thumbUrl = await cropCharacterFromFrame(frames[fIdx], item.bbox || item.bbox_2d);
      addSwapCharacter({
        selected: true,
        label: item.label || item.name || "Character",
        description: item.description || item.appearance || "",
        sourceThumbUrl: thumbUrl,
      });
    }
    const via = data.detector ? ` (${data.detector})` : "";
    $("detectStatus").textContent = `Detected ${detected.length} character(s) across ${frames.length} frames${via}. Use 'Crop' or 'Select Characters from Frame' to refine.`;
    buildSwapPrompt();
  } catch (err) {
    $("detectStatus").textContent = `Auto-detection unavailable (${err.message}). Drag boxes on the frame below to separate each character.`;
    openFramePicker(null);
  } finally {
    $("detectCharsBtn").disabled = false;
  }
};

$("swapAudioMode").onchange = () => {
  $("swapAudioNoteWrap").style.display = $("swapAudioMode").value === "custom" ? "block" : "none";
  buildSwapPrompt();
};
$("buildSwapPromptBtn").onclick = (e) => {
  e.preventDefault();
  e.stopPropagation();
  buildSwapPrompt(true);
};
for (const id of [
  "swapScope", "optKeepOthers", "optKeepPose", "optKeepCamera", "optKeepLighting",
  "optKeepBackground", "optKeepExpression", "optKeepOcclusion",
  "optHighRealism", "swapAudioNote", "swapExtraPrompt"
]) {
  $(id).addEventListener("change", () => buildSwapPrompt());
  $(id).addEventListener("input", () => buildSwapPrompt());
}

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

let lorasLoaded = false;

async function loadLoras() {
  if (lorasLoaded) return;
  try {
    const r = await fetch("/v1/videos/loras");
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
    $("loraLabel").textContent = `LoRA Adapters (${list.length})`;
    $("loraFilter").oninput = () => {
      const q = $("loraFilter").value.trim().toLowerCase();
      for (const row of wrap.children) {
        const name = (row.dataset.loraName || "").toLowerCase();
        row.style.display = !q || name.includes(q) ? "flex" : "none";
      }
    };
    for (const item of list) {
      const name = item.name;
      if (!name) continue;
      const row = document.createElement("div");
      row.dataset.loraName = name;
      row.style.cssText = "display:flex; align-items:center; gap:8px; background:var(--panel-2); border:1px solid var(--border); border-radius:8px; padding:6px 10px;";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.style.width = "auto";
      cb.dataset.loraName = name;
      cb.checked = false;
      const lbl = document.createElement("span");
      lbl.textContent = name;
      lbl.style.cssText = "flex:1; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;";
      const sc = document.createElement("input");
      sc.type = "number";
      sc.step = "0.05";
      sc.min = "-10";
      sc.max = "10";
      sc.value = "1.0";
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
      out.push({ name: cb.dataset.loraName, scale: parseFloat(sc.value) || 1.0 });
    }
  }
  return out;
}

async function pollHealth() {
  try {
    const r = await fetch("/ready");
    if (r.ok) {
      $("healthDot").className = "dot ok";
      $("healthText").textContent = "Server ready";
      loadLoras();
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
  $("runBtn").textContent = isPaused ? "Paused" : (isBusy ? "Generating..." : tabRunLabel());
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

$("videoForm").onsubmit = (e) => {
  e.preventDefault();
};

$("runBtn").onclick = async () => {
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
  const swapVideoFile = activeTab === "swap" ? $("swapVideo").files[0] : null;
  if (activeTab === "edit" && !editVideoFile) {
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = "Error: Please attach a Source Video to edit.";
    return;
  }
  let swapRefFiles = [];
  if (activeTab === "swap") {
    if (!swapVideoFile) {
      $("statusBar").className = "status-bar error";
      $("statusBar").textContent = "Error: Please attach a Source Video for character swap.";
      return;
    }
    swapRefFiles = swapChars.filter((ch) => ch.selected && ch.file).map((ch) => ch.file);
    if (swapRefFiles.length === 0) {
      $("statusBar").className = "status-bar error";
      $("statusBar").textContent = "Error: Select at least one character and upload a replacement character image.";
      return;
    }
  }
  const sourceVideoFile = editVideoFile || swapVideoFile;
  const refFiles = activeTab === "swap"
    ? [...swapRefFiles, ...refItems.map((item) => item.file)].slice(0, 9)
    : refItems.map((item) => item.file);
  const matchAspect = $("matchFirstAspect").checked;
  const stepReuse = $("stepReuse").checked;
  const activeBlocks = Math.min(50, Math.max(3, parseInt($("activeBlocks").value, 10) || 50));
  const cfgScale = Math.min(20, Math.max(1, parseFloat($("cfgScale").value) || 1.0));
  const negativePrompt = $("negativePrompt").value.trim();

  const gufoOpts = {
    preset,
    steps,
    lock_resolution: !matchAspect,
    step_reuse: stepReuse
  };
  if (cfgScale > 1.0) gufoOpts.cfg_scale = cfgScale;
  if (activeBlocks < 50 && !(stepReuse && activeBlocks === 45)) {
    gufoOpts.active_blocks = activeBlocks;
  }
  if (activeTab === "swap" && $("swapAudioMode").value === "silent") {
    gufoOpts.decode_audio = false;
  }
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
    const loras = selectedLoras();
    let resp;
    if (firstFile || lastFile || sourceVideoFile || refFiles.length > 0) {
      const form = new FormData();
      form.append("model", model);
      form.append("prompt", prompt);
      if (negativePrompt) form.append("negative_prompt", negativePrompt);
      form.append("size", size);
      form.append("seconds", seconds);
      form.append("gufo", JSON.stringify(gufoOpts));
      if (loras !== null) form.append("loras", JSON.stringify(loras));
      if (firstFile) form.append("first_frame", firstFile);
      if (lastFile) form.append("last_frame", lastFile);
      if (sourceVideoFile) form.append("reference_video_0", sourceVideoFile);
      refFiles.forEach((f, idx) => form.append(`reference_${idx}`, f));
      resp = await fetch("/v1/videos", { method: "POST", body: form });
    } else {
      const payload = { model, prompt, size, seconds: parseFloat(seconds), gufo: gufoOpts };
      if (negativePrompt) payload.negative_prompt = negativePrompt;
      if (loras !== null) payload.loras = loras;
      resp = await fetch("/v1/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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


def _extract_json_array(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict) and isinstance(parsed.get("characters"), list):
        return parsed["characters"]
    if isinstance(parsed, list):
        return parsed
    raise ValueError("Vision response did not contain a JSON array of characters")


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _crop_hsv_hist(img, box: tuple[float, float, float, float]):
    import cv2  # type: ignore

    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(box[0] * w)))
    y1 = max(0, min(h - 1, int(box[1] * h)))
    x2 = max(x1 + 1, min(w, int(box[2] * w)))
    y2 = max(y1 + 1, min(h, int(box[3] * h)))
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _detect_characters_local_cv2(frames: list[str]) -> list[dict]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    alt2_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
    default_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_xml = alt2_xml if os.path.exists(alt2_xml) else default_xml
    face_cascade = cv2.CascadeClassifier(face_xml) if os.path.exists(face_xml) else None

    profile_xml = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
    profile_cascade = cv2.CascadeClassifier(profile_xml) if os.path.exists(profile_xml) else None

    per_frame_detections: list[tuple[int, tuple[float, float, float, float], float, object]] = []
    for f_idx, data_url in enumerate(frames[:6]):
        if not isinstance(data_url, str) or "," not in data_url:
            continue
        raw_bytes = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        min_face = max(28, int(min(h, w) * 0.06))
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        # Collect (box, score) for this frame
        frame_candidates: list[tuple[tuple[float, float, float, float], float]] = []

        # 1. Frontal face cascade (alt2 is much cleaner than default)
        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_face, min_face)
            )
            for (fx, fy, fw, fh) in faces:
                box = (
                    max(0.0, (fx - 0.55 * fw) / w),
                    max(0.0, (fy - 0.35 * fh) / h),
                    min(1.0, (fx + 1.55 * fw) / w),
                    min(1.0, (fy + 2.6 * fh) / h),
                )
                frame_candidates.append((box, 2.0))

        # 2. Profile face cascade (left & right via horizontal flip)
        if profile_cascade is not None and not profile_cascade.empty():
            for flipped in (False, True):
                g_in = cv2.flip(gray, 1) if flipped else gray
                profs = profile_cascade.detectMultiScale(
                    g_in, scaleFactor=1.1, minNeighbors=6, minSize=(min_face, min_face)
                )
                for (fx, fy, fw, fh) in profs:
                    if flipped:
                        fx = w - (fx + fw)
                    box = (
                        max(0.0, (fx - 0.55 * fw) / w),
                        max(0.0, (fy - 0.35 * fh) / h),
                        min(1.0, (fx + 1.55 * fw) / w),
                        min(1.0, (fy + 2.6 * fh) / h),
                    )
                    frame_candidates.append((box, 1.5))

        # 3. HOG full/upper-body detector with score threshold + NMS
        rects, weights = hog.detectMultiScale(
            img, winStride=(8, 8), padding=(8, 8), scale=1.05, hitThreshold=0.55
        )
        if len(rects) > 0:
            nms_boxes = [[int(px), int(py), int(pw), int(ph)] for (px, py, pw, ph) in rects]
            nms_scores = [float(wt) for wt in weights]
            keep = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=0.55, nms_threshold=0.45)
            for k in np.array(keep).flatten():
                px, py, pw, ph = rects[int(k)]
                box = (px / w, py / h, (px + pw) / w, (py + ph) / h)
                frame_candidates.append((box, float(weights[int(k)])))

        # Deduplicate overlapping boxes within the same frame (keep highest-priority box)
        frame_candidates.sort(key=lambda item: item[1], reverse=True)
        kept: list[tuple[tuple[float, float, float, float], float]] = []
        for box, score in frame_candidates:
            if any(_box_iou(box, kbox) > 0.35 for kbox, _ in kept):
                continue
            kept.append((box, score))

        for box, score in kept:
            hist = _crop_hsv_hist(img, box)
            per_frame_detections.append((f_idx, box, score, hist))

    if not per_frame_detections:
        return []

    # Cluster across frames by appearance (HSV correlation) + spatial proximity,
    # while NEVER merging two detections from the same frame.
    clusters: list[dict] = []
    for f_idx, box, score, hist in per_frame_detections:
        cx = (box[0] + box[2]) / 2.0
        area = (box[2] - box[0]) * (box[3] - box[1])
        best_cl = None
        best_sim = 0.0
        for cl in clusters:
            if f_idx in cl["frames"]:
                continue
            sim = float(cv2.compareHist(cl["hist"], hist, cv2.HISTCMP_CORREL))
            spatial_bonus = 0.12 if abs(cl["cx"] - cx) < 0.2 else 0.0
            total_sim = sim + spatial_bonus
            if sim >= 0.52 and total_sim > best_sim:
                best_sim = total_sim
                best_cl = cl

        quality = score * (0.5 + area)
        if best_cl is None:
            clusters.append({
                "cx": cx,
                "frame_index": f_idx,
                "bbox": list(box),
                "quality": quality,
                "hist": hist,
                "frames": {f_idx},
            })
        else:
            best_cl["frames"].add(f_idx)
            if quality > best_cl["quality"]:
                best_cl["cx"] = cx
                best_cl["frame_index"] = f_idx
                best_cl["bbox"] = list(box)
                best_cl["quality"] = quality
                best_cl["hist"] = hist

    # Drop single-frame low-confidence glitches if we inspected >= 3 frames and have multi-frame tracks
    if len(frames) >= 3:
        multi = [cl for cl in clusters if len(cl["frames"]) >= 2]
        if multi:
            clusters = multi

    clusters.sort(key=lambda c: c["cx"])
    results = []
    for idx, cl in enumerate(clusters, start=1):
        cx = cl["cx"]
        pos = "on the left" if cx < 0.35 else ("on the right" if cx > 0.65 else "in the center")
        results.append({
            "label": f"Character {idx}",
            "description": f"the person {pos}",
            "frame_index": cl["frame_index"],
            "bbox": [round(v * 1000) for v in cl["bbox"]],
        })
    return results


class ManagedVlmServer:
    def __init__(
        self,
        gufo_exe: str,
        model_path: str,
        mmproj_path: str = "",
        port: int = 0,
    ) -> None:
        self.gufo_exe = gufo_exe
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.requested_port = port
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def is_configured(self) -> bool:
        return bool(self.gufo_exe and self.model_path)

    def ensure_running(self, timeout_sec: float = 120.0) -> str:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return f"http://127.0.0.1:{self.port}/v1/chat/completions"
            self.port = self.requested_port if self.requested_port > 0 else _find_free_port(8082)
            cmd = [
                self.gufo_exe,
                "serve",
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--no-webui",
                "--model", self.model_path,
                "--served-model-name", "vlm",
            ]
            if self.mmproj_path:
                cmd.extend(["--mmproj", self.mmproj_path])
            print(f"Starting on-demand Gufo VLM server: {' '.join(cmd)}", flush=True)
            self.proc = subprocess.Popen(cmd)

            ready_url = f"http://127.0.0.1:{self.port}/ready"
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"VLM server exited with code {self.proc.returncode}")
                try:
                    with urllib.request.urlopen(ready_url, timeout=2) as r:
                        if r.status == 200:
                            return f"http://127.0.0.1:{self.port}/v1/chat/completions"
                except Exception:
                    pass
                time.sleep(0.5)
            self.stop_locked()
            raise RuntimeError("Timed out waiting for VLM server to become ready")

    def stop_locked(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            print("Stopping on-demand Gufo VLM server to free memory...", flush=True)
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def stop(self) -> None:
        with self._lock:
            self.stop_locked()


def make_handler(
    backend_url: str,
    default_model: str,
    save_dir: str = "",
    default_vision_url: str = "",
    default_vision_model: str = "",
    vlm_manager: ManagedVlmServer | None = None,
):
    parsed = urllib.parse.urlsplit(backend_url)
    backend_host = parsed.hostname or "127.0.0.1"
    backend_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    video_content_re = re.compile(r"^/v1/videos/([^/]+)/content(?:\?.*)?$")

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self) -> None:
            html = (
                HTML_PAGE.replace("DEFAULT_MODEL_PLACEHOLDER", default_model)
                .replace("DEFAULT_VISION_URL_PLACEHOLDER", default_vision_url)
                .replace("DEFAULT_VISION_MODEL_PLACEHOLDER", default_vision_model)
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_detect_characters(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            frames = req.get("frames") or []
            if not isinstance(frames, list) or not frames:
                self._send_json(400, {"error": "No video frames provided."})
                return

            vlm_prompt = (
                f"List EVERY distinct person or character visible across these {len(frames[:6])} video frames (indexed 0 to {len(frames[:6]) - 1}). "
                "CRITICAL: If multiple characters appear in the same frame, output a SEPARATE JSON object for EACH individual person with their own tight bounding box. Never group multiple people into one entry. "
                "Return ONLY a JSON array of objects with keys: "
                "\"label\" (short name/role like 'Man in blue suit'), "
                "\"description\" (clear visual description including clothing, hair, and position in frame), "
                "\"frame_index\" (0-based integer index of the frame where this character is clearest), and "
                "\"bbox\" ([x_min, y_min, x_max, y_max] normalized 0-1000 bounding box around ONLY that individual character)."
            )

            vision_url = (req.get("vision_url") or default_vision_url or "").strip()
            used_managed_vlm = False
            if not vision_url and vlm_manager is not None and vlm_manager.is_configured():
                try:
                    vision_url = vlm_manager.ensure_running()
                    used_managed_vlm = True
                except Exception as vlm_exc:
                    try:
                        characters = _detect_characters_local_cv2(frames)
                        self._send_json(200, {
                            "characters": characters,
                            "detector": f"local OpenCV fallback (VLM launch failed: {vlm_exc})",
                        })
                    except ImportError:
                        self._send_json(502, {"error": f"VLM launch failed: {vlm_exc}"})
                    return

            if not vision_url:
                analyze_url = f"http://{backend_host}:{backend_port}/v1/videos/analyze"
                valid_frames = [
                    f for f in frames[:6] if isinstance(f, str) and f.startswith("data:image/")
                ]
                analyze_payload = json.dumps({
                    "images": valid_frames,
                    "prompt": vlm_prompt,
                    "max_tokens": 1024,
                }).encode("utf-8")
                analyze_req = urllib.request.Request(
                    analyze_url,
                    data=analyze_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(analyze_req, timeout=300) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                    characters = _extract_json_array(resp_data.get("text", ""))
                    self._send_json(200, {
                        "characters": characters,
                        "detector": "MiniMax H3 VL (Qwen3-VL-32B)",
                    })
                    return
                except urllib.error.HTTPError as h3_http_exc:
                    err_detail = f"HTTP {h3_http_exc.code}"
                    try:
                        err_json = json.loads(h3_http_exc.read().decode("utf-8"))
                        err_detail = (err_json.get("error") or {}).get("message") or err_detail
                    except Exception:
                        pass
                    try:
                        characters = _detect_characters_local_cv2(frames)
                        self._send_json(200, {
                            "characters": characters,
                            "detector": f"local OpenCV fallback (H3 VL error: {err_detail})",
                        })
                    except ImportError:
                        self._send_json(502, {"error": f"H3 VL character detection failed: {err_detail}"})
                    return
                except Exception as h3_exc:
                    try:
                        characters = _detect_characters_local_cv2(frames)
                        self._send_json(200, {
                            "characters": characters,
                            "detector": f"local OpenCV fallback (H3 VL unavailable: {h3_exc})",
                        })
                    except ImportError:
                        self._send_json(502, {"error": f"H3 VL character detection failed: {h3_exc}"})
                    return
            if not vision_url.endswith("/chat/completions"):
                base = vision_url.rstrip("/")
                vision_url = (
                    f"{base}/chat/completions"
                    if base.endswith("/v1")
                    else f"{base}/v1/chat/completions"
                )

            vision_model = (
                req.get("vision_model")
                or default_vision_model
                or ("vlm" if used_managed_vlm else "default")
            ).strip()
            content = [{"type": "text", "text": vlm_prompt}]
            for frame_url in frames[:6]:
                if isinstance(frame_url, str) and frame_url.startswith("data:image/"):
                    content.append({"type": "image_url", "image_url": {"url": frame_url}})

            payload = json.dumps({
                "model": vision_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": 1024,
            }).encode("utf-8")

            http_req = urllib.request.Request(
                vision_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(http_req, timeout=60) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                msg = resp_data["choices"][0]["message"]["content"]
                characters = _extract_json_array(msg)
                self._send_json(200, {"characters": characters})
            except urllib.error.URLError as exc:
                self._send_json(502, {"error": f"Vision backend unreachable: {exc}"})
            except Exception as exc:
                self._send_json(502, {"error": f"Failed to parse characters from VLM: {exc}"})

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
                    if save_dir and self.command == "GET" and resp.status == 200:
                        m = video_content_re.match(self.path)
                        if m:
                            os.makedirs(save_dir, exist_ok=True)
                            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", m.group(1))
                            out_path = os.path.join(save_dir, f"{safe_id}.mp4")
                            if not os.path.exists(out_path):
                                with open(out_path, "wb") as f:
                                    f.write(resp_body)
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
            if self.path == "/api/detect-characters":
                self._handle_detect_characters()
            else:
                if self.path == "/v1/videos" and vlm_manager is not None:
                    vlm_manager.stop()
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


def _find_free_port(start_port: int = 8082) -> int:
    for port in range(start_port, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port available for VLM server")


def _find_gufo_exe(explicit: str = "") -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for rel in (
        os.path.join("build", "windows-release", "gufo.exe"),
        os.path.join("build", "release", "gufo.exe"),
        os.path.join("build", "release", "gufo"),
    ):
        cand = os.path.join(repo_root, rel)
        if os.path.isfile(cand):
            return cand
    found = shutil.which("gufo.exe") or shutil.which("gufo")
    return found or ""


def _find_mmproj_for_model(model_path: str) -> str:
    model_dir = os.path.dirname(os.path.abspath(model_path))
    for search_dir in (model_dir, os.path.dirname(model_dir)):
        matches = sorted(glob.glob(os.path.join(search_dir, "mmproj*.gguf")))
        if matches:
            return matches[0]
    return ""


def _auto_discover_vlm() -> tuple[str, str]:
    search_roots = []
    hf_hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_hub):
        search_roots.append(hf_hub)
    for local_root in (r"C:\AI", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))):
        if os.path.isdir(local_root) and local_root not in search_roots:
            search_roots.append(local_root)
    if not search_roots:
        return "", ""

    candidates: list[tuple[int, str, str]] = []
    seen = set()
    for root in search_roots:
        for mmproj in glob.glob(os.path.join(root, "**", "mmproj*.gguf"), recursive=True):
            snap_dir = os.path.dirname(mmproj)
            parent_dir = os.path.dirname(snap_dir)
            ggufs = []
            for d in (snap_dir, parent_dir):
                ggufs.extend(glob.glob(os.path.join(d, "*.gguf")))
            for g in ggufs:
                bn = os.path.basename(g).lower()
                if (
                    bn.startswith("mmproj")
                    or bn.startswith("mtp-")
                    or bn.startswith("dspark-")
                    or "-draft" in bn
                    or "draft." in bn
                ):
                    continue
                shard_m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", bn)
                if shard_m and shard_m.group(1) != "00001":
                    continue
                norm_g = os.path.normcase(os.path.abspath(g))
                if norm_g in seen:
                    continue
                seen.add(norm_g)
                try:
                    sz = os.path.getsize(g)
                except OSError:
                    continue
                candidates.append((sz, g, mmproj))
    if not candidates:
        return "", ""
    # Prefer Qwen3.8-Flash-Next over Inkling when available, then smallest GGUF.
    candidates.sort(key=lambda item: (0 if "qwen3.8-flash-next" in item[1].lower() else 1, item[0]))
    _, best_model, best_mmproj = candidates[0]
    return best_model, best_mmproj


def main() -> int:
    parser = argparse.ArgumentParser(description="Gufo Video Studio WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="minimax-h3")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--vision-backend", default="")
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--gufo-exe", default="")
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--vlm-mmproj", default="")
    parser.add_argument("--vlm-port", type=int, default=0)
    parser.add_argument("--auto-vlm", action="store_true", help="Auto-discover a local GGUF VLM (enabled by default unless --no-vlm)")
    parser.add_argument("--no-vlm", action="store_true", help="Disable auto-discovery of a local GGUF VLM")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if args.parent_pid > 0:
        threading.Thread(target=_watch_parent, args=(args.parent_pid,), daemon=True).start()

    vision_backend = args.vision_backend
    vision_model = args.vision_model
    vlm_model_path = args.vlm_model
    vlm_mmproj_path = args.vlm_mmproj
    if not args.no_vlm and not vlm_model_path and not vision_backend:
        vlm_model_path, discovered_mmproj = _auto_discover_vlm()
        if not vlm_mmproj_path:
            vlm_mmproj_path = discovered_mmproj

    vlm_manager = None
    if vlm_model_path and not vision_backend:
        gufo_exe = _find_gufo_exe(args.gufo_exe)
        if gufo_exe:
            if not vlm_mmproj_path:
                vlm_mmproj_path = _find_mmproj_for_model(vlm_model_path)
            vlm_manager = ManagedVlmServer(
                gufo_exe=gufo_exe,
                model_path=vlm_model_path,
                mmproj_path=vlm_mmproj_path,
                port=args.vlm_port,
            )
            print(f"On-demand VLM ready: {os.path.basename(vlm_model_path)}", flush=True)
        elif args.vlm_model:
            print("Error: --vlm-model specified, but gufo.exe was not found.", file=sys.stderr, flush=True)
            return 1

    server = ReusableThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            args.backend,
            args.model,
            args.save_dir,
            vision_backend,
            vision_model,
            vlm_manager=vlm_manager,
        ),
    )
    print(f"Gufo Video WebUI listening on http://{args.host}:{args.port} -> {args.backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if vlm_manager is not None:
            vlm_manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
