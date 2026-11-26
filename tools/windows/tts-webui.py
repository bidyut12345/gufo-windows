#!/usr/bin/env python3
"""Minimal WebUI and reverse proxy for Gufo's OpenAI-compatible speech API.

Serves a self-contained Qwen3-TTS UI on --port and proxies /v1/*, /ready,
/health and /props to the Gufo TTS server (--backend). Responses are streamed
through, so PCM speech plays while it is generated. Uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import glob
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gufo Speech Studio</title>
<style>
:root {
  --bg: #0f1115; --panel: #171a21; --panel-2: #1e232d; --border: #2a3140;
  --fg: #eef2f7; --muted: #98a2b3; --accent: #3b82f6; --accent-hover: #2563eb;
  --ok: #22c55e; --warn: #f59e0b; --err: #ef4444; --radius: 12px;
}
@media (prefers-color-scheme: light) {
  :root { --bg: #f4f5f7; --panel: #fff; --panel-2: #f8fafc; --border: #e2e8f0;
          --fg: #0f172a; --muted: #64748b; --accent: #2563eb; --accent-hover: #1d4ed8; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
         gap: 12px; padding: 14px 24px; background: var(--panel); border-bottom: 1px solid var(--border); }
.brand { font-weight: 700; font-size: 17px; }
.badge { display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px; border-radius: 999px;
         font-size: 12px; font-weight: 600; background: var(--panel-2); border: 1px solid var(--border); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }
.dot.ok { background: var(--ok); } .dot.err { background: var(--err); }
main { max-width: 1240px; margin: 0 auto; padding: 20px 24px 40px; display: grid;
       grid-template-columns: 440px 1fr; gap: 20px; align-items: start; }
@media (max-width: 960px) { main { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; }
label { display: block; font-size: 12px; font-weight: 600; color: var(--muted); margin: 12px 0 5px; }
input, textarea, select { width: 100%; font: inherit; color: var(--fg); background: var(--panel-2);
       border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
textarea { min-height: 120px; resize: vertical; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.hint { font-size: 11px; color: var(--muted); margin-top: 4px; }
.hidden { display: none !important; }
details { margin-top: 14px; border-top: 1px solid var(--border); padding-top: 10px; }
summary { cursor: pointer; font-weight: 600; color: var(--muted); }
.check { display: flex; align-items: center; gap: 8px; margin-top: 12px; font-size: 13px; }
.check input { width: auto; }
button.primary { margin-top: 16px; width: 100%; padding: 11px; border: 0; border-radius: 9px;
       background: var(--accent); color: #fff; font: inherit; font-weight: 700; cursor: pointer; }
button.primary:hover { background: var(--accent-hover); }
button.primary:disabled { opacity: .55; cursor: wait; }
.status-bar { margin-bottom: 14px; padding: 10px 12px; border-radius: 9px; background: var(--panel-2);
       border: 1px solid var(--border); color: var(--muted); }
.status-bar.error { color: var(--err); border-color: var(--err); }
.item { border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-bottom: 12px; background: var(--panel-2); }
.item .text { white-space: pre-wrap; word-break: break-word; margin-bottom: 8px; }
.item .meta { font-size: 11px; color: var(--muted); margin-bottom: 8px; }
.item audio { width: 100%; }
.item a { color: var(--accent); font-size: 12px; font-weight: 600; }
.empty { color: var(--muted); text-align: center; padding: 40px 0; }
</style>
</head>
<body>
<header>
  <div class="brand">Gufo Speech Studio <select id="variantSelect" class="badge" style="width:auto;padding:4px 10px;margin-left:6px" title="Switch Qwen3-TTS model"></select></div>
  <div class="badge"><span id="dot" class="dot"></span><span id="health">Connecting...</span></div>
</header>
<main>
  <section class="card">
    <label for="text">Text</label>
    <textarea id="text" placeholder="Type what to say...">Hello from Gufo, running Qwen3 text to speech on this machine.</textarea>
    <div class="hint"><span id="charCount">0</span> chars (<span id="chunkCount">1</span> chunk)</div>

    <div class="row">
      <div id="voiceBox">
        <label for="voice">Voice</label>
        <select id="voice"></select>
      </div>
      <div>
        <label for="language">Language</label>
        <select id="language">
          <option value="auto">Auto</option>
          <option value="english" selected>English</option>
          <option value="chinese">Chinese</option>
          <option value="japanese">Japanese</option>
          <option value="korean">Korean</option>
          <option value="german">German</option>
          <option value="french">French</option>
          <option value="russian">Russian</option>
          <option value="portuguese">Portuguese</option>
          <option value="spanish">Spanish</option>
          <option value="italian">Italian</option>
        </select>
      </div>
    </div>

    <div id="instructBox">
      <label for="instructions">Voice / style instructions <span id="instructReq"></span></label>
      <textarea id="instructions" style="min-height:70px" placeholder="e.g. A warm, calm female narrator in her thirties, speaking slowly."></textarea>
    </div>

    <div id="cloneBox">
      <label for="refFile">Reference recording (voice to clone)</label>
      <input id="refFile" type="file" accept="audio/*">
      <div class="hint">Any format the browser can decode; converted to WAV locally. 3-20 s of clean speech works best.</div>
      <audio id="refPreview" controls class="hidden" style="width:100%;margin-top:8px"></audio>
      <label for="refText">Transcript of the recording</label>
      <textarea id="refText" style="min-height:60px" placeholder="Exact words spoken in the reference recording (closest match)."></textarea>
      <div class="check"><input id="xvecOnly" type="checkbox"><label for="xvecOnly" style="margin:0">No transcript: copy only the speaker's voice</label></div>
    </div>

    <div class="check"><input id="stream" type="checkbox" checked><label for="stream" style="margin:0">Stream (play while generating)</label></div>

    <details>
      <summary>Sampling</summary>
      <div class="row-3">
        <div><label for="temperature">Temperature</label><input id="temperature" type="number" step="0.05" min="0" placeholder="0.9"></div>
        <div><label for="topK">Top-k</label><input id="topK" type="number" min="0" placeholder="50"></div>
        <div><label for="topP">Top-p</label><input id="topP" type="number" step="0.01" min="0" max="1" placeholder="1.0"></div>
      </div>
      <div class="row-3">
        <div><label for="repPen">Repetition pen.</label><input id="repPen" type="number" step="0.01" min="0" placeholder="1.05"></div>
        <div><label for="seed">Seed</label><input id="seed" type="number" min="0" placeholder="random"></div>
        <div><label for="maxTok">Max tokens</label><input id="maxTok" type="number" min="1" max="32768" placeholder="auto"></div>
      </div>
    </details>

    <button id="runBtn" class="primary">Speak</button>
  </section>

  <section class="card">
    <div id="statusBar" class="status-bar">Ready.</div>
    <div id="results"><div class="empty" id="empty">Generated speech appears here.</div></div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
let VARIANT = "DEFAULT_VARIANT_PLACEHOLDER";
let MODEL = "DEFAULT_MODEL_PLACEHOLDER";
const SAMPLE_RATE = 24000;
let busy = false;
let switching = false;
let refWavB64 = null;
let isBase = false;
let isDesign = false;
let voicesLoaded = false;

function applyVariant(v, m) {
  VARIANT = v;
  if (m) MODEL = m;
  isBase = VARIANT === "Base";
  isDesign = VARIANT === "VoiceDesign";
  $("cloneBox").classList.toggle("hidden", !isBase);
  $("instructBox").classList.toggle("hidden", isBase);
  $("instructReq").textContent = isDesign ? "(required)" : "(optional)";
  $("voiceBox").classList.toggle("hidden", isDesign);
  voicesLoaded = false;
}
applyVariant(VARIANT, MODEL);

function splitSpeechText(text, maxChars = 14000) {
  const chunks = [];
  let cur = "";
  const flushCur = () => {
    const s = cur.trim();
    if (s) chunks.push(s);
    cur = "";
  };
  const appendUnit = (unit, sep) => {
    const u = unit.trim();
    if (!u) return;
    if (u.length > maxChars) {
      flushCur();
      let rest = u;
      while (rest.length > maxChars) {
        let cut = rest.lastIndexOf(" ", maxChars);
        if (cut < maxChars * 0.4) cut = maxChars;
        chunks.push(rest.slice(0, cut).trim());
        rest = rest.slice(cut).trim();
      }
      if (rest) cur = rest;
      return;
    }
    const next = cur ? cur + sep + u : u;
    if (next.length > maxChars) {
      flushCur();
      cur = u;
    } else {
      cur = next;
    }
  };
  for (const para of text.split(/\n+/)) {
    const p = para.trim();
    if (!p) continue;
    if (p.length <= maxChars) {
      appendUnit(p, "\n");
      continue;
    }
    const sentences = p.match(/[^.!?。！？]+(?:[.!?。！？]+["'”’)\]]*\s*|$)/g) || [p];
    for (const sent of sentences) appendUnit(sent, " ");
  }
  flushCur();
  return chunks.length ? chunks : [text.trim()];
}

const updateCount = () => {
  const v = $("text").value.trim();
  $("charCount").textContent = $("text").value.length;
  $("chunkCount").textContent = v ? splitSpeechText(v).length : 0;
};
$("text").addEventListener("input", updateCount); updateCount();

async function loadVariantList() {
  const sel = $("variantSelect");
  try {
    const r = await fetch("/webui/models", { cache: "no-store" });
    if (!r.ok) throw new Error();
    const d = await r.json();
    sel.innerHTML = "";
    for (const v of (d.available || [VARIANT])) sel.add(new Option(v, v));
    sel.value = d.current || VARIANT;
    sel.disabled = !d.can_switch || (d.available || []).length <= 1;
  } catch {
    sel.innerHTML = "";
    sel.add(new Option(VARIANT, VARIANT));
    sel.disabled = true;
  }
}

$("variantSelect").onchange = async () => {
  const next = $("variantSelect").value;
  if (!next || next === VARIANT || busy || switching) {
    $("variantSelect").value = VARIANT;
    return;
  }
  switching = true;
  $("variantSelect").disabled = true;
  $("runBtn").disabled = true;
  $("dot").className = "dot";
  $("health").textContent = `Switching to ${next}...`;
  $("statusBar").className = "status-bar";
  $("statusBar").textContent = `Switching model to ${next}...`;
  try {
    const r = await fetch("/webui/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: next })
    });
    const d = await r.json();
    if (!r.ok) throw new Error((d.error && d.error.message) || `HTTP ${r.status}`);
    applyVariant(d.variant, d.model);
    for (let i = 0; i < 60; i++) {
      await new Promise((res) => setTimeout(res, 250));
      try {
        const rr = await fetch("/ready", { cache: "no-store" });
        if (rr.ok) break;
      } catch {}
    }
    await refreshHealth();
    $("statusBar").textContent = `Switched to ${VARIANT}.`;
  } catch (e) {
    $("variantSelect").value = VARIANT;
    $("statusBar").className = "status-bar error";
    $("statusBar").textContent = `Switch failed: ${e.message}`;
  } finally {
    switching = false;
    $("variantSelect").disabled = false;
    $("runBtn").disabled = false;
  }
};

async function refreshHealth() {
  if (switching) return;
  try {
    const r = await fetch("/ready", { cache: "no-store" });
    $("dot").className = "dot " + (r.ok ? "ok" : "");
    $("health").textContent = r.ok ? "Ready" : "Loading model...";
    if (r.ok && !voicesLoaded) loadVoices();
  } catch { $("dot").className = "dot err"; $("health").textContent = "Server offline"; }
}
async function loadVoices() {
  try {
    const mr = await fetch("/v1/models");
    if (mr.ok) {
      const m = await mr.json();
      if (m.data && m.data[0] && m.data[0].id) MODEL = m.data[0].id;
    }
  } catch {}
  try {
    const vr = await fetch("/v1/audio/voices");
    if (!vr.ok) return;
    const v = await vr.json();
    const names = Array.isArray(v) ? v : (v.voices || []);
    const sel = $("voice"); sel.innerHTML = "";
    if (isBase) sel.add(new Option("Upload a reference below", ""));
    for (const n of names) sel.add(new Option(typeof n === "string" ? n : n.id, typeof n === "string" ? n : n.id));
    if (!names.length && !isBase) $("voiceBox").classList.add("hidden");
    else $("voiceBox").classList.remove("hidden");
    if (isBase) sel.onchange = () => $("cloneBox").classList.toggle("hidden", !!sel.value);
    voicesLoaded = true;
  } catch {}
}
refreshHealth(); setInterval(refreshHealth, 3000); loadVariantList(); loadVoices();

// Decode any audio file, downmix to mono, resample to 24 kHz and encode a 16-bit WAV.
async function fileToWavB64(file) {
  const buf = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(buf.slice(0));
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * SAMPLE_RATE), SAMPLE_RATE);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  const rendered = await off.startRendering();
  const wav = encodeWav(rendered.getChannelData(0), SAMPLE_RATE);
  let bin = ""; const bytes = new Uint8Array(wav);
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}
function encodeWav(samples, rate) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return buf;
}
function pcmToWavBlob(chunks, total) {
  const pcm = new Uint8Array(total); let o = 0;
  for (const c of chunks) { pcm.set(c, o); o += c.length; }
  const head = new ArrayBuffer(44), v = new DataView(head);
  const w = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + total, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SAMPLE_RATE, true); v.setUint32(28, SAMPLE_RATE * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, total, true);
  return new Blob([head, pcm], { type: "audio/wav" });
}

$("refFile").onchange = async () => {
  refWavB64 = null;
  const f = $("refFile").files[0];
  if (!f) { $("refPreview").classList.add("hidden"); return; }
  $("refPreview").src = URL.createObjectURL(f); $("refPreview").classList.remove("hidden");
  try { refWavB64 = await fileToWavB64(f); $("statusBar").textContent = "Reference recording loaded."; }
  catch (e) { $("statusBar").className = "status-bar error"; $("statusBar").textContent = "Could not decode reference: " + e.message; }
};

function addResult(blob, text, meta) {
  $("empty")?.remove();
  const url = URL.createObjectURL(blob);
  const div = document.createElement("div"); div.className = "item";
  const t = document.createElement("div"); t.className = "text"; t.textContent = text;
  const m = document.createElement("div"); m.className = "meta"; m.textContent = meta;
  const a = document.createElement("audio"); a.controls = true; a.src = url;
  const d = document.createElement("a"); d.href = url; d.download = `gufo-tts-${Date.now()}.wav`; d.textContent = "Download WAV";
  div.append(t, m, a, d); $("results").prepend(div);
  return a;
}

function numberField(payload, id, key, parse) {
  const s = $(id).value.trim(); if (s !== "") payload[key] = parse(s);
}

$("runBtn").onclick = async () => {
  if (busy) return;
  const text = $("text").value.trim();
  if (!text) return;
  const payload = { model: MODEL, input: text, language: $("language").value };
  const voice = $("voice").value;
  if (!isDesign && voice) payload.voice = voice;
  const instr = $("instructions").value.trim();
  if (!isBase && instr) payload.instructions = instr;
  if (isDesign && !instr) { $("statusBar").className = "status-bar error"; $("statusBar").textContent = "VoiceDesign needs instructions describing the voice."; return; }
  if (isBase && !voice) {
    if (!refWavB64) { $("statusBar").className = "status-bar error"; $("statusBar").textContent = "Choose a reference recording to clone."; return; }
    payload.reference_audio = refWavB64;
    if ($("xvecOnly").checked) payload.voice_clone_mode = "speaker_embedding_only";
    else {
      const rt = $("refText").value.trim();
      if (!rt) { $("statusBar").className = "status-bar error"; $("statusBar").textContent = "Enter the reference transcript, or tick 'No transcript'."; return; }
      payload.voice_clone_mode = "icl"; payload.reference_text = rt;
    }
  }
  numberField(payload, "temperature", "temperature", parseFloat);
  numberField(payload, "topK", "top_k", (s) => parseInt(s, 10));
  numberField(payload, "topP", "top_p", parseFloat);
  numberField(payload, "repPen", "repetition_penalty", parseFloat);
  numberField(payload, "maxTok", "max_new_tokens", (s) => parseInt(s, 10));
  payload.seed = $("seed").value.trim() !== "" ? parseInt($("seed").value, 10) : Math.floor(Math.random() * 4294967295);
  const stream = $("stream").checked;
  payload.response_format = stream ? "pcm" : "wav";

  const textChunks = splitSpeechText(text);
  busy = true; $("runBtn").disabled = true;
  $("statusBar").className = "status-bar";
  $("statusBar").textContent = textChunks.length > 1 ? `Generating (chunk 1/${textChunks.length})...` : "Generating...";
  const t0 = performance.now();
  try {
    let firstMs = null, total = 0;
    const pcmChunks = [];
    const ctx = stream ? new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE }) : null;
    let playAt = ctx ? ctx.currentTime + 0.15 : 0;
    for (let idx = 0; idx < textChunks.length; idx++) {
      const chunkPayload = { ...payload, input: textChunks[idx] };
      if (textChunks.length > 1 && idx > 0) {
        $("statusBar").textContent = `Generating chunk ${idx + 1}/${textChunks.length}...`;
      }
      const resp = await fetch("/v1/audio/speech", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(chunkPayload) });
      if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try { const e = await resp.json(); msg = (e.error && e.error.message) || msg; } catch {}
        throw new Error(msg);
      }
      if (stream) {
        let carry = null;
        const reader = resp.body.getReader();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          if (firstMs === null) {
            firstMs = performance.now() - t0;
            $("statusBar").textContent = textChunks.length > 1
              ? `Playing chunk ${idx + 1}/${textChunks.length} (first audio after ${Math.round(firstMs)} ms)...`
              : `Playing (first audio after ${Math.round(firstMs)} ms)...`;
          }
          pcmChunks.push(value); total += value.length;
          let bytes = value;
          if (carry) { const m = new Uint8Array(carry.length + value.length); m.set(carry); m.set(value, carry.length); bytes = m; carry = null; }
          const even = bytes.length - (bytes.length % 2);
          if (even < bytes.length) carry = bytes.slice(even);
          if (!even) continue;
          const view = new DataView(bytes.buffer, bytes.byteOffset, even);
          const f = new Float32Array(even / 2);
          for (let i = 0; i < f.length; i++) f[i] = view.getInt16(i * 2, true) / 32768;
          const ab = ctx.createBuffer(1, f.length, SAMPLE_RATE); ab.copyToChannel(f, 0);
          const src = ctx.createBufferSource(); src.buffer = ab; src.connect(ctx.destination);
          playAt = Math.max(playAt, ctx.currentTime); src.start(playAt); playAt += ab.duration;
        }
      } else {
        const buf = await resp.arrayBuffer();
        if (buf.byteLength > 44) {
          const pcm = new Uint8Array(buf, 44);
          pcmChunks.push(pcm);
          total += pcm.length;
        }
      }
    }
    const blob = pcmToWavBlob(pcmChunks, total);
    const audioSec = total / 2 / SAMPLE_RATE;
    const dt = (performance.now() - t0) / 1000;
    const meta = `${VARIANT}${payload.voice ? " | " + payload.voice : ""} | ${payload.language} | seed ${payload.seed} | ${audioSec.toFixed(1)}s audio in ${dt.toFixed(1)}s` + (firstMs !== null ? ` | first audio ${Math.round(firstMs)} ms` : "");
    const el = addResult(blob, text, meta);
    if (!stream) el.play().catch(() => {});
    $("statusBar").textContent = `Done: ${audioSec.toFixed(1)}s of speech in ${dt.toFixed(1)}s (${(audioSec / dt).toFixed(2)}x real time).`;
  } catch (err) {
    $("statusBar").className = "status-bar error"; $("statusBar").textContent = `Error: ${err.message}`;
  } finally { busy = false; $("runBtn").disabled = false; }
};
</script>
</body>
</html>
"""

HOP_BY_HOP = {"host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade"}


VARIANT_MODELS = {
    "CustomVoice": "qwen3-tts-12hz-1.7b-customvoice",
    "VoiceDesign": "qwen3-tts-12hz-1.7b-voice-design",
    "Base": "qwen3-tts-12hz-1.7b-base",
}


class BackendManager:
    def __init__(self, backend_url: str, default_model: str, variant: str,
                 gufo_exe: str = "", model_location: str = "",
                 backend_port: int = 8091, context: int = 32768,
                 max_request_bytes: int = 33554432, npu: bool = False) -> None:
        self.backend_url = backend_url
        self.current_model = default_model
        self.current_variant = variant
        self.gufo_exe = gufo_exe
        self.model_location = model_location
        self.backend_port = backend_port
        self.context = context
        self.max_request_bytes = max_request_bytes
        self.npu = npu
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.variant_dirs = self._discover_variants() if gufo_exe and model_location else {}
        self.available = list(self.variant_dirs.keys()) or [variant]
        self.can_switch = bool(self.variant_dirs)

    def _discover_variants(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for v in VARIANT_MODELS:
            name = f"Qwen3-TTS-12Hz-1.7B-{v}"
            for candidate in (os.path.join(self.model_location, "models", name),
                              os.path.join(self.model_location, name)):
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    found[v] = candidate
                    break
        return found

    def start(self, variant: str) -> None:
        if not self.can_switch:
            return
        if variant not in self.variant_dirs:
            raise ValueError(f"Variant '{variant}' not found under '{self.model_location}'")
        self.stop()
        model_dir = self.variant_dirs[variant]
        model_name = VARIANT_MODELS[variant]
        cmd = [
            self.gufo_exe, "serve",
            "--host", "127.0.0.1",
            "--port", str(self.backend_port),
            "--max-request-bytes", str(self.max_request_bytes),
            "--no-webui",
            "tts",
            "--model", model_dir,
            "--served-model-name", model_name,
            "--served-model-name", "qwen3-tts",
        ]
        if self.context > 0:
            cmd += ["--context", str(self.context)]
        if self.npu:
            cmd.append("--npu")
        if variant == "Base":
            voice_dir = os.path.join(self.model_location, "voices")
            os.makedirs(voice_dir, exist_ok=True)
            for wav in sorted(glob.glob(os.path.join(voice_dir, "*.wav"))):
                stem = os.path.splitext(os.path.basename(wav))[0]
                cmd += ["--voice", f"{stem}={wav}"]
        print(f"Starting Gufo TTS server ({variant}): {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd)
        self.current_variant = variant
        self.current_model = model_name

    def switch(self, variant: str) -> tuple[str, str]:
        with self.lock:
            if not self.can_switch:
                raise RuntimeError("Model switching is only available when launched via run-qwen3-tts.ps1")
            if variant != self.current_variant or self.proc is None or self.proc.poll() is not None:
                self.start(variant)
            return self.current_variant, self.current_model

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None


def make_handler(manager: BackendManager):
    parsed = urllib.parse.urlsplit(manager.backend_url)
    backend_host = parsed.hostname or "127.0.0.1"
    backend_port = parsed.port or 80

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            pass

        def _send_bytes(self, status: int, ctype: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            headers["Connection"] = "close"
            conn = http.client.HTTPConnection(backend_host, backend_port, timeout=3600)
            headers_sent = False
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status, resp.reason)
                known_length = resp.getheader("Content-Length")
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                if known_length is not None:
                    self.send_header("Content-Length", known_length)
                    self.end_headers()
                    headers_sent = True
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                else:
                    # Stream PCM/SSE through as it arrives.
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    headers_sent = True
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.close_connection = True
            except (BrokenPipeError, ConnectionResetError, http.client.IncompleteRead):
                self.close_connection = True
            except (http.client.HTTPException, OSError) as exc:
                if headers_sent:
                    self.close_connection = True
                else:
                    err = json.dumps({"error": {"message": f"Backend unreachable: {exc}"}}).encode()
                    try:
                        self._send_bytes(502, "application/json; charset=utf-8", err)
                    except OSError:
                        pass
            finally:
                conn.close()

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                page = (HTML_PAGE.replace("DEFAULT_MODEL_PLACEHOLDER", manager.current_model)
                        .replace("DEFAULT_VARIANT_PLACEHOLDER", manager.current_variant))
                self._send_bytes(200, "text/html; charset=utf-8", page.encode("utf-8"))
            elif self.path == "/webui/models":
                payload = json.dumps({
                    "current": manager.current_variant,
                    "model": manager.current_model,
                    "available": manager.available,
                    "can_switch": manager.can_switch,
                }).encode("utf-8")
                self._send_bytes(200, "application/json; charset=utf-8", payload)
            elif self.path == "/sw.js":
                js = (b"self.addEventListener('install',()=>self.skipWaiting());"
                      b"self.addEventListener('activate',e=>e.waitUntil(self.registration.unregister()));")
                self._send_bytes(200, "application/javascript; charset=utf-8", js)
            else:
                self._proxy()

        def do_POST(self) -> None:
            if self.path == "/webui/switch":
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    req = json.loads(raw.decode("utf-8"))
                    target = req.get("variant", "")
                    v, m = manager.switch(target)
                    self._send_bytes(200, "application/json; charset=utf-8",
                                     json.dumps({"ok": True, "variant": v, "model": m}).encode("utf-8"))
                except ValueError as exc:
                    self._send_bytes(400, "application/json; charset=utf-8",
                                     json.dumps({"error": {"message": str(exc)}}).encode("utf-8"))
                except Exception as exc:
                    self._send_bytes(500, "application/json; charset=utf-8",
                                     json.dumps({"error": {"message": str(exc)}}).encode("utf-8"))
            else:
                self._proxy()

    return ProxyHandler


def _watch_parent(pid: int, manager: BackendManager | None = None) -> None:
    if pid <= 0 or os.name != "nt":
        return
    handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        if manager is not None:
            manager.stop()
        os._exit(0)
    ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
    ctypes.windll.kernel32.CloseHandle(handle)
    if manager is not None:
        manager.stop()
    os._exit(0)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Gufo Speech Studio WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--backend", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="qwen3-tts")
    parser.add_argument("--variant", default="CustomVoice", choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--gufo-exe", default="")
    parser.add_argument("--model-location", default="")
    parser.add_argument("--backend-port", type=int, default=8091)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--max-request-bytes", type=int, default=33554432)
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    manager = BackendManager(
        backend_url=args.backend,
        default_model=args.model,
        variant=args.variant,
        gufo_exe=args.gufo_exe,
        model_location=args.model_location,
        backend_port=args.backend_port,
        context=args.context,
        max_request_bytes=args.max_request_bytes,
        npu=args.npu,
    )
    atexit.register(manager.stop)
    if manager.can_switch:
        manager.start(args.variant)

    if args.parent_pid > 0:
        threading.Thread(target=_watch_parent, args=(args.parent_pid, manager), daemon=True).start()

    server = ReusableThreadingHTTPServer((args.host, args.port),
                                         make_handler(manager))
    print(f"Gufo Speech WebUI listening on http://{args.host}:{args.port} -> {args.backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
