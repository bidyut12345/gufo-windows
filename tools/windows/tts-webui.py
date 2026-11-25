#!/usr/bin/env python3
"""Minimal WebUI and reverse proxy for Gufo's OpenAI-compatible speech API.

Serves a self-contained Qwen3-TTS UI on --port and proxies /v1/*, /ready,
/health and /props to the Gufo TTS server (--backend). Responses are streamed
through, so PCM speech plays while it is generated. Uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import ctypes
import http.client
import json
import os
import socket
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
  <div class="brand">Gufo Speech Studio <span id="variantLabel" class="badge"></span></div>
  <div class="badge"><span id="dot" class="dot"></span><span id="health">Connecting...</span></div>
</header>
<main>
  <section class="card">
    <label for="text">Text</label>
    <textarea id="text" placeholder="Type what to say...">Hello from Gufo, running Qwen3 text to speech on this machine.</textarea>
    <div class="hint"><span id="charCount">0</span> / 16384 bytes</div>

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
        <div><label for="maxTok">Max tokens</label><input id="maxTok" type="number" min="1" max="8192" placeholder="3000"></div>
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
const VARIANT = "DEFAULT_VARIANT_PLACEHOLDER";
let MODEL = "DEFAULT_MODEL_PLACEHOLDER";
const SAMPLE_RATE = 24000;
const $ = (id) => document.getElementById(id);
let busy = false;
let refWavB64 = null;

$("variantLabel").textContent = VARIANT;
const isBase = VARIANT === "Base";
const isDesign = VARIANT === "VoiceDesign";
$("cloneBox").classList.toggle("hidden", !isBase);
$("instructBox").classList.toggle("hidden", isBase);
$("instructReq").textContent = isDesign ? "(required)" : "(optional)";
if (isDesign) $("voiceBox").classList.add("hidden");

const updateCount = () => { $("charCount").textContent = new TextEncoder().encode($("text").value).length; };
$("text").addEventListener("input", updateCount); updateCount();

async function refreshHealth() {
  try {
    const r = await fetch("/ready", { cache: "no-store" });
    $("dot").className = "dot " + (r.ok ? "ok" : "");
    $("health").textContent = r.ok ? "Ready" : "Loading model...";
  } catch { $("dot").className = "dot err"; $("health").textContent = "Server offline"; }
}
async function loadVoices() {
  try {
    const m = await (await fetch("/v1/models")).json();
    if (m.data && m.data[0] && m.data[0].id) MODEL = m.data[0].id;
  } catch {}
  try {
    const v = await (await fetch("/v1/audio/voices")).json();
    const names = Array.isArray(v) ? v : (v.voices || []);
    const sel = $("voice"); sel.innerHTML = "";
    if (isBase) sel.add(new Option("Upload a reference below", ""));
    for (const n of names) sel.add(new Option(typeof n === "string" ? n : n.id, typeof n === "string" ? n : n.id));
    if (!names.length && !isBase) $("voiceBox").classList.add("hidden");
    if (isBase) sel.onchange = () => $("cloneBox").classList.toggle("hidden", !!sel.value);
  } catch {}
}
refreshHealth(); setInterval(refreshHealth, 5000); loadVoices();

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

  busy = true; $("runBtn").disabled = true;
  $("statusBar").className = "status-bar"; $("statusBar").textContent = "Generating...";
  const t0 = performance.now();
  try {
    const resp = await fetch("/v1/audio/speech", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { const e = await resp.json(); msg = (e.error && e.error.message) || msg; } catch {}
      throw new Error(msg);
    }
    let blob, firstMs = null, audioSec;
    if (stream) {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      let playAt = ctx.currentTime + 0.15, carry = null, total = 0;
      const chunks = []; const reader = resp.body.getReader();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (firstMs === null) { firstMs = performance.now() - t0; $("statusBar").textContent = `Playing (first audio after ${Math.round(firstMs)} ms)...`; }
        chunks.push(value); total += value.length;
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
      blob = pcmToWavBlob(chunks, total); audioSec = total / 2 / SAMPLE_RATE;
    } else {
      blob = await resp.blob(); audioSec = Math.max(0, (blob.size - 44) / 2 / SAMPLE_RATE);
    }
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


def make_handler(backend_url: str, default_model: str, variant: str):
    parsed = urllib.parse.urlsplit(backend_url)
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
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            headers["Connection"] = "close"
            conn = http.client.HTTPConnection(backend_host, backend_port, timeout=3600)
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
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except OSError as exc:
                err = json.dumps({"error": {"message": f"Backend unreachable: {exc}"}}).encode()
                try:
                    self._send_bytes(502, "application/json; charset=utf-8", err)
                except OSError:
                    pass
            finally:
                conn.close()

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                page = (HTML_PAGE.replace("DEFAULT_MODEL_PLACEHOLDER", default_model)
                        .replace("DEFAULT_VARIANT_PLACEHOLDER", variant))
                self._send_bytes(200, "text/html; charset=utf-8", page.encode("utf-8"))
            elif self.path == "/sw.js":
                js = (b"self.addEventListener('install',()=>self.skipWaiting());"
                      b"self.addEventListener('activate',e=>e.waitUntil(self.registration.unregister()));")
                self._send_bytes(200, "application/javascript; charset=utf-8", js)
            else:
                self._proxy()

        def do_POST(self) -> None:
            self._proxy()

    return ProxyHandler


def _watch_parent(pid: int) -> None:
    if pid <= 0 or os.name != "nt":
        return
    handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        os._exit(0)
    ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
    ctypes.windll.kernel32.CloseHandle(handle)
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
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    if args.parent_pid > 0:
        threading.Thread(target=_watch_parent, args=(args.parent_pid,), daemon=True).start()

    server = ReusableThreadingHTTPServer((args.host, args.port),
                                         make_handler(args.backend, args.model, args.variant))
    print(f"Gufo Speech WebUI listening on http://{args.host}:{args.port} -> {args.backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
