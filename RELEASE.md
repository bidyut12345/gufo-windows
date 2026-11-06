# Gufo for Windows v0.31.0 (AMD Strix Halo gfx1151)

High-performance native local LLM inference engine optimized for AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`) on Windows 11. Built natively with AMD TheRock ROCm 10.1.0.

---

### 🚀 Highlights in v0.31.0

- **Full Upstream Sync with `gufo-org/gufo` v0.31.0**:
  - **Modular Model Extensions (API 0.2.7)**: Discovered source extension architecture separating core inference from model packages. Fully discovers and integrates:
    - `qwen38_flash_next` (Qwen 3.8 Flash-Next)
    - `qwen` (Qwen 3.8 27B / Qwen family)
    - `deepseek_v4_flash` (DeepSeek V4 Flash)
    - `embedding_gemma2` (EmbeddingGemma 2 text/audio/video)
    - `minimax_h3` (MiniMax H3 audio & video generation)
    - `qwen3_asr` (Qwen3 ASR speech recognition)
    - `qwen3_tts` (Qwen3 TTS speech synthesis)
    - `qwen_image_21` (Qwen Image 2.1 generation)
  - **Unified Text Provider & Runner Architecture**: Text serving and benchmarking now leverage `ext::TextModelRunner` and `ext::ProbeTextProvider` under a shared scheduler, prompt cache, and HTTP layer.
  - **Flash-Next Dynamic YaRN Context Scaling**: Native YaRN extension (`--enable-yarn`) supporting contexts well beyond native 262k tokens, dynamically rotating cached keys.
  - **DFlash-2 & DSpark Speculation**: Multi-token prediction and speculative decoding support across models.
- **Windows 11 & AMD TheRock ROCm 10.1.0 Toolchain**:
  - Full native build and runtime compatibility with ROCm 10.1.0 (Clang 24.0.0git) on Windows 11 x64.
  - App-local deployment of all required ROCm, vcpkg, and OpenMP runtime dependencies.
  - Integrated `qwen35moe` MoE engine and restored unsloth `UD-IQ4_XS` expert kernels (`IQ3_S`, `IQ4_XS`, `IQ4_NL`).
- **Web Chat Portal & Tool Permissions**:
  - Built-in Web UI with SQLite conversation persistence and sandboxed per-user workspaces (`UserData/`).
  - `/status`, `/busy`, and extended `/health` endpoints for orchestration and monitoring.

---

### 📊 Measured Benchmarks: Qwen3.8 Flash-Next (Windows 11, ROCm 10.1.0, gfx1151)

**Model:** `Qwen3.8 Flash-Next UD-Q4_K_XL` (103.69 GiB)  
**Speculative Head:** `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`  
**Hardware:** AMD Ryzen AI MAX+ 395 (`gfx1151`), 128 GB Unified Memory  
**Platform:** Windows 11 x64, AMD TheRock ROCm 10.1.0 (HIP)  
**Model Load Time:** **17.47 s**

| Workload | Mode | Measured Throughput | Target Baseline (`BENCHMARKS.md`) |
|:---|:---|:---:|:---:|
| **pp2048** | Prompt Processing (Prefill) | **1,634.12 tok/s** | ~1,635 – 1,870 tok/s |
| **tg128** | Speculative Decode (MTP) | **61.57 tok/s** | ~62.6 tok/s (Candidate PR #877) |

---

### 📦 Quick Start

```powershell
# 1. Run prerequisites check
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1

# 2. Benchmark Qwen3.8 Flash-Next with MTP
.\bin\gufo.exe bench -p 2048 -n 128 `
  --model "C:\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" `
  --speculative mtp `
  --mtp-model "C:\models\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"

# 3. Launch HTTP Server & Web Chat Portal
.\bin\gufo.exe serve --host 0.0.0.0 --port 8080 llm `
  --model "C:\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" `
  --speculative mtp `
  --mtp-model "C:\models\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
```
