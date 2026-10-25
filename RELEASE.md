# Gufo for Windows v0.24.0 (AMD Strix Halo gfx1151)

High-performance native local LLM inference engine optimized for AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`) on Windows 11. Built natively with AMD TheRock ROCm 10.1.0.

---

### 🚀 Highlights in v0.24.0

- **Full Upstream Sync with `gufo-org/gufo` v0.24.0**:
  - Native Windows x86-64 hosted support merged upstream in `src/core/platform/`.
  - Upstream feature integration across EmbeddingGemma 2, DFlash-2 speculative drafting, and DeepSeek V4 Flash.
- **AMD TheRock ROCm 10.1.0 Toolchain**:
  - Full compatibility with ROCm 10.1.0 (Clang 24.0.0git) on Windows 11 x64.
- **Prompt Processing & Prefill Optimization**:
  - Raised default scheduler prefill chunk from 512 to 2048 tokens (`--prefill-chunk 2048`).
  - Optimized dense F16 matrix core GEMM threshold (`kDenseF16MinRows = 1024`) for higher prefill throughput.
  - Streaming continuation disk persistence with zero staging stall (`--cache-disk-staging-bytes 0`).
- **Server Status & Busy API**:
  - Real-time monitoring endpoints `/status`, `/v1/status`, `/busy`, `/v1/busy`, and extended `/health` reporting server busy state, active slots, and capacity.
- **Chat Portal & Tool Permissions**:
  - llama.cpp-style Web UI with SQLite persistence and sandboxed per-user workspaces (`UserData/`).
  - Interactive approval prompts (`[✓ Approve & Run]` / `[✕ Deny]`) and interactive clarification inquiries (`ask_user`).

---

### 📊 Measured Benchmarks: Qwen3.8 Flash-Next (Windows 11, ROCm 10.1.0, gfx1151)

**Model:** `Qwen3.8 Flash-Next UD-Q4_K_XL` (103.69 GiB)  
**Hardware:** AMD Ryzen AI MAX+ 395 (`gfx1151`), Unified Memory  
**Platform:** Windows 11 x64, AMD TheRock ROCm 10.1.0 (HIP)  
**Model Load Time:** **17.71 s**

#### Autoregressive (AR) vs. Speculative (MTP) Performance

| Context Depth | Mode | pp2048 (t/s) | pp4096 (t/s) | pp32768 (t/s) | tg128 (t/s) | tg1024 (t/s) | tg4096 (t/s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **d0 (Pure AR)** | Autoregressive (Baseline) | 1,468.17 ± 0.00 | 1,511.16 ± 0.00 | 1,423.12 ± 0.00 | 52.69 ± 0.00 | 41.27 ± 0.00 | 45.57 ± 0.00 |
| **d16384 (16k)** | Speculative MTP | 1,340.66 ± 0.00 | 1,406.34 ± 0.00 | 1,382.49 ± 0.00 | 63.26 ± 0.00 *(+20%)* | 66.50 ± 0.00 *(+61%)* | 69.75 ± 0.00 *(+53%)* |
| **d32768 (32k)** | Speculative MTP | 1,310.61 ± 0.00 | 1,376.73 ± 0.00 | 1,358.81 ± 0.00 | 86.02 ± 0.00 *(+63%)* | 90.37 ± 0.00 *(+119%)* | 72.96 ± 0.00 *(+60%)* |

#### Key Takeaways:
1. **Prompt Processing (pp)**: Peak throughput of **1,511.16 t/s** at `pp4096`, sustaining **>1,310–1,423 t/s** across deep 16k–32k context histories with `kDenseF16MinRows = 1024` and `--cache-disk-staging-bytes 0`.
2. **Token Generation (tg)**: Single-user generation scales from ~41–53 t/s in pure AR up to **90.37 t/s** with MTP speculative decoding at 32k context (+119% throughput increase).

---

### 📦 Quick Start

```powershell
# 1. Run prerequisites check
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1

# 2. Launch Gufo with Qwen3.8 Flash-Next and MTP
.\build\release\gufo.exe serve --host 0.0.0.0 --port 8080 llm `
  --model "C:\models\Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf" `
  --draft-model "C:\models\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" `
  --draft-tokens 7 --prefill-chunk 2048 --cache-disk-staging-bytes 0
```
