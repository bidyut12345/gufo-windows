# Build and Benchmark Rules

Whenever building a release or test binary of Gufo, you must always run an MTP benchmark of `Qwen3.8-Flash-Next UD-Q4_K_XL`:
- Model: `C:\AI\gufo-data\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
- MTP Head: `--speculative mtp --mtp-model "C:\AI\gufo-data\models\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"`
- Always report the measured prefill and generation tok/s against the documented targets in `BENCHMARKS.md`.
