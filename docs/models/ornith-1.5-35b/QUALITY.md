# Ornith-1.5-35B quality

## Validation gates (2026-09-29, Windows port build)

- Scalar oracle: the float32 reference (`gufo_qwen35moe_reference`) is
  token-identical with llama.cpp greedy completion on the same raw prompt
  ("The capital of France is" → " Paris.", llama.cpp b11243 CPU build).
  This pins the graph semantics: pre-norm residual + post-attention norm,
  Gated DeltaNet with SiLU output gate and tiled value-head pairing
  (v-head h reads key head h % 16), gated GQA with sigmoid output gate,
  softmax top-8 renormalized MoE with gated shared expert.
- GPU vs oracle: per-position logits agree at cosine 0.988–0.996 with
  matching argmax over mixed prompt positions (residual gap = the GPU's
  Q8_0-requantized dense weights, F16 activations/KV, and WMMA attention
  against exact F32 on the original Q4_K).
- Batched-prefill gate: `gufo bench --validate-prefill 64` passes
  (scalar_winner_rank=1, cosine 0.9977, finite).
- Native decode views (2026-09-29 perf pass): GPU vs oracle cosine
  0.979-0.996 across two prompts; 6/8 argmax positions agree exactly and
  the two disagreements are near-ties (both paths quantize activations;
  the native path keeps the artifact's own Q4_K weights). Greedy streams
  are deterministic under repetition. MTP draft acceptance restored at
  63/88 (72%) after fixing the split-projection view leak.
- Operator tests: `qwen35moe.attention_ops` (16-head WMMA vs per-token
  reference, chunk-invariance exact) and `qwen35moe.gdn_ops` (32 value
  heads, conv + row-split contracts) pass on gfx1151.

## Q8_0 artifact gates (2026-09-29)

- Upload identity: Q8_0 tensors stream to the device unchanged (no
  requantization), so the GPU reads the artifact's exact bytes and the
  float32 oracle dequantizes the same blocks.
- GPU vs oracle on the identical 9-token prompt: worst cosine +0.9965
  (Q4_K_M on the same tokens: +0.9890) — the higher-precision artifact
  tracks the oracle more closely than the Q4_K_M build does.
- `gufo bench --validate-prefill 8`: cosine 0.99957, scalar_winner_rank 1.
- `qwen35moe.config`, `qwen35moe.mtp_sampling`, `qwen35moe.gdn_ops` and
  `qwen38_flash_next.routed_wmma_ops` (which covers the routed F16 Q8_0
  decode the prefill tier uses) pass. `qwen35moe.attention_ops` fails its
  graph-replay preparation case at a 2-ulp half difference — it fails
  identically with the Q8_0 changes stashed, so it is pre-existing on this
  host and unrelated. The routed_wmma test binary needs the HIP runtime
  DLLs staged beside it, like the qwen35moe tests get via
  `gufo_stage_hip_runtime`.
- llama.cpp b11243's Q8_0 greedy is NOT a ground truth for this artifact:
  it diverges from the F32 oracle at a 0.6-logit near-tie that the GPU
  and oracle both resolve the same way. The oracle gates above are the
  contract.
- Driver hash gates: single-user AR and MTP completions and the multi-ar
  C1-C8 cohort all hash-match the isolated AR reference
  (`artifacts-q8/`).

## Known gaps

- Serve-path draft-acceptance counters were verified manually (63–79%);
  the automated sweep for long-context MTP-after-restore parity from the
  Flash-Next port applies here too.
- The MTP cost priors (`mtp_costs.hpp`) still hold the Flash-Next
  measurements; the draft-length controller converges online, but the
  priors should be re-measured for this model. Do not retune them before
  the per-draft cost drops: at the measured ~5.9 ms/draft (full head pass
  plus a stream synchronization per draft), the current priors already
  steer to the throughput-optimal short chains (forced 7-draft cycles
  measure 47.9 tok/s against 79 adaptive).

## Benchmark method (2026-09-29, Windows host)

- Driver: `tools/bench/model-bench.py` with
  `docs/models/ornith-1.5-35b/artifacts/bench.json` (Gufo-only; no llama.cpp
  reference exists on this host — reference columns are TODO until the Linux
  qualification run). Server: `gufo serve llm --think off
  --max-pending-per-client 8 --sessions C`, greedy, seed-free temperature 0.
- Native numbers: `gufo bench -p 2048 -n 128 -r 3`, AR and
  `--speculative mtp --min-draft-tokens 1 --draft-tokens 7`.
- Concurrency cohorts prepare every session with the exact 2040-token prose
  prompt plus one anchored token, one session at a time, then release the
  cohort together (`prepared_prefill_sequential`). Decoding one session at a
  time during preparation is required for exactness: two concurrent 2040-token
  prefills interleave scheduler chunks, and the width-sensitive prefill routes
  (the F32 router's hipBLAS sgemv among them) shift logits by ulps, flipping
  rare near-tie expert selections; the resulting session state decodes to
  different greedy text than the isolated reference. Sequentially prepared
  cohorts are hash-identical at every AR width (C1-C8) and for speculative
  verification up to the 8-row vector-kernel contract.
- Withheld rows: batched speculative verification co-batching sessions beyond
  8 total rows leaves the vector kernels for the tiled fallback, which does
  not reproduce the vector path's arithmetic; mixed text trips the gate at C4,
  repetitive text (wider logit margins) at C6. See `artifacts/unavailable.json`.
- Completion hashes: every published Gufo row (single-user AR, single-user
  MTP both workloads, multi-user AR C1-C8, multi-user MTP C1/C2 mixed and
  C1-C4 repetitive) matches the isolated AR C1 reference hash exactly.
- Reproduction: `python tools/bench/model-bench.py --model ornith-1.5-35b
  --gufo <binary> --gguf <artifact> run --target gufo --table single-ar,
  single-mtp,multi-ar,multi-mtp`, then `render --no-charts`.
  Windows needs the `servers.py` terminate fallback (no process groups).

## Chained MTP gates (2026-10-19)

- Every greedy MTP completion (Q4_K_M and Q8_0, tg128 r3 and tg512,
  interleaved same-binary A/B) hash-matches the isolated AR reference;
  the lossless property is unchanged by the device-side draft chain and
  the `kVerifyGreedy` frontier argmax. AR and pp2048 move within noise.
- `qwen35moe.mtp_sampling` covers the new width-controller semantics
  (probe-reaches-limit under full acceptance, score-optimal steady
  width, restore/backoff invariants). `qwen35moe.config`,
  `qwen35moe.gdn_ops` pass; `qwen35moe.attention_ops` fails only its
  documented pre-existing 2-ulp replay preparation case. Snapshot
  payloads are unchanged: the chain and prediction buffers are
  executor scratch, and the controller state rides the existing
  `MtpLengthState` (old payloads restore unchanged).
- Width-controller costs (`mtp_costs.hpp`) are now Ornith-measured for
  the chained path; the served greedy C1 path additionally learns cycle
  costs online (seeded, sampled and penalty-greedy requests keep the
  static table and the fixed policy for deterministic replay).

## Q6_K decode kernel gates (2026-10-01)

- Kernel vs exact host dequant over real artifact weights: the LM head's
  first 64 rows (Q6_K, extracted from the GGUF) through
  `qfn_mmq_moe_vec` E=1 land at worst 0.55% of row peak against the
  F64 accumulate of `DequantizeQ6_K` (Q4_K attn_q on the same harness:
  0.50%); a 5-token batched call with distinct activations per row is
  0.43% — the y-dim token batching shares the per-row reduction, so
  batching does not change the sum.
- `qwen38_flash_next.routed_wmma_ops` extends to Q6_K blocks (random
  ql/qh/scales with a real F16 scale; scalar, paired, batched widths
  2-640 including ragged 2561-row tails; inactive-expert zeroes and the
  nonfinite-scale-to-zero contract via the block's own scale field at
  byte 208). The run still aborts at the two pre-existing Windows
  failures first (vector-grouping width 2, paired SwiGLU 1 ulp) with
  byte-identical signatures to baseline; with those gated exactly as on
  base, the suite passes including all Q6_K cases.
- Decode faithfulness improves: the Q6_K parts are the artifact's own
  bytes now, where the previous build read a Q6_K→Q8_0 requantization.
  End-to-end evidence: greedy AR, MTP and DFlash2 completions are
  sha-identical to each other; the BF16-trained DFlash draft's
  acceptance on open-ended text rises from 25% to 61%; MTP acceptance
  on prose holds at 63%.
- `gufo bench --validate-prefill 1` at pp2048: rmse 0, cosine 1.0,
  max_error 0 (prefill keeps the Q8_0/WMMA tier unchanged).
- `qwen35moe.gdn_ops` passes; `qwen35moe.attention_ops` shows only its
  documented pre-existing 2-ulp replay signature.

## Long-context depth gates (2026-10-20)

Native bench sweeps at depths 0-131072 (context 133376, C1, one warmed
sample per point, deterministic bench tokens) after the grouped decode
attention landed. Every greedy completion is hash-identical to the
27ad6a2 reference build at the same depth, for both artifacts and both
AR and MTP — the decode attention change is bitwise and the long-context
outputs did not move. AR and MTP completions stay hash-equal per depth
and quant (the lossless property holds at 131K). MTP acceptance is
depth-stable (82-84% of drafted, ~2.3 tokens per cycle at every depth);
the deep MTP decode gain is throughput, not acceptance. `--validate-prefill 8`
at d32768 reports cosine 0.999468, scalar_winner_rank 1 — the same
values as the retained 2026-10-19 prefill pass. `qwen35moe.attention_ops`
chunk cases (n=136 and n=2048, full vs chunked prefill against the
per-token reference) are bit-exact (`differing=0 max=0`); the suite
fails only its documented pre-existing 2-ulp replay preparation case.

## Quantized KV gates (2026-10-20)

The opt-in `--kv-cache q8_0` / `q8_0-q4_k` modes trade numerical
fidelity for half the per-token KV bytes. Gates, all on ef4326e:
`qwen35moe.attention_ops` gains an int8-dot oracle case — keys stored
through the preparation's Q8_0 layout, queries quantized by the same
kernel decode uses, and the gated context compared against dequantized
float64 attention: worst absolute error 5.5e-6 (n=4@d4096, n=32@d9000);
the case runs before the suite's documented pre-existing preparation
failure. F16 mode stays bitwise: greedy completions hash-identical to
the reference build at d0/d32K/d131072, both artifacts. Quantized mode
on the bench prompt produces completions hash-identical to F16 at
d32768 and d131072 (AR and MTP), and validate-prefill reports cosine
0.999539, scalar_winner_rank 1 at d32768 — the same envelope as the
F16 prefill. Natural-text probes: coherent and factually correct with
minor near-tie phrasing drift versus F16; treat the modes as lossy and
qualify quality-sensitive workloads on F16. Snapshots record both
plane dtypes, so a quantized session never restores into an F16 one or
vice versa.

## Batched-verify gates (2026-10-20)

Co-batched speculative verification now reproduces the single-session
arithmetic past the 8-row vector contract (batch paths read the decode
kernels' native bytes; see EXPERIMENTS.md). Gates on the fixed build,
prepared cohorts, greedy, versus the isolated AR reference: Q4_K_M
multi-mixed MTP passes C1/C2/C4 (the C2/C4 rows the 2026-09-29 run
withheld now publish), Q8_0 passes multi-mixed and multi-repetition
C1-C4 on both workloads; every published completion is hash-identical
to its isolated AR reference. F16 single-session paths are untouched
(d0 MTP sha identical to the pre-fix build); `qwen35moe`
config/mtp_sampling/gdn_ops pass and `qwen35moe.attention_ops` shows
only its documented pre-existing 2-ulp replay preparation case.
Known gap: the synthetic repetition workload's AR reference round on
the Q4_K_M artifact stops at ~102 of 128 tokens (greedy EOS, see
EXPERIMENTS.md), so that gate cannot complete on this build; not a
batch-path defect (single-session, both modes agree).
