# Ornith-1.5-35B benchmarks

Scope: native Windows port, single AMD Strix Halo (gfx1151, Ryzen AI Max+ 395,
110 GiB visible carve-out), build under `feature/window-native` with the
TheRock HIP 7.15 nightly runtime, 2026-09-29. `gufo bench` depth 0, C=1, and
`gufo serve llm` HTTP (chat completions, greedy, temperature 0, thinking off).
Greedy HTTP completions are hash-compared against the isolated AR reference;
every published row matches. The llama.cpp reference columns are TODO on this
host (the pinned reference runtime is Nix/Linux only; the Linux qualification
run fills them).

[Quality and measurement details](QUALITY.md)

> 2026-10-20 (later): batched speculative verification now reproduces the
> single-session arithmetic past the 8-row vector contract (see
> EXPERIMENTS.md), so multi-session MTP publishes again. Same host and
> method, concurrency 1/2/4, prepared cohorts, greedy completions
> hash-identical to the isolated AR reference on every published row.
> Multi-user AR (this build): Q4_K_M 69.6/109.8/178.7 and Q8_0
> 52.9/95.4/158.6 decode tok/s at C1/C2/C4. Multi-user MTP mixed
> (this build): Q4_K_M 85.4/122.6/152.3 - the C2 and C4 rows the
> 2026-09-29 run withheld now pass the gate - and Q8_0 63.9/104.3/155.0
> (C4 +4.4% over the pre-fix build). Q8_0 repetition publishes
> 63.5/106.2/148.4; the Q4_K_M repetition C1 gate cannot complete on
> this build (the AR reference round's greedy continuation emits EOS at
> ~102 of 128 tokens; see QUALITY.md). Four simultaneous 2048-token
> cold prefills aggregate 2046-2089 tok/s at every session count
> (prefill saturates the GPU; see EXPERIMENTS.md for the probe).
> · [Experiments](EXPERIMENTS.md)
> · [Quality](QUALITY.md)
>
> 2026-10-20: the grouped decode attention pass flattens the long-context
> decode slope; the single-user depth tables below are new, and the
> depth-0 rows supersede the notes above (Q4_K_M pp2048 2943, AR tg128
> 70.4, MTP tg128 101.8; Q8_0 pp2048 3212, AR tg128 53.6, MTP tg128
> 84.5). Greedy completions are hash-identical to the 27ad6a2 reference
> build at every measured depth. The prefill depth slope is unchanged —
> see [Experiments](EXPERIMENTS.md) for the rejected kernel variants.
> · [Experiments](EXPERIMENTS.md)
>
> 2026-10-19: the chained greedy MTP pass and the device-side routed
> dispatch supersede the single-user rows below (tables are
> driver-rendered and not yet re-rendered on Windows). Same host and
> method, interleaved same-binary A/B, greedy completions hash-identical
> to the isolated AR reference on every row: Q4_K_M pp2048 2951 → 3005
> (+1.9%), AR tg128 69.5 (unchanged), MTP tg128 98.6 → 101.9 (+3.3%),
> MTP tg512 105.4 → 110.4 (+4.7%); Q8_0 pp2048 3102 → 3233 (+4.2%), AR
> tg128 52.6 (unchanged), MTP tg128 82.5 → 84.4 (+2.4%), MTP tg512
> 86.4 → 86.8 (+0.4%). The whole table moved since the 2026-10-01 note
> (upstream merges: AR tg128 69.4, MTP tg128 98.6 at 2a289ca).
> · [Experiments](EXPERIMENTS.md)
>

## Single user, long context (2026-10-20)

Native `gufo bench`, one warmed sample per point, deterministic bench
tokens, depth = cached prefix prepared before the timed region, context
133376, C1. Build `feature/window-native` with the grouped decode
attention; greedy completions hash-identical to the 27ad6a2 reference
build at every row (AR and MTP, both artifacts). Prefill is
autoregressive in every mode; MTP changes decode only.

| Depth (tokens) | Q4_K_M pp2048 | Q4_K_M AR tg128 | Q4_K_M MTP tg128 | Q8_0 pp2048 | Q8_0 AR tg128 | Q8_0 MTP tg128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2943 | 70.4 | 101.8 | 3212 | 53.6 | 84.5 |
| 4,096 | 2768 | 68.9 | 102.3 | 2949 | 52.3 | 80.5 |
| 8,192 | 2496 | 66.8 | 99.7 | 2668 | 51.2 | 79.3 |
| 12,288 | 2231 | 65.4 | 95.0 | 2380 | 50.3 | 76.1 |
| 16,384 | 2060 | 62.9 | 95.9 | 2181 | 49.5 | 76.2 |
| 32,768 | 1551 | 57.5 | 82.1 | 1612 | 45.4 | 69.2 |
| 65,536 | 1040 | 49.3 | 68.6 | 1069 | 40.5 | 58.6 |
| 131,072 | 622 | 37.1 | 50.6 | 637 | 32.5 | 46.0 |

Against the 27ad6a2 reference build the decode rows at 131072 moved
28.0 → 37.1 (Q4_K_M AR), 29.1 → 50.6 (Q4_K_M MTP), 24.6 → 32.5
(Q8_0 AR), 29.4 → 46.0 (Q8_0 MTP); the AR degradation at 131K eases
from −60% to −47% (Q4_K_M) and −50% to −39% (Q8_0), and MTP from −71%
to −50% and −65% to −45%. Prefill rows are unchanged within session
noise (the dense prefill attention slope is architectural — a dense
model reads the whole prefix every chunk; the kernel-level variants
rejected on the way are recorded in [Experiments](EXPERIMENTS.md)).
For reference, Qwen3.8-Flash-Next (sparse-indexer attention, quantized
KV) reports −13.5% pp and −1.6% AR tg at 131072 in its tables.

## Single user, quantized KV (`--kv-cache q8_0`, 2026-10-20)

Same host, method and build as the table above with the opt-in Q8_0
key/value planes (`ef4326e`): the decode scores run as v_dot4 integer
dots over the cache codes against per-step quantized queries. One
warmed sample per point; the quantized completions are hash-identical
to the F16 rows at every depth (the lossless greedy property holds in
quant mode on this workload), and the oracle envelope is in
[QUALITY.md](QUALITY.md).

| Depth (tokens) | pp2048 | AR tg128 | MTP tg128 |
| ---: | ---: | ---: | ---: |
| 0 | 2970 | 69.2 | 87.0 |
| 4,096 | 2693 | 67.8 | 103.0 |
| 8,192 | 2407 | 63.2 | 98.0 |
| 12,288 | 1904 | 59.5 | 94.6 |
| 16,384 | 1754 | 59.5 | 96.7 |
| 32,768 | 1274 | 55.1 | 85.6 |
| 65,536 | 869 | 50.1 | 65.8 |
| 131,072 | 493 | 38.9 | 51.1 |

Read against the F16 table: the modes are a deep-context trade. Deep
token generation gains (AR +4.8% at 131072, +1.7% at 65536; MTP +4.3%
at 32768) and every KV-resident byte halves (2048 → 1072 per token),
while shallow MTP pays the per-cycle query-quantization launches
(−15% at d0) and deep prefill pays the fused kernel's in-kernel
dequant (pp −21% at 131072). F16 stays the default and the
mixed-workload pick; `--kv-cache q8_0` is the long-context and
memory-constrained option. Snapshots record the plane dtypes, so
quantized sessions never restore into F16 ones.

> 2026-10-01: the Q6_K decode pass supersedes the single-user AR/MTP rows
> below (tables are driver-rendered and not yet re-rendered on Windows):
> native pp2048 2922 +/- 27, tg128 AR 64.79 +/- 0.44, tg128 MTP 88.48 +/-
> 1.36 (d0) and 86.5 (d4096), DFlash2 open-ended 88.9; HTTP serve MTP
> decodes 67-70 tok/s at 56-63% acceptance. A/B against 479175f and the
> full measurement record: [Experiments](EXPERIMENTS.md). · [Experiments](EXPERIMENTS.md)

## Native vs HTTP (single user, depth 0)

Native is `gufo bench` (kernel-iteration path, direct engine calls); HTTP is
the serving path with a prepared 2040-token prompt (one token generated to
anchor the cache, then the measured request reuses the whole prefix).
Prefill is autoregressive in every mode; MTP changes decode only.

| Mode | Native pp2048 (tok/s) | HTTP pp (tok/s) | Native tg128 (tok/s) | HTTP tg128 (tok/s) | HTTP gain vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autoregressive | 2870.9 ± 14.7 | 2653.5 | 63.21 ± 0.09 | 62.11 | -1.7% |
| MTP, mixed text | 2887.4 ± 5.0 | 2618.8 | 78.90 ± 0.18 | 73.18 | -7.3% |
| MTP, repetitive text | — | 2618.3 | 78.90 ± 0.18 | 79.50 | +0.8% |

HTTP decode sits within 2% of native for AR; the MTP gap on mixed text is
host-side draft chaining (a full-vocabulary head pass plus a stream
synchronization per draft step, which the native bench also pays but the HTTP
request path amortizes worse). Draft acceptance: 63% mixed, 78% repetitive.
Earlier (2026-09-29, unprepared ~1.2k-token prompts) HTTP rows of pp 2379 /
tg 49.0 measured a shorter steady-state window with full prefills; the
prepared-cohort method above replaces them.

## Single user, autoregressive

Depth 0 only on this host; deeper cached-prefix sweeps belong to the Linux
qualification run.

<!-- bench:single-ar -->
| Ornith Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2653.48 | TODO | TODO | 62.11 | TODO | TODO |
<!-- /bench -->

## Single user, MTP

<!-- bench:single-mtp -->
| Ornith Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2618.81 | 73.18 | TODO | TODO | 79.50 | TODO | TODO |
<!-- /bench -->

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Every session is prepared (full prefill plus one anchored token, one session
at a time) before the measured cohort is released together; throughput sums
individual request decode rates. Every level's completions are hash-identical
to the isolated AR reference.

<!-- bench:multi-ar -->
| Ornith Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 61.67 | TODO | TODO |
| 2 | 94.48 | TODO | TODO |
| 4 | 150.36 | TODO | TODO |
| 6 | 172.35 | TODO | TODO |
| 8 | 198.26 | TODO | TODO |
<!-- /bench -->

Decode scales to 3.2x C1 at C8; the aggregate stops growing linearly because
the quantized GEMV kernels become multiply-accumulate-bound once several rows
share each weight read.

## Multiple users, MTP

Same prompts and cohort method as the AR table. C1/C2 complete and match the
AR reference exactly. C4 (mixed) and C6 (repetitive) fail the greedy-equivalence
gate: co-batched speculative verification crosses the 8-row vector-kernel
contract, whose tiled fallback is not bit-compatible, so wider levels are
withheld (see `artifacts/unavailable.json`). The engine fix is tracked in
[EXPERIMENTS.md](EXPERIMENTS.md).

<!-- bench:multi-mtp -->
| Ornith Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72.61 | TODO | TODO | 78.69 | TODO | TODO |
| 2 | 105.23 | TODO | TODO | 116.90 | TODO | TODO |
| 4 | TODO | N/A | N/A | 172.00 | TODO | TODO |
| 6 | TODO | N/A | N/A | TODO | N/A | N/A |
| 8 | TODO | N/A | N/A | TODO | N/A | N/A |
<!-- /bench -->

## Single user, DFlash (2026-10-01)

`--speculative dflash2` with the converted
[Ornith-1.5-35B-A3B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash)
draft (Q8_0 GGUF, `tools/ornith/convert_dflash_gguf.py`), Q4_K_M target, same
host and build as the tables above. The draft was trained against the BF16
target; verification stays lossless (greedy completions hash-match the AR
reference on every measured point, including natural prose).

| Metric | AR | MTP | DFlash adaptive (<=7) | DFlash fixed 2 |
| --- | ---: | ---: | ---: | ---: |
| pp2048 d0 (tok/s) | 2915.6 | 2864.9 | 2241.5 | — |
| tg128 d0 (tok/s) | 62.57 ± 0.41 | 78.76 ± 0.46 | 52.08 ± 0.69 | 61.85 |
| tg128 d4096 (tok/s) | 59.80 ± 0.11 | 85.69 ± 0.12 | 127.09 ± 2.74 | — |
| tg192 natural prose d0 (tok/s) | 62.60 | 73.60 | 53.70 | — |

DFlash prefill injects the tapped trunk rows into the draft ring after each
chunk (one host round trip per 2048-token chunk), a ~20% prefill cost that
MTP does not pay.

Acceptance (accepted/drafted): d0 repetitive bench 25% adaptive (59% at
width 1, 27% at position 2, 13% at position 3), d4096 repetitive bench 100%
(7/7 every cycle), natural prose lower than the bench rows. The d4096 row
regurgitates the repeated pattern, which flatters any drafter; treat it as an
upper bound rather than a general-text result. Conclusions:

- MTP (quantization-aware in-file draft) is the general-purpose winner for
  this model: +18-26% tg on mixed text at every depth.
- DFlash trades blows with AR at shallow depth on open-ended text and pays a
  ~20% prefill tax for the feature injection; it wins decisively when the
  continuation is highly predictable (2.1x AR, 1.5x MTP on the d4096 row).

## Q8_0 artifact (2026-09-29)

The `Ornith-1.5-35B-Q8_0.gguf` artifact (37.8 GB, 35.2 GiB resident — no
load-time dequantized copies) runs every projection on its native Q8_0
bytes: decode reads the raw file bytes through the vector kernels, prefill
runs the routed F16 WMMA tier with in-kernel Q8_0 decode, and the upload
is a raw stream (load ≈ 13 s against ≈ 19 s for Q4_K_M). Same build,
method and hash gates as the Q4_K_M tables above; artifacts in
`artifacts-q8/`.

| Metric | Q4_K_M | Q8_0 | Δ |
| --- | ---: | ---: | ---: |
| Native pp2048 (tok/s) | 2870.9 ± 14.7 | 3073.7 ± 9.9 | +7.1% |
| Native tg128 AR (tok/s) | 63.21 ± 0.09 | 52.29 ± 0.05 | -17.3% |
| Native tg128 MTP (tok/s) | 78.90 ± 0.18 | 81.54 ± 0.88 | +3.3% |
| HTTP pp2048 (tok/s) | 2653.5 | 2780.9 | +4.8% |
| HTTP tg128 AR (tok/s) | 62.11 | 51.01 | -17.9% |
| HTTP MTP, mixed (tok/s) | 73.18 | 63.0 | -13.9% |
| HTTP MTP, repetitive (tok/s) | 79.50 | 60.9 | -23.4% |

Both quants decode on the memory-bandwidth roofline, so single-stream AR
pays the artifact's 1.7× expert-byte traffic directly (-17%); the gap
closes as concurrency amortizes weight reads over MAC-bound batched GEMVs
(C4 -3%, C6 +2%, C8 -2%). Prefill exceeds Q4_K_M in both paths. HTTP MTP
gives the native MTP gain back through the host-side draft-chaining tax:
the sharper quant drafts more per cycle but accepts less (55% mixed / 58%
repetitive against Q4_K_M's 63% / 78%) — see EXPERIMENTS.md.

| Ornith Q8 AR<br>Users | Gufo AR (tok/s) | Q4_K_M AR (tok/s) |
| ---: | ---: | ---: |
| 1 | 51.77 | 61.67 |
| 2 | 81.20 | 94.48 |
| 4 | 145.75 | 150.36 |
| 6 | 176.22 | 172.35 |
| 8 | 194.34 | 198.26 |

### Q8_0 multiple users, MTP

Same cohort method and hash gates. Q8_0 passes the greedy-equivalence
gate through C4 on both workloads (Q4_K_M mixed trips at C4); C6 and C8
are withheld by the same 8-row vector-kernel contract
(`artifacts-q8/unavailable.json`).

| Ornith Q8 MTP<br>Users | Gufo mixed (tok/s) | Q4_K_M mixed (tok/s) | Gufo repetitive (tok/s) | Q4_K_M repetitive (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62.16 | 72.61 | 60.50 | 78.69 |
| 2 | 99.94 | 105.23 | 96.46 | 116.90 |
| 4 | 143.21 | withheld | 144.20 | 172.00 |

## Memory

Weights resident ≈ 21.2 GiB device (20.2 GiB artifact plus dequantized-Q8_0
dense copies and the Q6_K→Q8_0 head); peak device use 24.7 GiB at 4096 context
with one session and 26.1 GiB with `--sessions 8` — far inside the carve-out,
nothing spills to WDDM shared memory. The Q8_0 artifact holds 35.2 GiB of
weights with no dequantized copies beside them: 37.8 GiB device at
`--sessions 8`, 4096 context, same carve-out margin.

## TODO

- llama.cpp reference columns (AR and MTP, single-user and concurrency):
  Nix-pinned reference runtime; land with the Linux gfx1151 qualification run.
- Depth sweeps (4K-32K cached prefixes) for AR and MTP, loading time, and the
  deep-context memory rows: same.
- Engine work before wider MTP concurrency can be published: make batched
  speculative verification reproduce the single-session arithmetic (and, for
  serving correctness in general, make concurrent prefill chunk interleaving
  width-invariant — see EXPERIMENTS.md).
