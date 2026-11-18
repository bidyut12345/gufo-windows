# Ornith-1.5-35B experiments

- 2026-09-29: ported the qwen35moe runtime from the Flash-Next package by
  stripping the PLE n-gram, hyper-connection and sparse-indexer stages;
  attention runs dense only. Retained.
- Reusing the vendored qfn_mmq kernel build instead of duplicating it:
  retained (generic quant kernels; ds4 precedent).
- Native Q4_K routed down projections on the WMMA tier (48-row tiles):
  retained; Q6_K downs dequantize to Q8_0 at load.
- 2026-09-29 decode pass: dense decode-width projections keep a raw Q4_K
  byte view beside the Q8_0 wide-batch view and run `qfn_mmq_moe_vec` E=1
  (shared expert via `moe_gated_vec`); +11% AR (57.3 to 63.4 tok/s).
  Retained. The MTP draft's split fc_embedding/fc_hidden projections must
  NOT carry the combined tensor's whole-row view — the first cut leaked it
  and the draft proposals collapsed to 0% acceptance. Rejected: event-poll
  sync and pipeline-replay probes (WDDM overhead is not on the critical
  path; graph replay confirmed the decode graph itself is GPU-bound at
  ~17 ms/token, i.e. bandwidth).
- 2026-09-29 benchmark pass (native vs HTTP, concurrency): prepared-cohort
  HTTP decode lands within 2% of native AR (62.1 vs 63.2 tok/s) and scales
  to 198.3 tok/s aggregate at C8 with hash-exact completions. Retained
  (sequential session preparation during cohort setup is required; see
  QUALITY.md). Two engine findings, unfixed, both reproducible:
  (1) concurrent prefill requests interleave scheduler chunks and the
  width-sensitive prefill routes (F32 router hipBLAS sgemv; F16-route
  threshold at 2048 rows) shift logits by ulps, so a session prefilled
  alongside another can decode to different greedy text than the isolated
  reference — fix direction: width-invariant router kernel or chunk-invariant
  routes; (2) co-batched speculative verification beyond 8 total rows leaves
  the q8_1 vector kernels for the tiled fallback with different quantization
  and reduction, tripping the greedy-equivalence gate at C4 (mixed) / C6
  (repetitive) — fix direction: cap co-batched verify rows at the vector
  contract or make the tiled path reproduce it.
- 2026-09-29 Q8_0 artifact pass: every projection keeps its native Q8_0
  bytes end to end. Three engine changes, retained: (1) `DequantizeRow`
  decodes Q8_0 sources — the raw-block memcpy fallback would have
  corrupted the dense stacks and the alpha/beta stacks of a Q8_0 artifact;
  (2) `CopyDequantQ8_0` streams Q8_0 parts straight to the device instead
  of a dequant+requant round trip, and the split MTP projection drains the
  staging pipeline before it frees the combined buffer — that free was
  only safe while the upload was synchronous (first Q8_0 load died in the
  staging worker with `invalid argument`); (3) routed Q8_0 gate/up joins
  the F16 WMMA tier (the routed kernel already decoded Q8_0 downs):
  pp2048 2530 → 3074 tok/s, +21% over the int8 raw-moe fallback and +7%
  over the Q4_K_M build. Findings: single-stream AR decode is
  bandwidth-bound and pays the artifact's 1.7× expert bytes (-17% vs
  Q4_K_M), converging to Q4_K_M at C4-C8 where batched GEMVs are
  MAC-bound (C6 +2%); HTTP MTP loses (-14/-23%) even though native MTP
  gains (+3%), because acceptance drops (55/58% vs 63/78%) while the
  adaptive policy drafts more per cycle — sharper draft and target logits
  agree less often on near-ties. llama.cpp b11243's Q8_0 greedy is NOT
  ground truth for this artifact: it diverges from our F32 oracle at a
  0.6-logit near-tie where the GPU matches the oracle; use the oracle
  gates in QUALITY.md for Q8_0 comparisons.
- 2026-10-19 Q6_K decode GEMV pass (retained): the vendored mmvq gained
  Q6_K (vec_dot, VDR, `mul_mat_vec_moe_dispatch`, `qfn_mmq_moe_vec`
  whitelist) and `VecSupported` admits Q6_K, so `CopyDequantOne` parts
  (LM head, routed Q6_K downs, mixed shared-expert downs) keep a raw
  native view beside their Q8_0 wide-batch copy, and `Executor::Experts`
  reads the native view below the tiled threshold. The Q4_K_M artifact
  stores its head (417 MB) and 20/40 routed down stacks as Q6_K; decode
  previously read requantized Q8_0 copies of them. A/B, same prompt,
  greedy, internal bench, baseline 479175f vs this change: pp2048
  2903 ± 49 → 2922 ± 27 (prefill keeps the Q8_0/WMMA tier, unchanged);
  tg128 AR 62.16 ± 0.52 → 64.79 ± 0.44 (+4.2%); tg128 MTP 77.56 ± 0.47 →
  88.48 ± 1.36 (+14.1%); MTP d4096 85.1 → 86.5; DFlash2 open-ended
  48.7 → 88.9 (+82%) because acceptance jumped 25% → 61% — the draft is
  trained against the BF16 target and the exact Q6_K decode tracks it
  better than the requantized copy did. Greedy AR == MTP == DFlash2
  sha-identical per build (lossless property intact). Flash-Next
  Q4_K_XL sha ffa993c8 and the Ornith Q8_0 artifact sha e552e58b are
  byte-identical across the change; Flash-Next pp/tg move within noise.
  Device memory at sessions 4 rises ~4.6 GiB (native copies of the
  routed Q6_K downs and head) to 30.7 GiB. Rejected variant: per-part
  native stacks for the mixed-format attention projections (14/30 GDN
  qkv and 6/10 GQA v are Q6_K beside Q4_K parts, ~1.6% more AR) — the
  MTP draft block consumes projections across calls at different widths
  with `projections_ready`, so a width-dependent combined/per-part
  split breaks the producer-consumer layout agreement; per-part tensors
  would also need Q8_0 prefill copies. The head dominates the win
  (microbench: head shape 2.34 → 1.86 ms/token at 225 GB/s).
- 2026-10-19 chained greedy MTP (retained): ported Flash-Next's
  device-side draft chain. The catch-up and every greedy proposal run as
  one submission that feeds each step's argmax into the next step's
  embedding, so the k per-draft stream synchronizations and host round
  trips collapse into one chain download; the chain is submitted eagerly
  (its launch submission hides behind the previous cycle's GPU work —
  per-shape graph capture instead stalls a decode cycle on every new
  (catch-up, proposals) shape). Verification gained `kVerifyGreedy`:
  the batched argmax over all verified rows including the frontier rides
  the verify graph into pinned memory, which removes the per-cycle
  frontier download and the separate predictions synchronization, and
  the next anchor comes from the frontier argmax via `DeferSample`.
  Two synchronizations remain per cycle (chain, verify). Retuned
  `mtp_costs.hpp` to the chained per-width cycle costs measured at
  2048-token depth (C1/context-0 ladder 15.5/19.5/22.5/27.5/34/41/49/56.5
  ms for widths 1-8; context slope and cohort shape keep the Flash-Next
  measurements), warmed the acceptance prior with the draft's real
  profile (~90% first proposal, decaying with depth) and stepped the
  probe one width at a time so probe failures land on the deep depths
  the short-width scores do not read. Interleaved same-binary A/B at
  2a289ca, greedy completions hash-identical to the AR reference on
  every row (lossless property intact; AR and pp within noise):
  Q4_K_M MTP tg128 98.6 → 101.9 (+3.3%), tg512 105.4 → 110.4 (+4.7%);
  Q8_0 MTP tg128 82.5 → 84.4 (+2.4%), tg512 86.4 → 86.8 (+0.4%). The
  mtp_sampling policy test now asserts probe-reaches-limit and a
  score-optimal steady width under full acceptance (the depth-decaying
  priors keep unobserved deep widths pessimistic, so the steady choice
  under perfect acceptance is the cost-optimal width rather than the
  limit; consecutive full acceptances still probe to the limit first).
  Rejected variants, both bit-exact but slower: a proposal shortlist
  (gather kMtpDraftShortlist head rows once per cycle, rank chained
  steps over them) — Flash-Next's 512-row net collapses depth>=1
  acceptance on this 248k vocabulary and the candidate reduction kernel
  cannot Keep > its 1024-wide tile (each pass would grow), so every
  chained step ranks the full head; and reading the verify tokens
  straight from the device chain ring (one synchronization per cycle) —
  the width controller's acceptance evidence shifted and settled d1-heavy
  on both artifacts (Q4_K_M tg128 66.5 tok/s). Prefill keeps the
  2026-10-01 pass's WMMA routed arithmetic; see the 2026-10-19 device
  dispatch entry below for its stage profile.
- 2026-10-19 device-side routed dispatch (retained): temporary hipEvent
  stage spans around the prefill showed every layer stopping at the
  expert-counts readback — `hipEventSynchronize(counts_ready_)` plus a
  three-pass host tile-map build and upload, up to 676 ms of host wall
  per 2048-token chunk. The readback was not the wall-clock limiter (the
  host was waiting on the GPU), but it serialized every layer and it was
  the only host round trip left in the chunk. The maps now build on the
  device: one 256-thread `RoutedTileMaps` kernel compacts each map's
  (expert, row tile) entries from the device-side expert counts and pads
  the rest of a fixed per-map capacity with 0xFFFFFFFF sentinels, and the
  routed GEMM grids run at that capacity, skipping sentinels in-kernel —
  no `n_tiles` crosses to the host and `RouteHints` reads nothing. The
  count-dependent map refinements (the 128-row pair map and the
  wide-down 48/64 choice) folded into fixed token-count-and-format
  choices; their effect is inside the measured delta. Interleaved
  same-binary A/B at b0d9209, greedy completions hash-identical to the
  isolated AR reference on every row: pp2048 Q4_K_M 2951 → 3005 (+1.9%),
  Q8_0 3102 → 3233 (+4.2% — the all-native artifact's heavier per-layer
  dispatch made the readback costlier); AR tg and MTP tg unchanged within
  noise. `gufo bench --validate-prefill 8` passes (cosine 0.9995,
  scalar_winner_rank 1); qwen35moe config/gdn/mtp_sampling pass and
  attention_ops shows only its documented pre-existing 2-ulp replay
  case. The corrected prefill stage profile (hipEvent spans, one
  2048-token chunk ≈ 700 ms): GDN linear attention ~219 ms (31%), routed
  gate/up+down F16 GEMMs ~165 ms (all 256 experts' weights read per
  chunk at near the memory roofline), shared expert ~67 ms, GQA ~67 ms,
  router ~17 ms, epilogue ~30 ms, compaction and norms ~20 ms — the
  prefill is GPU-bound, and the next prefill levers are the GDN scan and
  the shared-expert F16 GEMMs (~8-16 TFLOPS effective), not the host
  path.
- 2026-10-20 long-context decode attention (retained): the per-token
  attention kernel now serves kGroup=2 query heads per block for decode
  (4 for wider verification batches), so every K and V row a block loads
  feeds both heads' unchanged arithmetic, and each wave issues the rows of
  8 consecutive keys of its 32-key tile before using any, so their loads
  overlap instead of each waiting out its own latency. The per-head block
  reductions run in one pass (lane tree, then waves in order — the same
  order BlockMax/BlockSum kept) over DPP row-xmask transports
  (`XorLane`), and the lane's eight score products keep the plain
  accumulation form the compiler has always contracted. Ported from the
  evolved Flash-Next kernel (F16 dense path only; the mask/selection
  branch was dead — every caller passes a null mask). Interleaved A/B at
  27ad6a2, native bench, greedy completions hash-identical to the
  reference build at every depth 0-131072 (AR and MTP): Q4_K_M AR tg128
  69.7 → 70.4 at d0, 50.6 → 57.5 at d32K (+14%), 28.0 → 37.1 at d131K
  (+33%); Q8_0 AR tg128 48.8 → 53.6 at d0 (+10%), 40.3 → 45.4 at d32K
  (+13%), 24.6 → 32.5 at d131K (+32%). With MTP the same kernel serves
  every verify row, and the deep rows gain the most: Q4_K_M MTP tg128
  101.5 → 101.8 at d0, 66.2 → 82.1 at d32K (+24%), 47.9 → 68.6 at d64K
  (+43%), 29.1 → 50.6 at d131K (+74%). The implied KV-read bandwidth of
  depth slope rises from ~126 GB/s to ~215 GB/s against the ~240 GB/s
  measured roofline. `qwen35moe.attention_ops` chunk cases stay
  bit-exact; the suite fails only its documented pre-existing 2-ulp
  replay case; `--validate-prefill 8` at d32K passes with the retained
  values (cosine 0.999468, scalar_winner_rank 1). AR/MTP completions
  remain hash-equal per depth and quant (the lossless property).
- 2026-10-20 prefill long-context experiments (all rejected, reverted):
  the depth slope of prefill is the dense WMMA causal sweep, and five
  structural variants of that kernel all measured neutral-to-worse on
  pp2048 at d32K/d131K: a launch-order swizzle that spreads concurrent
  query groups over the prefix (bit-exact, +0%), 32-key tiles instead of
  16 (halves the per-tile softmax and barrier rounds; +0%), split-K
  (eight key-range blocks per query group with an LSUM partials merge;
  -2%), moving the per-wave scaled-query fragments from registers to
  LDS to cut the kernel from 226 to 180 VGPRs (+7%), and the same diet
  with deferred V loads (+0%). Probes pin the limiter inside the tile
  loop rather than memory or scheduling: with every global K/V load
  removed the deep sweep still costs its time (attn 563 ms of a 1208 ms
  d32K chunk, vs 591 ms with loads), and with the loop's block barriers
  neutralized it only drops ~15%. The remaining cost tracks the
  per-tile LDS fragment loads (q and K re-read from `kv_lds`/`q_lds` per
  16-key tile for the WMMA operands): the q-in-LDS variant's +7% matches
  its added LDS traffic, and the 226-VGPR original and 180-VGPR variant
  land within noise of each other. A follow-up per-phase clock probe
  (thread 0 of every block, summed per launch) refined that read: ~54%
  of thread-0 cycles sit in the two inter-tile barriers — the block's
  eight waves are imbalanced (S runs on only four waves, the four-lane
  softmax on only waves 0-3), so each of the four per-tile barriers
  waits on the slowest wave's serial path of roughly 2 µs per 16-key
  tile, and every phase's cost reappears inside that path no matter
  what the memory system does. A wave-balanced rewrite (four dim
  quarters so all waves compute S, an eight-lane softmax, fragments in
  registers under ~128 VGPRs) shortens the chain at best ~2-3x, which
  still leaves prefill near -70% at 131K — well short of the -13.5%
  Flash-Next reports, because Flash-Next's flat prefill reads its
  indexer's top selection rather than the whole prefix. Reaching that
  profile with dense attention would need an entirely different kernel
  architecture (64-128 query rows per KV pass, register-resident
  fragments); until then the long-context prefill slope is
  architectural for this model and the depth cost of one extra
  2048-token chunk grows ~16 µs/token of prefix.
- 2026-10-20 deep MTP width controller (retained unchanged): a flat
  optimistic acceptance prior (0.9 at every depth, probe build) moved
  drafted/cycle only 2.34 → 2.45 at d131072 and cost 49.6 vs 50.6
  tok/s, so the depth-decaying prior is not what limits deep widths.
  The real constraint is the marginal cost of another verify row at
  depth: the narrow trunk runs its vector kernels per row and the
  per-token attention kernel reads the prefix per row (no cross-row KV
  amortization below the 8-row vector contract), ~21 ms per row at
  131K against ~22 ms of expected tokens — width 2-3 is the true
  optimum, and the controller's extrapolated cost ladder (whose
  depth slope and marginals are both several times off) lands there
  through compensating errors. Batching verify rows through the wide
  arithmetic would break the bitwise greedy equivalence the model's
  lossless gate requires. Deep MTP already outruns the reference
  profile: Ornith Q4_K_M MTP at 131K decodes 50.6 tok/s against
  Flash-Next's reported 39.9 mixed-text row.
- 2026-10-20 opt-in quantized KV (rejected and reverted): the full
  Flash-Next KV machinery was ported and wired end to end — f16/q8_0/
  q8_0-q4_k modes, fused-preparation quantized stores (including the
  precise-math Q4_K sub-block search), quantized load paths in both the
  per-token and WMMA attention kernels, executor allocation/rollback/
  snapshot dtype fields, and the `--kv-cache` flag through bench,
  prompt and serve — to test whether halved KV bytes flatten the
  remaining decode slope. The F16 path stayed bitwise (every greedy
  sha identical to bf74834 at d0/d4K/d131K), the quantized modes ran
  correctly end to end, and the deep quantized outputs hash-identical
  to F16 on the bench prompt (the pattern text tolerates the error),
  but every quantized configuration measured SLOWER: Q4_K_M q8_0 AR
  tg128 28.7 at d131072 per-key loads, 27.6 with the row loads batched
  like the F16 path, against 37.1 for F16 — and pp2048 at d131072
  fell 622 → 483 under the in-kernel dequant. The mechanism: the
  decode reads the whole prefix per token, so the per-element
  dequantize ALU (scalar convert+multiply per key per head) costs
  more than the halved DRAM bytes save — effective KV bandwidth
  collapsed to ~70 GB/s against the F16 path's ~215 GB/s. Flash-Next
  quantizes KV profitably because its indexer reads ~2,048 of 131,072
  keys per token; Ornith's dense attention has no such cut. An int8
  dot-product route (quantized queries, v_dot4) would shrink the ALU
  but is a separate kernel architecture. All quantized-KV code was
  reverted; the shipped build is byte-identical to bf74834 plus the
  2026-10-20 documentation.
- 2026-10-20 int8-dot decode scores over quantized KV (attempted,
  reverted): v_dot4_i32_i8 compiles on this toolchain via inline asm and
  the full quantized-KV machinery was rebuilt around it — query rows
  quantized to Q8_0 element layout per decode step, key scores as two
  4-byte integer dots per lane rescaled by the query and key block
  scales, values dequantized through the packed half2 code affine (Q8_0)
  or a batched nibble unpack (Q4_K). The F16 instantiations stayed
  bitwise (all greedy shas identical to the reference at d0/d32K/d131K),
  and the quantized speed beat F16 on paper: tg128 58.5 at d32768 and
  39.9 at d131072 against 57.4/37.1 for F16 — halved KV bytes with the
  dequant ALU finally under the bandwidth cost. It does not ship: the
  quantized outputs corrupt natural-text generation (meta-text drift on
  the sky-blue probe, pattern collapse on the bench prompt), and one
  real defect was found and fixed during triage (the query-scale fill
  wrote every block's scale to one shared slot) — after which outputs
  got worse, not better, so a second numerics defect remains and the
  +7.5% was measured on unsound scores. Shipping a lossy mode requires
  operator-level score validation against an F32 oracle (an int8
  attention case in attention_ops) before its speed is meaningful; the
  route is architecturally sound and the numbers above bound its
  upside at roughly +8% deep tg. All quantized-KV code is reverted
  again; the shipped build remains bf74834's F16 path plus
  documentation.
- 2026-10-20 int8-dot quantized KV, landed as opt-in (`--kv-cache`):
  the two numerics defects from the reverted attempt were found and
  fixed — the query-scale fill wrote every block's scale to one shared
  slot, and the Q8_0 value path consumed half2 bit patterns as float32 —
  and a new F32-oracle case in `qwen35moe.attention_ops` now validates
  the full int8-dot context against dequantized float64 attention
  (worst absolute error 5.5e-6 at n=4 and n=32, both depths checked
  before the suite's known preparation failure). With sound scores the
  quantized mode is a small win everywhere it was measured: AR tg128
  57.96 vs 57.41 at d32768 and 38.95 vs 37.85 at d131072, quantized
  completions hash-identical to F16 on the bench prompt at both depths,
  MTP near parity at d131072 (51.1 vs 50.6 in the full sweeps, same
  hash), validate-prefill cosine 0.9995 rank 1, and natural-text probes
  remain coherent (minor near-tie phrasing drift). KV bytes halve (2048
  to 1072 per token for the 2x256 K+V planes). The full depth sweeps
  scope the modes honestly: the win is deep — AR tg128 +4.8% and
  +1.7% at d131072/d65536, MTP +4.3% at d32768 — while shallow batches
  pay the per-cycle query-quantization launches (MTP d0 -15%, AR d0
  -1.8%) and deep prefill pays the WMMA in-kernel dequant (pp2048 at
  d131072 -21%), with mid-depth AR rows noisy between the paired and
  sweep measurements. Treat `--kv-cache` as the deep-context and
  memory option; F16 remains the default and the mixed-workload pick.
  The F16 default is untouched and bitwise; the modes are opt-in only.
- 2026-10-20 quantized KV on both artifacts (verification): the final
  int8-dot build's `--kv-cache q8_0-q4_k` mode had only been swept on
  the Q4_K_M artifact, and through the earlier dequant-based code at
  that; both artifacts now verify end to end on the landed kernel.
  Q4_K_M and Q8_0 each run AR and MTP at d0/d32768: within a mode AR
  and MTP completions hash identically (the lossless property), and
  the deep completions hash-identical to the F16 reference at d32768
  on both artifacts (`d856f78e...`); d0 stays coherent with the
  documented near-tie phrasing drift. Evidence:
  `.bench-tmp/multi-sess/q*-kv8q4-*.log`.
- 2026-10-20 batched verification on the single-session arithmetic
  (retained): co-batched MTP verification beyond 8 total rows still
  left the decode kernels for wide-batch views, and the chained-MTP
  width controller now drafts wide enough that ordinary serving
  cohorts cross that line — Q4_K_M multi-session MTP regressed from
  publishing C2 (2026-09-29) to failing the greedy-equivalence gate
  at C2 on this build. Three batch-path defects, all reading bytes
  the isolated decode never reads: DenseBatch routed tensors with a
  native view past the Q8_0 wide-batch copy (the Q6_K output head and
  the mixed shared-expert downs among them), GatedDenseBatch
  unconditionally on the prequantized Q8_0 view, and the MoeBatch
  Q4_K shortcut reading the down stacks' requantized Q8_0 copies
  while also pushing up to `rows x used` slot rows through a routed
  vector contract of 64. The batch paths now read what the
  single-session decode reads: DenseBatch native tensors run one
  moe_vec pass over the artifact's own bytes for the whole batch (the
  kernel quantizes each row independently, so 9-64 rows cost one
  weight read and reproduce the decode arithmetic exactly),
  GatedDenseBatch chunks the native gated kernel when native views
  exist, and MoeBatch drops the Q4_K shortcut for the same
  kDecodeRows-row chunks of MoeExperts the isolated session runs
  (routed weight traffic follows the selected slots, so the chunks
  re-read little). The E=1 zero ids buffer spans the batch instead of
  kVecBatch. F16 mode is bitwise unchanged (single-session paths are
  untouched; d0 MTP sha identical); `qwen35moe` config/mtp_sampling/
  gdn_ops pass and attention_ops keeps only its documented 2-ulp
  replay case. Greedy-equivalence gates, prepared cohorts, same
  build: Q4_K_M mixed MTP C1 85.4, C2 122.6, C4 152.3 tok/s — C2 and
  C4 pass the gate and publish where the 2026-09-29 run withheld C4
  (and this build's controller drafts past the old 8-row line at C2);
  Q8_0 mixed MTP C4 148.5 -> 155.0 (+4.4%) and repetition MTP
  publishes C1-C4 (63.5/106.2/148.4); every AR row within noise.
  Multi-session pp is already saturated: four simultaneous 2048-token
  cold prefills aggregate 2046-2089 tok/s across sessions 1/2/4
  against ~2050-2120 solo (the scheduler runs full-width prefill
  chunks round-robin while no decode or capture waits). The
  concurrent-arrival prefill divergence remains measured and unfixed:
  2 of 4 sessions prefilled alongside others decoded different greedy
  text than sequential arrival on the same build (capture-pending
  prefills take 512-token slices, and prefill routes stay
  width-sensitive); sequential or prepared cohorts are exact. One
  workload note: the synthetic repetition prompt's greedy continuation
  now emits EOS before the 128-token window on the Q4_K_M artifact
  (the AR reference round reproduces it; the sharper Q6_K-head decode
  shifted near-tie text, and a probe shows the model meta-framing the
  synthetic passage instead of repeating it), so the multi-repetition
  C1 gate cannot complete on this build; the Q8_0 artifact still
  repeats and publishes all rows. Evidence: `.bench-tmp/multi-sess/`
  (run logs, artifacts, `pp_probe.py`).
