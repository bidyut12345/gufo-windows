// SYNCPROBE-2026-09-29A
#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/vision/rope.hpp"

#include "src/models/qwen35moe/kernels/rocm/kv_layout.hpp"

/// Model-private HIP launchers for everything outside the quantized GEMM
/// tier. Activations are row-major float32 [tokens][dim] unless noted; every
/// launch is asynchronous on `stream`. Weights referenced here are device
/// pointers in their GGUF encoding.
namespace gufo::models::qwen35moe::rocm {

/// Stable per-prefix rollback addresses. Growing the depth does not move
/// existing buffers or invalidate graphs captured for a smaller batch.
struct RollbackRows {
  float* rows[7]{};
};

// Recurrent rollback keeps the state after the first token, then exact
// FP32 keys/decays/betas/errors for later tokens. No inverse or quantization.
constexpr std::size_t GdnRollbackRowFloats(std::uint32_t row,
                                           std::uint32_t k_heads,
                                           std::uint32_t v_heads,
                                           std::uint32_t dim) {
  return row == 0 ? std::size_t{v_heads} * dim * dim
                  : std::size_t{k_heads} * dim + v_heads * (dim + 2);
}
void RestoreGdnState(float* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t k_heads, std::uint32_t v_heads,
                     hipStream_t stream);

/// GGUF type ids the runtime accepts for the small-matrix and lookup paths.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kIQ4_NL = 20,
  kQ4_K = 12,
  kQ5_K = 13,
  kIQ3_S = 21,
  kBF16 = 30,
};

/// res[t][s][hidden] = table[tokens[t]] for every stream s (Q8_0 rows).
/// res[i] += block_out[i], the plain residual update of a standard
/// pre-norm block.
void CombineAdd(const float* block_out, float* res, std::size_t count,
                hipStream_t stream);

void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream);

/// out[t][d] = rmsnorm(x[t][d]) * gamma[d]; `groups` independent norms of
/// `dim/groups` elements each share the gamma row. gamma may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// x[i] = silu(x[i] * scale), in place over `count` floats.
void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream);
/// gate[i] = silu(gate[i]) * up[i], in place in `gate`.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);
/// out[i] = silu(gate[i]) * up[i] as F16 (`count` elements).
void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream);
/// silu(gate) * up over n_rows rows of k elements, written only into the
/// W8A8 tiled Q8 layout `out_q8`; false (nothing launched) unless k % 32 == 0.
bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream);
/// x[i] *= sigmoid(g[i]).
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);

/// out[t][m] = sum_k W[m][k] * x[t][k] for F32/BF16/F16 weights; meant for
/// the narrow projections (routers, alpha/beta, indexer, inject) that the
/// quantized tier does not cover.
/// Converts `count` floats to BF16 (or F16) for a 16-bit hipBLAS GEMM.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// W8A8 route for wide batches over Q8_0 weights: activations quantized per
/// 32-wide block into the tiled fragment layout (`Q8TiledBytes(batch, k)`
/// bytes, padded to the GEMM's 128-token macro tile), then an int8 WMMA
/// GEMM. out is [batch][m]. W8A8Gemm returns false, launching nothing, when
/// k is not a multiple of 32.
std::size_t Q8TiledBytes(std::size_t batch, std::size_t k);
void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream);
bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream);
/// Native wave64 launch, selected by W8A8Gemm for wide output projections.
void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream);

/// F16 activation rows [batch][k], Q8_0 weights dequantized to F16 in LDS,
/// F32 accumulation. out is [batch][m]. Unsupported shapes launch nothing.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream);

/// SSM Q8_0 projection fused with its four-tap convolution. Supports
/// [m=16384,k=2560,channels=10240] and at least 1024 tokens. qkvz retains
/// every Z row plus the QKV rows required for tile boundaries and rolling
/// history; other QKV rows are not written. convolved is [tokens][channels].
/// history is read only; pass convolved=true to GatedDeltaNet to consume
/// this result and update history. Speculative snapshots need all raw QKV
/// rows and must use DenseF16Gemm instead. Unsupported shapes launch nothing.
bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Builds up to three routed-GEMM launch maps from the device-side expert
/// counts so no layer waits for a counts readback. Map m (token rows
/// `rows.m`, 0 = unused) lands at `tiles` offset `caps[0] (+ caps[1])`,
/// holds `caps.m` entries and packs live (expert | row tile << 16) tiles
/// compactly from the front with 0xFFFFFFFF sentinels behind, which the
/// capacity-sized GEMM grids skip. One block, 256 threads.
void RoutedTileMaps(const std::uint32_t* counts, std::uint32_t experts,
                    uint3 rows, uint3 caps, std::int32_t* tiles,
                    hipStream_t stream);
/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, 48-row token tile) as expert | tile << 16,
/// so empty tiles cost nothing. Exactly one of `out` (F32) and `out_half`
/// (F16) receives the result; a non-null `swiglu_gate` (the gate
/// projection's F32 output over the same rows) turns it into
/// silu(gate) * result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 for Q5_1/Q8_0.
/// Q4_K and Q5_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other types
/// and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Paired Q4_K/Q5_K gate/up GEMM with SwiGLU and F16 output. Uses the same
/// routing layout as RoutedF16Gemm, with 64 or 128 token rows per tile. The two
/// projections share one launch and never materialize the gate output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

/// F32/BF16/F16 projection in groups of up to eight token rows. Each token
/// keeps the same accumulation order at every batch width.
void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);

/// Gated DeltaNet over a chunk of tokens for one layer. Runs the causal
/// conv (with rolling `conv_state`, [kernel-1][channels]) and the recurrence
/// on `state` ([v_heads][d][d]) sequentially over tokens, parallel over heads
/// and value dims. `alpha_beta` holds both projections per token:
/// [t][alpha(v_heads) | beta(v_heads)]. Outputs the normalized,
/// sigmoid-gated attention rows [tokens][v_heads*d] ready for the output
/// projection.
/// `state_snapshots` ([n_tokens-1][v_heads*d*d]) and `conv_snapshots`
/// ([n_tokens-1][(kernel-1)*channels]), when populated, receive the state after
/// each proper prefix. The live state already holds the full batch.
/// Scratch: `conv_scratch` ((n_tokens + kernel) * channels), `qn`/`kn`
/// (n_tokens * k_heads * d), `raw` (n_tokens * v_heads * d).
/// `qkv` rows are `qkv_stride` floats apart and `z` rows `z_stride`, so a
/// stacked [qkv|z] projection feeds both without unpacking. A non-null
/// `out_q8` receives the rows quantized into the W8A8 tiled layout
/// (K = v_heads * d) instead of `out`. With null `out_q8`, a non-null
/// `out_half` receives F16 rows instead of `out`. `convolved` consumes
/// convolution rows already in conv_scratch; it requires null speculative
/// snapshots.
void GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half = nullptr);

/// Private rows for one request in a decode batch. Scratch regions and all
/// recurrent/history/rollback buffers must be disjoint between requests.
struct GdnBatchItem {
  const float* qkv;
  const float* z;
  const float* alpha_beta;
  float* conv_state;
  float* conv_scratch;
  float* qn;
  float* kn;
  float* raw;
  float* state;
  float* out;
  RollbackRows state_snapshots;
  RollbackRows conv_snapshots;
  std::uint32_t n_tokens;
};

/// Runs the decode arithmetic for 1–8 independent requests of 1–8 rows,
/// 128-wide heads and four convolution taps. `items` is device-accessible;
/// inactive requests are untouched. The caller zeros their output rows.
bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream);

/// Splits the interleaved [q|gate] projection (rows `qg_stride` apart) into
/// q [t][heads][d] and gate [t][heads*d]. With non-null `k`, the row
/// continues with k and v (`kv_width` each), copied out contiguously.
void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream);

/// Unpacks a stacked Q/gate/K/V projection, normalizes and rotates Q/K,
/// and writes the caches in `kv_dtypes`' formats. Returns false for
/// unsupported geometries.
bool PrepareAttention(const float* packed, std::uint32_t stride,
                      const float* q_gamma, const float* k_gamma, float* q,
                      float* gate, void* k_cache, void* v_cache,
                      std::uint32_t n_tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t rotary_dim, const std::uint32_t* start_pos,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope = nullptr,
                      bool prefill = false, KvDtypes kv_dtypes = {});

/// Quantizes rotated query rows to the Q8_0 element layout for the int8-dot
/// attention scores (one row = d code bytes + d/32 F16 scales).
void QuantizeAttentionQ(const float* q, unsigned char* q8,
                        std::uint32_t n_tokens, std::uint32_t heads,
                        std::uint32_t d, hipStream_t stream);

/// NEOX partial rotary on x [t][heads][d] at positions start_pos + t.
/// Positions are read from device memory (`start_pos` points at the
/// session's control block) so a captured decode graph replays at any
/// position.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope = nullptr);

/// Stores f32 rows into the f16 cache at positions start_pos + t:
/// cache[(start_pos + t)][row_dim].
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream);

/// Per-token attention (decode and narrow batches). With `partials`
/// (n_tokens * heads * splits * (d + 2) floats) the key tiles are split
/// across `splits` blocks per row and merged in a second launch.
void Attention(const float* q, const unsigned char* q8, const void* k_cache,
               const void* v_cache, const std::uint32_t* mask,
               std::uint32_t mask_words, float* out, float* partials,
               std::uint32_t splits, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
               hipStream_t stream, KvDtypes kv_dtypes = {});

/// Fused causal attention on the WMMA cores for wide batches: scores, online
/// softmax, PV and the sigmoid output gate in one launch. `mask` must be null
/// (dense windows only). Returns false, launching nothing, when the geometry
/// is not 16 or 24 query heads of 256 over two KV heads. `last_only` computes
/// only the final dense query tile, retaining its key sweep and leaving
/// earlier output rows untouched.
bool WmmaCausalAttention(const float* q, const float* gate,
                         const void* k_cache, const void* v_cache,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream,
                         bool last_only = false, KvDtypes kv_dtypes = {});

/// counts[e] = number of (token, slot) pairs routed to expert e.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), top-k
/// selection, renormalized weights. ids [t][k] int32, weights [t][k] f32.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);
/// Same contract, four adjacent lanes per thread (`dim % 4 == 0`, else the
/// scalar kernel runs).
void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream);
/// MoeEpilogueVec4 over F16 expert rows.
void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream);

/// dst[t] = *row < 0 ? alt[t] : base[*row + t] (rows of `width` floats).
void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream);

/// Add the projected embedding once to each projected hidden branch.
void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream);

inline constexpr std::uint32_t kArgmaxParts = 64;
struct ArgmaxCandidate {
  float value;
  std::int32_t index;
};
/// Full-vocabulary argmax into out[t], with the lowest index winning ties.
/// Matches std::max_element, including NaNs. Scratch holds
/// n_tokens * kArgmaxParts candidates.
void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream);

/// Gather selected IDs and their original logits from independent rows.
/// The caller checks finite values only for the verification prefix it visits.
void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream);

/// Number of IDs in each of the two selection buffers.
std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab);

/// Exact MTP top-64 selection, descending score with lowest-token-ID ties.
/// Nonfinite scores become -infinity. Both ID buffers have room for
/// MtpCandidateWorkspaceSize(vocab); scores receives min(vocab, 64) values.
/// Scores may reuse the ID workspace starting at offset 64.
/// All candidate operations are graph-safe.
void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream);

/// Advances the draft control block after one chained draft stage: the draft
/// position moves by `count` tokens and `hidden_row` falls back to the
/// carried residual. Graph-safe.
void MtpAdvanceControl(std::uint32_t* mtp_position, std::int32_t* hidden_row,
                       std::uint32_t count, hipStream_t stream);

}  // namespace gufo::models::qwen35moe::rocm

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_KERNELS_HPP_
