#ifndef GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/models/qwen35moe/kernels/rocm/kv_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/core/session_mode.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen35moe/kernels/rocm/blaslt.hpp"
#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/mtp_sampling.hpp"

namespace gufo::models::qwen35moe {

namespace rocm {

struct SnapshotHeader;

/// Per-sequence state on the device: recurrent SSM state, KV caches, plus
/// the MTP draft block's private caches. A speculative forward additionally
/// keeps per-token snapshots of every recurrent buffer so the batch can be
/// cut back to its accepted prefix.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token; the next Forward starts at position 0.
  void Reset();
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool CheckCancellation(std::string* error) const;
  [[nodiscard]] bool Cancelled() const noexcept { return cancelled_; }
  [[nodiscard]] std::uint64_t MutationEpoch() const noexcept {
    return mutation_epoch_;
  }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt,
                       std::shared_ptr<qwen::vision::Encoder> encoder,
                       hipStream_t stream);
  void RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                           hipStream_t stream);
  [[nodiscard]] const qwen::vision::RopeLayout& VisionLayout() const noexcept {
    return vision_input_.layout();
  }

private:
  friend class Executor;
  Session() = default;

  struct LinearState {
    float* conv_state{nullptr};    ///< [kernel-1][channels]
    float* state{nullptr};         ///< [v_heads][d][d]
    RollbackRows conv_snapshots;   ///< [max_spec-1][kernel-1][channels]
    RollbackRows state_snapshots;  ///< [max_spec-1][v_heads][d][d]
  };
  struct AttentionState {
    const qwen::vision::DeviceRope* rope{nullptr};
    /// [max_context] rows of KvRowBytes(kv_heads*d, dtype) bytes in the
    /// executor's configured key/value formats.
    std::byte* k_cache{nullptr};
    std::byte* v_cache{nullptr};
  };
  /// Per-launch values the kernels read from device memory, so a captured
  /// graph replays at any position.
  struct Control {
    std::uint32_t position;      ///< first position of the trunk batch
    std::uint32_t mtp_position;  ///< first position of the draft batch
    std::int32_t hidden_row;     ///< kept trunk row the draft reads, or -1
  };
  struct MtpState {
    std::byte* k_cache{nullptr};
    std::byte* v_cache{nullptr};
    float* h{nullptr};              ///< [hidden] carried draft residual
    float* target_hidden{nullptr};  ///< last max_speculative trunk rows
    std::uint32_t position{0};
  };
  /// DFlash target-feature capture: the residual rows of the tap layers of
  /// every trunk batch, laid out [row][tap][hidden]. Filled by the trunk
  /// forward and read by the host after the batch is committed; rollback
  /// never rewrites it, so the accepted prefix stays valid.
  struct DFlashTapState {
    bool enabled{false};
    std::uint32_t num_taps{0};
    float* rows{nullptr};  ///< [max_batch][num_taps][hidden]
  };

  mutable bool cancelled_{false};
  std::uint64_t mutation_epoch_{0};
  const Executor* owner_{nullptr};
  qwen::vision::DeviceInput vision_input_;
  std::uint32_t max_context_{0};
  bool mtp_enabled_{false};
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  std::uint32_t spec_base_{0};    ///< position before the speculative batch
  std::uint32_t spec_tokens_{0};  ///< tokens of the pending speculative batch
  MtpState mtp_;
  DFlashTapState dflash_;
  Control* control_{nullptr};
  /// Captured decode graphs by batch shape, and the shapes that ran eagerly
  /// once (the GEMM tier's arena must be grown before capture).
  std::unordered_map<std::uint64_t, hipGraphExec_t> graphs_;
  std::unordered_set<std::uint64_t> warmed_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  std::size_t rollback_bytes_{0};
  std::function<bool()> is_cancelled_;
  std::vector<float*> rollback_allocations_;
  std::uint32_t rollback_depth_{0};
  void TrimRollback(std::uint32_t depth) noexcept;
};

/// Runs the trunk graph on the GPU for one session at a time. Buffers are
/// sized once for `max_batch` tokens; longer prompts are fed in chunks.
class Executor {
public:
  struct Options {
    std::uint32_t max_batch{1};
    /// Rows of logits (and hidden states) a Forward call may return.
    std::uint32_t max_logit_rows{1};
    /// Longest speculative batch; bounds the recurrent snapshot storage.
    std::uint32_t max_speculative{1};
    /// Trunk layer outputs (zero-based) captured for the DFlash draft.
    /// Empty disables capture; sessions still choose per-session use.
    std::vector<std::uint32_t> dflash_tap_layers;
    /// Attention KV cache storage formats (F16 by default; lossy modes
    /// shrink the per-token bytes and are not snapshot-interchangeable).
    KvDtypes kv_dtypes{};
  };

  ~Executor();
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, Options options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr, bool enable_dflash = false) const;
  [[nodiscard]] bool EnsureRollback(Session& session, std::uint32_t depth,
                                    std::string* error_msg) const;
  [[nodiscard]] std::size_t SessionBytes(
      core::SessionMode mode, std::uint32_t max_context,
      std::uint32_t rollback_depth) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;

  enum class ForwardMode { kDecode, kVerify, kVerifyGreedy, kPrefill };

  /// Appends `tokens` (at most max_batch) to the session and returns the
  /// logits of the last `n_logits` tokens in `logits` (n_logits * vocab
  /// floats, host memory). A null `logits` keeps the rows on the GPU for
  /// verification. The final residual of those tokens stays on the device
  /// for MtpForward. kVerify permits Rollback (at most max_speculative
  /// rows). kPrefill uses consistent prompt arithmetic at every chunk width.
  [[nodiscard]] bool Forward(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::uint32_t n_logits, float* logits,
                             ForwardMode mode, std::string* error_msg) const;

  struct BatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    bool speculative;
  };
  /// Packs independent decode chains for shared projections. Every chain
  /// keeps its own recurrent/KV state and the decode arithmetic (<=8 rows).
  /// Logit rows remain in batch order until SelectBatchLogits is called.
  [[nodiscard]] bool ForwardBatch(std::span<const BatchItem> items,
                                  std::string* error_msg) const;
  [[nodiscard]] bool SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                       float* logits,
                                       std::string* error_msg) const;

  /// Keeps the first `keep` (1..n) tokens of the last speculative batch and
  /// discards the rest. If `logits` is supplied, copies the kept frontier
  /// into it using the same synchronization as rollback.
  [[nodiscard]] bool Rollback(Session& session, std::uint32_t keep,
                              std::string* error_msg,
                              float* logits = nullptr) const;

  /// Runs the draft block over `tokens` (at most max_batch) at the session's
  /// MTP position. The hidden input of token i is the trunk residual of row
  /// `hidden_row + i` of the last Forward batch, or, with hidden_row < 0
  /// (single token), the draft block's own residual from the previous call.
  /// Only requested outputs are computed. Production requests a greedy
  /// token or compact candidate logits. Catch-up skips the output head
  /// when only the draft state is needed.
  struct MtpOutput {
    std::int32_t* token{nullptr};
    MtpCandidateLogits* candidates{nullptr};
    MtpTrace* trace{nullptr};  ///< final-row diagnostic; disables graph capture
  };
  struct MtpHeadItem {
    Session* session;
    MtpOutput output;
  };
  struct MtpBatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::int32_t hidden_row;
  };
  /// Runs independent short predictor chains with shared projections and
  /// private attention caches, positions and carried hidden states.
  [[nodiscard]] bool MtpForwardBatch(std::span<const MtpBatchItem> items,
                                     std::string* error_msg) const;
  /// Projects each session's carried draft hidden state with shared weights.
  [[nodiscard]] bool MtpHeads(std::span<const MtpHeadItem> items,
                              std::string* error_msg) const;
  [[nodiscard]] bool MtpForward(Session& session,
                                std::span<const std::int32_t> tokens,
                                std::int32_t hidden_row, MtpOutput output,
                                std::string* error_msg,
                                const float* hidden_source = nullptr) const;
  /// Runs the catch-up over `replay` rows and then `proposals` greedy draft
  /// steps as one device-side chain: every step embeds the previous step's
  /// argmax, so the host round trip happens once, after the last proposal.
  /// The pinned chain is readable through ChainTokens() until the next
  /// chained cycle. Graph-captured like MtpForward.
  [[nodiscard]] bool MtpDraftChain(Session& session,
                                   std::span<const std::int32_t> replay,
                                   std::int32_t hidden_row,
                                   std::uint32_t proposals,
                                   std::string* error_msg) const;
  /// The last MtpDraftChain's tokens: the anchor plus one proposal per step.
  [[nodiscard]] std::span<const std::int32_t> ChainTokens(
      std::uint32_t proposals) const noexcept;
  /// Leading kept trunk rows, for independent predictor qualification.
  [[nodiscard]] bool CopyTrunkHidden(const Session& session,
                                     std::span<float> hidden,
                                     std::string* error_msg) const;

  /// The captured DFlash tap rows [0, rows) of the last trunk batch:
  /// rows * num_taps * hidden floats. The session must have capture enabled.
  [[nodiscard]] bool CopyDFlashFeatures(const Session& session,
                                        std::uint32_t rows,
                                        std::span<float> features,
                                        std::string* error_msg) const;
  [[nodiscard]] std::uint32_t DFlashTapWidth() const noexcept {
    return static_cast<std::uint32_t>(options_.dflash_tap_layers.size());
  }

  /// Plain greedy verification keeps full logit rows on the GPU.
  [[nodiscard]] bool GreedyMtpPredictions(
      std::span<ArgmaxCandidate> predictions, std::string* error_msg) const;

  /// A kVerifyGreedy Forward ranks every verified row, the frontier
  /// included, inside its graph and parks the candidates in pinned memory;
  /// this copies them out after the Forward's synchronization.
  [[nodiscard]] bool ReadGreedyPredictions(
      std::span<ArgmaxCandidate> predictions, std::string* error_msg) const;

  /// The kVerifyGreedy body's tail: batched argmax over all verified rows
  /// into the pinned prediction buffer. Graph-safe.
  [[nodiscard]] bool GreedyArgmaxNodes(std::uint32_t n_logits,
                                       std::string* error_msg) const;

  /// A session's complete context as one host byte payload: recurrent state,
  /// KV caches up to the position, and the draft block's caches plus the
  /// `hidden_rows` most recent kept trunk rows. The payload restores into
  /// any session of this executor whose context holds the position; the
  /// pending speculative batch must be empty.
  struct SnapshotInfo {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
  };
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session,
                                            std::uint32_t hidden_rows) const;
  [[nodiscard]] bool SaveSnapshot(const Session& session,
                                  std::uint32_t hidden_rows,
                                  std::span<std::uint8_t> payload,
                                  std::string* error_msg) const;
  /// Reuse at most the rollback rows needed by the restored operation;
  /// restoration never grows scratch. Callers derive this bound from the
  /// restored policy (or the concrete verifier width in a diagnostic).
  [[nodiscard]] bool RestoreSnapshot(Session& session,
                                     std::span<const std::uint8_t> payload,
                                     SnapshotInfo* info, std::string* error_msg,
                                     std::uint32_t next_drafts = 0) const;

  /// Rewinds the draft block's own context.
  void MtpRewind(Session& session, std::uint32_t position) const noexcept {
    session.mtp_.position = position;
  }
  [[nodiscard]] std::uint32_t MtpPosition(
      const Session& session) const noexcept {
    return session.mtp_.position;
  }

  [[nodiscard]] const Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::uint32_t max_batch() const noexcept {
    return options_.max_batch;
  }
  [[nodiscard]] std::uint32_t max_speculative() const noexcept {
    return options_.max_speculative;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return model_->has_mtp(); }

private:
  Executor() = default;

  /// Visits every device region of a snapshot in payload order with
  /// (device pointer or null when sizing, payload offset, bytes, name).
  /// Returns the payload size, or 0 once a visit failed.
  template<typename Visit>
  static std::uint64_t WalkSnapshot(const SnapshotHeader& h,
                                    const Session* session, Visit&& visit);

  /// An activation batch quantized once for the decode GEMVs; `data` is
  /// null when the batch is wide enough for the tiled path.
  struct Q8Input {
    const float* x;
    const void* data;
    std::uint32_t n;
    std::uint32_t k;
  };
  bool Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                Q8Input* q, std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const Q8Input& q, float* out,
             std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const float* x, float* out,
             std::uint32_t n_tokens, std::string* error_msg) const;
  /// out = (up . x) * silu(gate . x); s_.shexp_gate is scratch.
  /// With `down` (the projection that consumes the result), a wide batch
  /// leaves only that projection's staged input form (F16 rows in
  /// s_.x_half or the tiled Q8 layout in s_.x_q8t, registered in the input
  /// cache); `out` then holds the gate projection, not the result.
  bool GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                  const float* x, float* out, std::uint32_t n_tokens,
                  const DeviceTensor* down, std::string* error_msg) const;
  /// Reuse or populate the F16 activation staging buffer.
  void PrepareHalfInput(const float* x, std::uint32_t rows,
                        std::uint32_t cols) const;
  /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
  bool DenseF16Route(const DeviceTensor& w, std::uint32_t n_tokens) const;
  void RoutedHints(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// Reads the routing of the current batch back and derives the tile
  /// hints for its expert GEMMs (tiled batches only).
  bool RouteHints(std::uint32_t n_tokens, std::string* error_msg) const;
  /// Routed gate/up projections followed by SwiGLU.
  bool GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t n_tokens, std::uint32_t n_used,
                    std::string* error_msg) const;
  bool Experts(const DeviceTensor& w, const float* x, const std::int32_t* ids,
               float* out, std::uint32_t n_rows, std::uint32_t n_used,
               std::uint32_t n_tokens, std::string* error_msg) const;
  bool LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                       const float* x, float* out, std::uint32_t n_tokens,
                       bool speculative, std::string* error_msg,
                       bool projections_ready = false,
                       bool project_output = true) const;
  /// Causal gated GQA attention over the full context; the output gate is
  /// applied inside the fused WMMA kernel or by SigmoidMul after the
  /// per-token kernel. `last_only` computes only the final query row
  /// (draft catch-up); `projections_ready` consumes the rows a producer
  /// left in the batch buffers.
  bool Attention(const DeviceLayer& l, Session::AttentionState& s,
                 const float* x, float* out, std::uint32_t n_tokens,
                 const std::uint32_t* pos, std::uint32_t start_pos,
                 std::string* error_msg, bool last_only = false,
                 bool projections_ready = false,
                 bool project_output = true) const;
  bool Moe(const DeviceLayer& l, const float* x, float* out,
           std::uint32_t n_tokens, std::string* error_msg,
           bool last_only = false) const;
  /// Runs routed experts after the router and shared expert are ready.
  bool MoeExperts(const DeviceLayer& l, const float* x, float* out,
                  std::uint32_t n_tokens, std::string* error_msg) const;
  bool MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t rows, std::string* error) const;
  /// The plain residual update: res += block_out.
  void Combine(float* res, const float* gamma, std::uint32_t n_tokens) const;
  /// Selects the greedy token or compact candidates from full MTP logits.
  /// A chained step (`chain_out`) parks its argmax on the device.
  bool MtpHead(const DeviceTensor& head_norm, const float* res, bool token,
               bool candidates, std::int32_t* chain_out,
               std::string* error_msg) const;
  /// Enqueues one trunk batch (control and token upload through logits).
  bool ForwardBody(Session& session, std::uint32_t n, std::uint32_t n_logits,
                   bool download_logits, bool speculative, bool greedy_argmax,
                   std::uint32_t start_pos, std::uint32_t first_layer,
                   std::uint32_t end_layer, std::string* error_msg) const;
  bool MtpBody(Session& session, std::uint32_t n, std::uint32_t pos, bool token,
               bool candidates, std::string* error_msg,
               const float* hidden_source, MtpTrace* trace,
               const std::int32_t* embed_tokens = nullptr,
               std::int32_t* chain_out = nullptr) const;
  /// Runs `body` eagerly, or as the session's captured graph for `key`
  /// when `graph` is set. A prefix may leave its work queued so the host
  /// can wait for disk reads while the GPU computes it.
  bool Run(Session& session, std::uint64_t key, bool graph,
           const std::function<bool()>& body, std::string* error_msg,
           bool synchronize = true) const;

  const DeviceModel* model_{nullptr};
  Options options_;
  /// Per-layer DFlash tap slot (into the captured feature rows), or -1.
  std::vector<std::int32_t> dflash_tap_slots_;
  hipStream_t stream_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::unique_ptr<BlasLt> blaslt_;

  // Scratch, sized for max_batch tokens. Names follow reference.cpp.
  struct Scratch {
    std::int32_t* tokens;
    void* x_half;   ///< activations narrowed to the weight's 16-bit type
    void* x_q8[2];  ///< Q8_1 activations of a decode batch, alternating
    void* x_q8t;    ///< tiled Q8 activations of a wide batch (W8A8 route)
    float* res;
    float* mixed;
    float* block_out;
    // linear attention
    float* qkv;
    float* z;
    float* qkvz;  ///< [t][qkv | z] from the stacked projection
    float* alpha_beta;
    float* conv_scratch;
    float* qn;
    float* kn;
    float* gdn_raw;
    float* gdn_out;
    // attention
    float* qg;  ///< [t][q|gate (; k ; v)]
    float* q;
    float* attn_gate;
    float* k;
    float* v;
    float* ctx;
    unsigned char* q8;  ///< quantized query rows (Q8_0 key plane only)
    float* attn_partials;  ///< split-key partials of a narrow batch
    // moe
    float* router;
    std::int32_t* ids;
    /// Expert-slot ids for the dense native GEMV (single expert, always 0).
    std::int32_t* zero_ids;
    std::uint32_t* expert_counts;
    // Routed WMMA route: 16-row padded bucket bounds, scatter cursors, the
    // compact row -> (token, slot) maps and the tiled Q8 gathered rows.
    std::int32_t* routed_bounds;
    std::int32_t* routed_cursors;
    std::int32_t* rows_token;
    std::int32_t* rows_slot;
    std::int32_t* routed_tiles;  ///< (expert | tile << 16) per launched tile
    float* weights;
    float* gate_e;
    float* up_e;
    float* down_e;
    float* shexp_gate;
    float* shexp_up;
    float* shexp_out;
    /// Shared-expert SwiGLU rows narrowed for the F16 down projection, so
    /// s_.x_half keeps the routed experts' token rows.
    __half* shexp_half;
    // head and hidden rows kept for the draft block
    float* logits;
    // mtp
    float* mtp_h;
    float* mtp_embd;
    float* mtp_eproj;
    float* mtp_res;
    ArgmaxCandidate* mtp_argmax;
    std::int32_t* mtp_token;
    std::uint32_t* mtp_ids;
    std::uint32_t* mtp_scratch_ids;
    float* mtp_scores;
    /// [anchor + proposals] device token ring of a chained greedy cycle:
    /// each step embeds its predecessor's argmax and writes its own.
    std::int32_t* mtp_chain;
  };
  mutable Scratch s_{};
  [[nodiscard]] Scratch RowScratch(const Scratch& base,
                                   std::uint32_t offset) const;
  void UseScratch(const Scratch& scratch) const;
  bool DenseBatch(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  bool GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                       const float* x, float* out, std::uint32_t rows,
                       std::string* error) const;
  bool QuantizeBatch(const float* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error) const;
  bool AllocateBatch(std::string* error_msg) const;
  std::vector<void*> allocations_;
  std::int32_t* zero_ids_{nullptr};
  /// Allocated only when concurrent decoding is first requested.
  mutable float* batch_logits_{nullptr};
  mutable Session::Control* batch_controls_{nullptr};
  mutable MtpCandidateLogits* batch_candidates_host_{nullptr};
  // Mapped descriptors, one slice per layer: GPU reads cannot race the host
  // preparing the next layer. ForwardBatch drains before reusing this table.
  mutable GdnBatchItem* batch_gdn_host_{nullptr};
  mutable GdnBatchItem* batch_gdn_{nullptr};
  mutable std::uint32_t batch_rows_{0};
  mutable const float* selected_logits_{nullptr};
  [[nodiscard]] const float* VerificationLogits() const {
    return selected_logits_ != nullptr ? selected_logits_ : s_.logits;
  }
  hipEvent_t counts_ready_{nullptr};
  // Pinned host staging the launched (or captured) work reads and writes.
  Session::Control* control_host_{nullptr};
  std::int32_t* tokens_host_{nullptr};
  std::uint32_t* counts_host_{nullptr};
  std::int32_t* tiles_host_{nullptr};         ///< routed tile map staging
  mutable std::uint32_t routed_max_rows_{0};  ///< 0 = no readback yet
  mutable std::uint32_t routed_n_tiles_{0};   ///< down projection tiles
  mutable std::uint32_t routed_64_tiles_{0};  ///< appended 64-token tiles
  mutable std::uint32_t routed_pair_offset_{0};
  mutable std::uint32_t routed_pair_tiles_{0};
  mutable std::uint32_t routed_pair_rows_{64};
  mutable std::uint32_t routed_tile_rows_{48};  ///< token rows per tile
  mutable int routed_tile_cols_{0};
  float* logits_host_{nullptr};
  std::int32_t* mtp_token_host_{nullptr};
  MtpCandidateLogits* mtp_candidates_host_{nullptr};
  /// Pinned anchor + draft proposals written by the captured MtpDraftChain
  /// download; readable without a synchronization after the chain's
  /// hipStreamSynchronize.
  std::int32_t* mtp_chain_host_{nullptr};
  /// Pinned per-row argmax candidates written by the captured kVerifyGreedy
  /// body; ReadGreedyPredictions() copies them after the Forward syncs.
  ArgmaxCandidate* mtp_predictions_host_{nullptr};
  // What s_.x_q8t / s_.x_half currently hold (input pointer, rows, cols,
  // and the half type), so a projection over the same rows skips its
  // activation pass. Cleared whenever the source buffer is rewritten.
  mutable const float* q8t_src_{nullptr};
  mutable std::uint32_t q8t_rows_{0};
  mutable std::size_t q8t_cols_{0};
  mutable const float* half_src_{nullptr};
  mutable std::uint32_t half_rows_{0};
  mutable std::size_t half_cols_{0};
  mutable bool half_bf16_{false};
  /// Set by GatedDense when s_.shexp_half holds the SwiGLU rows its F16
  /// down projection reads.
  mutable bool shexp_half_ready_{false};
  mutable unsigned q8_slot_{0};
};

}  // namespace rocm
}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_KERNELS_ROCM_EXECUTOR_HPP_
