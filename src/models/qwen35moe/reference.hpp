#ifndef GUFO_MODELS_QWEN35MOE_REFERENCE_HPP_
#define GUFO_MODELS_QWEN35MOE_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace gufo::models::qwen35moe {

/// Single-token, float32, scalar reference of the qwen35moe graph. It exists
/// to pin the semantics of every operator the GPU runtime must reproduce:
/// plain pre-norm residual streams, Gated DeltaNet, gated GQA attention,
/// softmax top-k MoE with the shared expert, and the in-file MTP draft block.
/// Speed is irrelevant here.
class ReferenceModel {
public:
  /// Float32 formulas versus an independent emulation of decode storage:
  /// Q8 activation blocks and F16 KV entries.
  enum class Storage { kFloat32, kDecode };
  ReferenceModel(const ModelWeights& weights, std::uint32_t max_context,
                 Storage storage = Storage::kFloat32);

  /// Runs one token at the next position; `logits` (vocab floats) receives
  /// the output distribution when non-empty. `hidden_out` (hidden floats)
  /// receives the final residual, the MTP block's hidden input.
  [[nodiscard]] bool Step(std::int32_t token, std::span<float> logits,
                          std::span<float> hidden_out = {},
                          std::string* error_msg = nullptr);

  /// Independent scalar predictor. An empty hidden input uses the previous
  /// predictor residual; otherwise use the supplied pre-head trunk stream.
  /// The shifted token always uses the text embedding table, including
  /// image-pad IDs. Visual information is carried by hidden and mRoPE.
  /// `stage_inputs` optionally supplies observed fusion/attention outputs as
  /// inputs to the following stages, isolating arithmetic from accumulated
  /// quantization drift. Attention caches remain independently computed.
  [[nodiscard]] bool MtpStep(const MtpWeights& mtp, std::int32_t token,
                             std::span<const float> hidden,
                             std::span<float> logits, MtpTrace* trace = nullptr,
                             const qwen::vision::RopeLayout* rope = nullptr,
                             std::string* error_msg = nullptr,
                             const MtpTrace* stage_inputs = nullptr);

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  void MatVec(const TensorRef& weight, std::uint64_t expert,
              std::span<const float> x, std::span<float> out,
              bool narrow_weights = false);
  struct LinearState {
    std::vector<float> conv;   ///< [kernel-1][channels], oldest first
    std::vector<float> state;  ///< [v_heads][head_dim(v)][head_dim(k)]
  };
  struct AttentionState {
    std::vector<float> k;  ///< [pos][kv_heads][head_dim], rotated
    std::vector<float> v;  ///< [pos][kv_heads][head_dim]
  };

  static void Norm(std::span<const float> res, const float* gamma,
                   std::span<float> out);
  static void Add(std::span<float> res, std::span<const float> block_out);
  void LinearAttention(const LayerWeights& l, LinearState& s,
                       std::span<const float> x, std::span<float> out);
  void Attention(const LayerWeights& l, AttentionState& s,
                 std::span<const float> x, std::uint32_t pos,
                 std::span<float> out,
                 const qwen::vision::RopeLayout* rope = nullptr);
  void Moe(const LayerWeights& l, std::span<const float> x,
           std::span<float> out);

  const ModelWeights& w_;
  const Config& c_;
  Storage storage_;
  std::uint32_t max_context_;
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  AttentionState mtp_attention_;
  std::vector<float> mtp_hidden_;
  std::uint32_t mtp_position_{0};
};

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_REFERENCE_HPP_
