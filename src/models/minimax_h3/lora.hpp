#ifndef GUFO_MODELS_MINIMAX_H3_LORA_HPP_
#define GUFO_MODELS_MINIMAX_H3_LORA_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <gufo/ext/experimental/lora_adapter.hpp>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::minimax_h3 {

using LoraDtype = lora::Dtype;
using LoraTensor = lora::Tensor;
using LoraPatch = lora::Patch;
using LoraTerm = lora::Term;
using LoraTargetMap = std::map<std::string, std::vector<LoraTerm>, std::less<>>;

struct LokrPatch {
  std::string target;
  std::vector<float> w1;
  std::vector<float> w2;
  int w1_rows{0};
  int w1_cols{0};
  int w2_rows{0};
  int w2_cols{0};
  float scale{1.0F};
};

struct LokrTerm {
  const LokrPatch* patch{nullptr};
  float strength{1.0F};
};

using LokrTargetMap = std::map<std::string, std::vector<LokrTerm>, std::less<>>;

inline std::vector<float> LoraFloats(const LoraTensor& tensor,
                                     std::size_t first, std::size_t count) {
  return lora::Floats(tensor, first, count);
}

/// Validated MiniMax H3 DiT adapter file (LoRA, DoRA, diffs).
class LoraAdapter : public lora::Adapter {
public:
  /// Base tensor shape by MiniMax H3 inventory name, or empty when absent.
  using BaseShape = std::function<std::vector<int>(std::string_view)>;
  LoraAdapter(const std::filesystem::path& path, const BaseShape& base);
};

/// Loads either a standard LoRA/DoRA/diff adapter or a LoKr adapter file.
struct LoadedAdapter {
  std::shared_ptr<const LoraAdapter> lora;
  std::vector<LokrPatch> lokr_patches;
};

[[nodiscard]] LoadedAdapter LoadMinimaxAdapter(
    const std::filesystem::path& path, const LoraAdapter::BaseShape& base);

}  // namespace gufo::minimax_h3
#endif
