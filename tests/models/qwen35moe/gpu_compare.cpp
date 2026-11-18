// Compares the ROCm runtime's per-position logits against the float32
// scalar reference for the same prompt. Each prefix length runs in its own
// session, so position 0 (no attention history) isolates the GDN/MoE path
// from the attention path.
//
//   gpu_compare --model MODEL.gguf --tokens 1,2,3

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen35moe/engine.hpp"
#include "src/models/qwen35moe/reference.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace q = gufo::models::qwen35moe;

namespace {

std::vector<std::int32_t> ParseTokens(const std::string& text) {
  std::vector<std::int32_t> out;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find(',', start);
    out.push_back(std::atoi(text.substr(start, end - start).c_str()));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out;
}

void PrintTop(const char* label, std::span<const float> logits,
              const gufo::tokenization::QwenTokenizer& tokenizer,
              std::uint32_t top) {
  std::vector<std::uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + top, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  std::printf("  %s top%u:", label, top);
  for (std::uint32_t i = 0; i < top; ++i) {
    const auto piece = tokenizer.DecodeToken(order[i]);
    std::printf(" %u(%.3f '%.*s')", order[i], logits[order[i]],
                static_cast<int>(piece.size()), piece.data());
  }
  std::printf("\n");
}

double Compare(std::span<const float> ref, std::span<const float> gpu,
               double* rmse_out, double* max_out) {
  double dot = 0, ref_norm = 0, gpu_norm = 0, sq = 0, max_diff = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    dot += static_cast<double>(ref[i]) * gpu[i];
    ref_norm += static_cast<double>(ref[i]) * ref[i];
    gpu_norm += static_cast<double>(gpu[i]) * gpu[i];
    sq += static_cast<double>(gpu[i] - ref[i]) * (gpu[i] - ref[i]);
    max_diff =
        std::max(max_diff, std::abs(static_cast<double>(gpu[i] - ref[i])));
  }
  *rmse_out = std::sqrt(sq / static_cast<double>(ref.size()));
  *max_out = max_diff;
  return dot / (std::sqrt(ref_norm) * std::sqrt(gpu_norm) + 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::string tokens_arg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--model") {
      model_path = next();
    } else if (arg == "--tokens") {
      tokens_arg = next();
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_path.empty() || tokens_arg.empty()) {
    std::fprintf(stderr, "--model and --tokens are required\n");
    return 2;
  }
  const auto tokens = ParseTokens(tokens_arg);

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  auto weights = q::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }

  auto model = q::Model::Load(model_path, {.max_context = 256}, &error);
  if (!model) {
    std::fprintf(stderr, "gpu load failed: %s\n", error.c_str());
    return 1;
  }

  double worst_cos = 2.0;
  std::size_t worst_pos = 0;
  for (std::size_t prefix = 1; prefix <= tokens.size(); ++prefix) {
    q::ReferenceModel ref(*weights, static_cast<std::uint32_t>(prefix),
                          q::ReferenceModel::Storage::kDecode);
    std::vector<float> ref_row(weights->config.vocab_size);
    for (std::size_t i = 0; i < prefix; ++i) {
      if (!ref.Step(tokens[i], ref_row, {}, &error)) {
        std::fprintf(stderr, "reference step failed: %s\n", error.c_str());
        return 1;
      }
    }
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, 256, &error);
    if (!session) {
      std::fprintf(stderr, "gpu session failed: %s\n", error.c_str());
      return 1;
    }
    const std::vector<std::int32_t> input(
        tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
    if (!session->Sync(input, &error)) {
      std::fprintf(stderr, "gpu sync failed: %s\n", error.c_str());
      return 1;
    }
    const auto gpu_row = session->Logits();
    double rmse = 0, max_diff = 0;
    const double cosine = Compare(ref_row, gpu_row, &rmse, &max_diff);
    const auto ref_argmax = static_cast<std::size_t>(
        std::max_element(ref_row.begin(), ref_row.end()) - ref_row.begin());
    const auto gpu_argmax = static_cast<std::size_t>(
        std::max_element(gpu_row.begin(), gpu_row.end()) - gpu_row.begin());
    std::printf(
        "pos %zu: cosine=%+.6f rmse=%.6f max=%.4f argmax %u vs %u (%s)\n",
        prefix - 1, cosine, rmse, max_diff,
        static_cast<std::uint32_t>(ref_argmax),
        static_cast<std::uint32_t>(gpu_argmax),
        ref_argmax == gpu_argmax ? "agree" : "DIVERGE");
    if (prefix == tokens.size()) {
      PrintTop("ref", ref_row, *tokenizer, 5);
      PrintTop("gpu", gpu_row, *tokenizer, 5);
    }
    if (cosine < worst_cos) {
      worst_cos = cosine;
      worst_pos = prefix - 1;
    }
  }
  std::printf("worst cosine %+.6f at pos %zu\n", worst_cos, worst_pos);
  return worst_cos < 0.999 ? 1 : 0;
}
