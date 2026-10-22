#include "src/models/qwen35moe/config.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace q35 = gufo::models::qwen35moe;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

using Value =
    std::variant<std::uint32_t, float, std::string, std::vector<std::uint64_t>,
                 std::vector<std::int64_t>, std::vector<double>>;
using Metadata = std::map<std::string, Value>;

// Minimal metadata-only GGUF: no model or GPU is needed to test the loader.
std::optional<q35::Config> Parse(const Metadata& fields, bool trunk = true,
                                 std::string* error_out = nullptr) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            using E = typename T::value_type;
            pod(std::uint32_t{std::is_same_v<E, std::uint64_t>  ? 10U
                              : std::is_same_v<E, std::int64_t> ? 11U
                                                                : 12U});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  assert(reader && error.empty());
  const auto result = q35::Config::FromGguf(*reader, trunk, &error);
  assert(result.has_value() == error.empty());
  if (error_out != nullptr)
    *error_out = error;
  return result;
}

Metadata ValidMetadata() {
  Metadata fields{{"general.architecture", std::string{"qwen35moe"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 41},
           {"nextn_predict_layers", 1},
           {"embedding_length", 2048},
           {"context_length", 262144},
           {"full_attention_interval", 4},
           {"attention.head_count", 16},
           {"attention.head_count_kv", 2},
           {"attention.key_length", 256},
           {"attention.value_length", 256},
           {"rope.dimension_count", 64},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 16},
           {"ssm.time_step_rank", 32},
           {"ssm.inner_size", 4096},
           {"expert_count", 256},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 512},
           {"expert_shared_feed_forward_length", 512},
       }) {
    fields[std::string{"qwen35moe."} + key] = value;
  }
  fields["qwen35moe.attention.layer_norm_rms_epsilon"] = 1e-6F;
  fields["qwen35moe.rope.freq_base"] = 1e7F;
  fields["qwen35moe.rope.dimension_sections"] =
      std::vector<std::uint64_t>{11, 11, 10, 0};
  return fields;
}

}  // namespace

int main() {
  try {
    // The pinned Ornith geometry parses with derived dimensions.
    std::string parse_error;
    const auto config = Parse(ValidMetadata(), true, &parse_error);
    Require(config.has_value(),
            ("valid qwen35moe metadata rejected: " + parse_error).c_str());
    Require(config->num_layers == 40, "trunk layer count wrong");
    Require(config->num_layers_all == 41, "block count wrong");
    Require(config->nextn_layers == 1, "nextn layers wrong");
    Require(config->hidden_size == 2048, "hidden size wrong");
    Require(config->SsmConvChannels() == 8192, "ssm conv channels wrong");
    Require(config->SsmValueDim() == 4096, "ssm value dim wrong");
    Require(config->AttentionQDim() == 4096, "attention q dim wrong");
    Require(config->AttentionKvDim() == 512, "attention kv dim wrong");
    Require(config->IsLinearLayer(0) && !config->IsLinearLayer(3) &&
                config->IsLinearLayer(39) == false,
            "attention interval layout wrong");
    Require(config->vocab_size == 0,
            "vocabulary comes from the embedding tensor, not metadata");

    // Every pinned dimension is load-bearing: a mismatch must be rejected.
    for (const auto& [key, wrong] :
         std::initializer_list<std::pair<const char*, std::uint32_t>>{
             {"block_count", 40},
             {"embedding_length", 2560},
             {"attention.head_count", 24},
             {"attention.head_count_kv", 4},
             {"attention.key_length", 128},
             {"ssm.time_step_rank", 48},
             {"ssm.group_count", 8},
             {"expert_count", 512},
             {"expert_used_count", 10},
             {"expert_feed_forward_length", 640},
             {"full_attention_interval", 8},
         }) {
      auto fields = ValidMetadata();
      fields[std::string{"qwen35moe."} + key] = wrong;
      auto broken = Parse(fields);
      Require(!broken.has_value(), "unsupported geometry accepted");
    }
    {
      auto fields = ValidMetadata();
      fields["qwen35moe.rope.dimension_sections"] =
          std::vector<std::uint64_t>{11, 11, 11, 0};
      Require(!Parse(fields).has_value(), "wrong mRoPE sections accepted");
    }
    {
      auto fields = ValidMetadata();
      fields["qwen35moe.ssm.inner_size"] = std::uint32_t{6144};
      Require(!Parse(fields).has_value(),
              "inner size inconsistent with value dim accepted");
    }

    // Other architectures are not this runtime's input.
    {
      auto fields = ValidMetadata();
      fields["general.architecture"] = std::string{"qwen4exp"};
      std::string error;
      Require(!Parse(fields, true, &error).has_value() &&
                  error.find("qwen35moe") != std::string::npos,
              "foreign architecture error is unclear");
    }

    // A draft-only header is not a trunk.
    {
      auto fields = ValidMetadata();
      fields["qwen35moe.block_count"] = std::uint32_t{1};
      Require(!Parse(fields, true).has_value(),
              "trunk-only header without a trunk accepted");
    }
    {
      auto fields = ValidMetadata();
      fields["qwen35moe.nextn_predict_layers"] = std::uint32_t{41};
      Require(!Parse(fields).has_value(),
              "nextn at or past block count accepted");
    }
  } catch (const std::exception& e) {
    std::cerr << "qwen35moe config test failed: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen35moe config tests passed\n";
  return 0;
}
