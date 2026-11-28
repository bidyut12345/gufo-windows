#include "src/models/minimax_h3/lora.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr std::string_view kPrefixes[] = {
    "diffusion_model.", "base_model.model.", "transformer."};

constexpr std::string_view kDitBlockModules[] = {
    "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2",
    "adaln_proj.linear"};

constexpr std::string_view kRefinerModules[] = {
    "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"};

constexpr std::string_view kGlobalModules[] = {
    "audio_patch_proj", "video_patch_proj", "final_layer.audio_out",
    "final_layer.video_out", "final_layer.adaln_proj.linear"};

lora::Schema Schema(const LoraAdapter::BaseShape& base) {
  lora::Schema schema;
  schema.base = base;
  schema.prefixes.assign(std::begin(kPrefixes), std::end(kPrefixes));
  const auto whole = [](const std::string& module) {
    return std::vector<lora::Slice>{{module + ".weight"}};
  };
  const auto register_alias = [&](const std::string& alias,
                                  const std::string& preferred_target) {
    if (!base(preferred_target + ".weight").empty()) {
      schema.modules.emplace(alias, whole(preferred_target));
    } else if (!base(alias + ".weight").empty()) {
      schema.modules.emplace(alias, whole(alias));
    }
  };
  for (int b = 0; !base("blocks." + std::to_string(b) + ".attn.qkv_proj.weight")
                       .empty();
       ++b) {
    const std::string block = "blocks." + std::to_string(b) + ".";
    for (auto module : kDitBlockModules) {
      schema.modules.emplace(block + std::string(module),
                             whole(block + std::string(module)));
    }
    register_alias(block + "adaln_proj", block + "adaln_proj.linear");
  }
  for (int b = 0; !base("token_refiner.blocks." + std::to_string(b) +
                        ".attn.qkv_proj.weight")
                       .empty();
       ++b) {
    const std::string block = "token_refiner.blocks." + std::to_string(b) + ".";
    for (auto module : kRefinerModules) {
      schema.modules.emplace(block + std::string(module),
                             whole(block + std::string(module)));
    }
  }
  for (auto module : kGlobalModules) {
    if (!base(std::string(module) + ".weight").empty()) {
      schema.modules.emplace(std::string(module), whole(std::string(module)));
    }
  }
  register_alias("final_layer.adaln_proj", "final_layer.adaln_proj.linear");
  schema.unknown = [](std::string_view) -> std::string {
    return "not a MiniMax H3 DiT module";
  };
  return schema;
}

std::string_view StripKnownPrefix(std::string_view name) {
  for (std::string_view prefix : kPrefixes) {
    if (name.starts_with(prefix)) {
      name.remove_prefix(prefix.size());
      break;
    }
  }
  return name;
}

std::vector<float> ReadSafetensorsFloats(const std::byte* base_ptr,
                                         std::string_view dtype,
                                         std::size_t count) {
  std::vector<float> out(count);
  if (dtype == "F32") {
    std::memcpy(out.data(), base_ptr, count * sizeof(float));
  } else if (dtype == "F16") {
    const auto* src = reinterpret_cast<const std::uint16_t*>(base_ptr);
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint32_t h = src[i];
      const std::uint32_t sign = (h & 0x8000U) << 16U;
      const std::uint32_t exp = (h >> 10U) & 0x1FU;
      const std::uint32_t mant = h & 0x3FFU;
      std::uint32_t bits = 0;
      if (exp == 0) {
        if (mant != 0) {
          float v = std::ldexp(static_cast<float>(mant), -24);
          if (sign) v = -v;
          out[i] = v;
          continue;
        }
        bits = sign;
      } else if (exp == 31) {
        bits = sign | 0x7F800000U | (mant << 13U);
      } else {
        bits = sign | ((exp + 112U) << 23U) | (mant << 13U);
      }
      out[i] = std::bit_cast<float>(bits);
    }
  } else if (dtype == "BF16") {
    const auto* src = reinterpret_cast<const std::uint16_t*>(base_ptr);
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16U;
      out[i] = std::bit_cast<float>(bits);
    }
  } else {
    throw std::runtime_error("unsupported LoKr tensor dtype: " +
                             std::string(dtype));
  }
  return out;
}

struct LokrEntry {
  std::string_view dtype;
  std::vector<int> shape;
  std::size_t begin{0};
  std::size_t end{0};
};

std::vector<LokrPatch> ParseLokrFile(const std::filesystem::path& path,
                                     const LoraAdapter::BaseShape& base,
                                     const std::byte* mapped, std::size_t size,
                                     std::size_t data_offset,
                                     const json::Value& header) {
  struct Group {
    const LokrEntry* w1{nullptr};
    const LokrEntry* w2{nullptr};
    const LokrEntry* alpha{nullptr};
  };
  std::vector<LokrEntry> entries;
  entries.reserve(header.members().size());
  std::map<std::string, Group, std::less<>> groups;

  for (const auto& [key, value] : header.members()) {
    if (key == "__metadata__") continue;
    const auto* dtype = value.find("dtype");
    const auto* shape = value.find("shape");
    const auto* offsets = value.find("data_offsets");
    if (!dtype || !dtype->is_string() || !shape || !shape->is_array() ||
        !offsets || !offsets->is_array() || offsets->items().size() != 2) {
      throw std::runtime_error(path.string() + ": invalid safetensors entry " +
                               key);
    }
    LokrEntry entry;
    entry.dtype = dtype->str();
    for (const auto& dim : shape->items()) {
      entry.shape.push_back(static_cast<int>(dim.as_double()));
    }
    entry.begin = static_cast<std::size_t>(offsets->items()[0].as_double());
    entry.end = static_cast<std::size_t>(offsets->items()[1].as_double());
    if (entry.end < entry.begin || data_offset + entry.end > size) {
      throw std::runtime_error(path.string() + ": truncated tensor " + key);
    }
    entries.push_back(std::move(entry));
    const LokrEntry* ptr = &entries.back();

    std::string_view kview = key;
    if (kview.ends_with(".lokr_w1")) {
      std::string mod(StripKnownPrefix(kview.substr(0, kview.size() - 8)));
      groups[mod].w1 = ptr;
    } else if (kview.ends_with(".lokr_w2")) {
      std::string mod(StripKnownPrefix(kview.substr(0, kview.size() - 8)));
      groups[mod].w2 = ptr;
    } else if (kview.ends_with(".alpha")) {
      std::string mod(StripKnownPrefix(kview.substr(0, kview.size() - 6)));
      groups[mod].alpha = ptr;
    } else {
      throw std::runtime_error(path.string() + ": unsupported LoKr key " + key);
    }
  }

  std::vector<LokrPatch> patches;
  patches.reserve(groups.size());
  for (const auto& [mod, grp] : groups) {
    if (!grp.w1 || !grp.w2 || grp.w1->shape.size() != 2 ||
        grp.w2->shape.size() != 2) {
      throw std::runtime_error(path.string() + ": incomplete LoKr module " +
                               mod);
    }
    std::string target = mod + ".weight";
    auto target_shape = base(target);
    if (target_shape.empty() && mod.ends_with(".adaln_proj")) {
      const std::string linear_target = mod + ".linear.weight";
      auto linear_shape = base(linear_target);
      if (!linear_shape.empty()) {
        target = linear_target;
        target_shape = std::move(linear_shape);
      }
    }
    if (target_shape.size() != 2) {
      throw std::runtime_error(path.string() + ": unknown MiniMax H3 target " +
                               target);
    }
    LokrPatch patch;
    patch.target = target;
    patch.w1_rows = grp.w1->shape[0];
    patch.w1_cols = grp.w1->shape[1];
    patch.w2_rows = grp.w2->shape[0];
    patch.w2_cols = grp.w2->shape[1];
    if (patch.w1_rows * patch.w2_rows != target_shape[0] ||
        patch.w1_cols * patch.w2_cols != target_shape[1]) {
      throw std::runtime_error(path.string() + ": LoKr shape mismatch for " +
                               target);
    }
    const std::size_t w1_count =
        static_cast<std::size_t>(patch.w1_rows) * patch.w1_cols;
    const std::size_t w2_count =
        static_cast<std::size_t>(patch.w2_rows) * patch.w2_cols;
    patch.w1 = ReadSafetensorsFloats(mapped + data_offset + grp.w1->begin,
                                     grp.w1->dtype, w1_count);
    patch.w2 = ReadSafetensorsFloats(mapped + data_offset + grp.w2->begin,
                                     grp.w2->dtype, w2_count);
    const int rank = std::min(patch.w1_rows, patch.w1_cols);
    float alpha = static_cast<float>(rank);
    if (grp.alpha != nullptr && grp.alpha->end > grp.alpha->begin) {
      alpha = ReadSafetensorsFloats(mapped + data_offset + grp.alpha->begin,
                                    grp.alpha->dtype, 1)[0];
    }
    patch.scale = rank > 0 ? alpha / static_cast<float>(rank) : 1.0F;
    patches.push_back(std::move(patch));
  }
  return patches;
}

}  // namespace

LoraAdapter::LoraAdapter(const std::filesystem::path& path,
                         const BaseShape& base)
    : lora::Adapter(path, Schema(base)) {}

LoadedAdapter LoadMinimaxAdapter(const std::filesystem::path& path,
                                 const LoraAdapter::BaseShape& base) {
  platform::RawMappedFile mapped;
  std::string open_error;
  if (!mapped.Open(path, platform::MapHint::kNone, &open_error)) {
    throw std::runtime_error(path.string() + ": " +
                             (open_error.empty() ? "cannot open adapter"
                                                 : open_error));
  }
  const auto* bytes = static_cast<const std::byte*>(mapped.data);
  const std::size_t file_size = mapped.size;
  if (file_size < 8) {
    throw std::runtime_error(path.string() + ": invalid safetensors file");
  }
  std::uint64_t header_len = 0;
  std::memcpy(&header_len, bytes, 8);
  if (header_len == 0 || 8 + header_len > file_size) {
    throw std::runtime_error(path.string() + ": invalid safetensors header");
  }
  const json::Value header = json::parse(std::string_view(
      reinterpret_cast<const char*>(bytes + 8), header_len));
  bool is_lokr = false;
  for (const auto& [key, val] : header.members()) {
    (void)val;
    if (key.find(".lokr_") != std::string::npos) {
      is_lokr = true;
      break;
    }
  }
  if (is_lokr) {
    LoadedAdapter out;
    out.lokr_patches =
        ParseLokrFile(path, base, bytes, file_size, 8 + header_len, header);
    mapped.Close();
    return out;
  }
  mapped.Close();
  LoadedAdapter out;
  out.lora = std::make_shared<LoraAdapter>(path, base);
  return out;
}

}  // namespace gufo::minimax_h3
