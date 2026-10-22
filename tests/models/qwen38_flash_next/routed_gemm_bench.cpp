// Times the routed expert GEMMs of one Flash-Next layer under the two
// shipped quant presets: the UD-Q4_K_XL shapes (paired Q4_K gate/up, Q5_1
// down) and the UD-IQ4_XS shapes (separate IQ3_S gate/up, IQ4_NL down),
// with the executor's tile maps over a skewed top-10 routing of 512 experts,
// or the per-expert counts of each line of a counts file (one layer per
// line; tokens = sum / 10). Prints the median time, the streamed weight
// bandwidth of the active experts and FNV-1a hashes of the outputs, then
// the totals over all layers. A second section times the decode-tier MoE
// vector projections (gated gate/up and the down projection) at 1 and 8
// tokens, the single-decode and speculative-verify shapes.
//
// routed_gemm_bench [reps] [tokens | counts-file] [preset: q4kxl|iq4xs|all]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kExperts = 512;
constexpr std::uint32_t kUsed = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kFf = 640;
constexpr std::size_t kQ4KBytes = 144;   // per 256 elements
constexpr std::size_t kQ51Bytes = 24;    // per 32 elements
constexpr std::size_t kIQ3SBytes = 110;  // per 256 elements
constexpr std::size_t kIQ4NLBytes = 18;  // per 32 elements

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

template<typename T>
T* Alloc(std::size_t count) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, count * sizeof(T)), "hipMalloc");
  CheckHip(hipMemset(p, 0, count * sizeof(T)), "hipMemset");
  return static_cast<T*>(p);
}

template<typename T>
std::uint64_t Hash(const T* device, std::size_t count) {
  std::vector<std::uint8_t> host(count * sizeof(T));
  CheckHip(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

std::uint16_t HalfBits(float v) {
  const __half h = __float2half(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

/// Random quantized blocks whose F16 scale fields hold small finite values.
std::vector<std::uint8_t> MakeBlocks(std::size_t blocks, std::size_t bytes,
                                     std::uint32_t seed) {
  std::vector<std::uint8_t> w(blocks * bytes);
  for (std::size_t i = 0; i < w.size(); i += 4) {
    const std::uint32_t r = Next(&seed);
    std::memcpy(&w[i], &r, std::min<std::size_t>(4, w.size() - i));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint16_t d = HalfBits(0.01F + 0.02F * (Next(&seed) % 97) / 97);
    const std::uint16_t m = HalfBits(0.005F + 0.01F * (Next(&seed) % 89) / 89);
    std::memcpy(&w[b * bytes], &d, 2);
    std::memcpy(&w[b * bytes + 2], &m, 2);
  }
  return w;
}

template<typename F>
float Median(int reps, F&& run) {
  hipEvent_t a;
  hipEvent_t b;
  CheckHip(hipEventCreate(&a), "event");
  CheckHip(hipEventCreate(&b), "event");
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    CheckHip(hipEventRecord(a), "record");
    run();
    CheckHip(hipEventRecord(b), "record");
    CheckHip(hipEventSynchronize(b), "sync");
    float t = 0.0F;
    CheckHip(hipEventElapsedTime(&t, a, b), "elapsed");
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

/// Synthetic skewed routing: expert e has weight 1 / (rank + 16).
std::vector<std::uint32_t> SkewedCounts(std::uint32_t tokens) {
  std::uint32_t seed = 0x9E3779B9U;
  std::vector<std::uint32_t> rank(kExperts);
  std::iota(rank.begin(), rank.end(), 0U);
  for (std::uint32_t i = kExperts - 1; i > 0; --i)
    std::swap(rank[i], rank[Next(&seed) % (i + 1)]);
  std::vector<double> cdf(kExperts);
  double sum = 0.0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    sum += 1.0 / (rank[e] + 16.0);
    cdf[e] = sum;
  }
  std::vector<std::uint32_t> counts(kExperts, 0);
  std::vector<std::uint32_t> row;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    row.clear();
    while (row.size() < kUsed) {
      const double x = sum * (Next(&seed) / 4294967296.0);
      const auto e = static_cast<std::uint32_t>(
          std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
      if (std::find(row.begin(), row.end(), e) == row.end())
        row.push_back(e);
    }
    for (std::uint32_t e : row)
      ++counts[e];
  }
  return counts;
}

// Per-token top-10 routing that allows the same expert across tokens, the
// shape the speculative verifier's grouped projections see.
std::vector<std::int32_t> SkewedIds(std::uint32_t tokens) {
  std::uint32_t seed = 0x0BADC0DEU;
  std::vector<std::uint32_t> rank(kExperts);
  std::iota(rank.begin(), rank.end(), 0U);
  for (std::uint32_t i = kExperts - 1; i > 0; --i)
    std::swap(rank[i], rank[Next(&seed) % (i + 1)]);
  std::vector<double> cdf(kExperts);
  double sum = 0.0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    sum += 1.0 / (rank[e] + 16.0);
    cdf[e] = sum;
  }
  std::vector<std::int32_t> ids(std::size_t{tokens} * kUsed);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::uint32_t j = 0; j < kUsed; ++j) {
      const double x = sum * (Next(&seed) / 4294967296.0);
      ids[std::size_t{t} * kUsed + j] = static_cast<std::int32_t>(
          std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
    }
  }
  return ids;
}

struct WeightFormat {
  q::WeightType type;
  std::size_t block_bytes;
  int block_elems;
};

constexpr WeightFormat kQ4K{q::WeightType::kQ4_K, kQ4KBytes, 256};
constexpr WeightFormat kQ51{q::WeightType::kQ5_1, kQ51Bytes, 32};
constexpr WeightFormat kIQ3S{q::WeightType::kIQ3_S, kIQ3SBytes, 256};
constexpr WeightFormat kIQ4NL{q::WeightType::kIQ4_NL, kIQ4NLBytes, 32};

/// The two shipped presets. `pair` mirrors WmmaExperts: Q4_K pairs the
/// gate/up projections over 128-row tiles, IQ3_S runs them separately.
struct Preset {
  const char* name;
  WeightFormat gate;
  bool pair;
  WeightFormat down;
};

constexpr Preset kPresets[] = {
    {"q4kxl", kQ4K, true, kQ51},
    {"q4kxl_sep", kQ4K, false, kQ51},
    {"iq4xs", kIQ3S, true, kIQ4NL},
    {"iq4xs_sep", kIQ3S, false, kIQ4NL},
};

struct Result {
  float pair_ms;
  float down_ms;
  double pair_gb;
  double down_gb;
};

Result RunLayer(const Preset& preset, const std::vector<std::uint32_t>& counts,
                int reps, const void* d_gate, const void* d_up,
                const void* d_down, bool verify_pair = false) {
  const std::uint32_t slots = std::accumulate(counts.begin(), counts.end(), 0U);
  const std::uint32_t tokens = slots / kUsed;
  // Expert e takes the next counts[e] tokens cyclically, so every token
  // receives kUsed distinct experts.
  std::vector<std::int32_t> ids(slots);
  std::vector<std::uint32_t> filled(tokens, 0);
  std::uint32_t cursor = 0;
  std::uint32_t active = 0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    active += counts[e] != 0;
    for (std::uint32_t i = 0; i < counts[e]; ++i) {
      const std::uint32_t t = cursor++ % tokens;
      ids[std::size_t{t} * kUsed + filled[t]++] = static_cast<std::int32_t>(e);
    }
  }
  // The executor's maps: 64-token down tiles, then 128-token pair tiles;
  // below 1,024 tokens one 16- or 48-token map serves separate gate, up and
  // down projections.
  const bool wide = tokens >= 1024;
  const std::uint32_t narrow_rows = slots >= 16 * kExperts ? 48 : 16;
  // Q4_K separate projections only accept the narrow tile widths, so the
  // structure-isolation preset runs their narrow map even at wide tokens.
  const std::uint32_t separate_rows =
      !preset.pair
          ? (wide && getenv("IQS_TILES") == nullptr ? 64 : narrow_rows)
          : narrow_rows;  // IQS_TILES=48 tests the 48-row instantiation
  std::vector<std::int32_t> tiles;
  const auto append = [&](std::uint32_t rows) {
    for (std::uint32_t e = 0; e < kExperts; ++e)
      for (std::uint32_t j = 0; j < (counts[e] + rows - 1) / rows; ++j)
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
  };
  append(separate_rows);
  const auto down_tiles = static_cast<std::uint32_t>(tiles.size());
  if (wide && preset.pair)
    append(128);
  const auto pair_tiles = static_cast<std::uint32_t>(tiles.size()) - down_tiles;
  std::printf("tokens %u  active %3u  down tiles %3u  pair tiles %3u  ", tokens,
              active, down_tiles, pair_tiles);

  const std::size_t compact = q::RoutedCompactRows(slots, kExperts);
  auto* d_ids = Upload(ids);
  auto* d_counts = Upload(counts);
  auto* d_tiles = Upload(tiles);
  auto* d_bounds = Alloc<std::int32_t>(kExperts + 1);
  auto* d_cursors = Alloc<std::int32_t>(kExperts);
  auto* d_rows_token = Alloc<std::int32_t>(compact);
  auto* d_rows_slot = Alloc<std::int32_t>(compact);
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, tokens, kUsed, kExperts, nullptr);

  std::uint32_t seed = 0x12345678U;
  std::vector<__half> x(std::size_t{tokens} * kHidden);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F);
  auto* d_x = Upload(x);
  auto* d_mid = Alloc<__half>(std::size_t{slots} * kFf);
  // The separate IQ3_S projections materialize the gate as F32 for the
  // fused SwiGLU, exactly like the executor's narrow path.
  auto* d_gate_out = Alloc<float>(std::size_t{slots} * kFf);
  auto* d_out = Alloc<__half>(std::size_t{slots} * kHidden);
  CheckHip(hipDeviceSynchronize(), "setup");

  const auto gate_run = [&](const void* w, bool with_swiglu) {
    float* swiglu = nullptr;
    float* out_f = nullptr;
    __half* out_h = nullptr;
    if (with_swiglu) {
      swiglu = d_gate_out;
      out_h = d_mid;
    } else {
      out_f = d_gate_out;
    }
    if (!q::RoutedF16Gemm(w, preset.gate.type, d_x, d_tiles, down_tiles,
                          separate_rows, d_bounds, d_rows_token, d_rows_slot,
                          swiglu, out_f, out_h, kFf, kHidden, nullptr))
      throw std::runtime_error("gate GEMM rejected");
  };
  const auto pair = [&] {
    if (preset.pair) {
      if (wide) {
        if (!q::RoutedGatedF16Gemm(d_gate, d_up, preset.gate.type, d_x,
                                   d_tiles + down_tiles, pair_tiles, d_bounds,
                                   d_rows_token, d_rows_slot, d_mid, kFf,
                                   kHidden, nullptr))
          throw std::runtime_error("gated GEMM rejected");
        return;
      }
      gate_run(d_gate, false);
      gate_run(d_up, true);
      return;
    }
    // IQ3_S: separate gate then fused-swiglu up, over the wide 64-row map.
    gate_run(d_gate, false);
    gate_run(d_up, true);
  };
  const auto down = [&] {
    if (!q::RoutedF16Gemm(d_down, preset.down.type, d_mid, d_tiles, down_tiles,
                          wide ? 64 : narrow_rows, d_bounds, d_rows_slot,
                          d_rows_slot, nullptr, nullptr, d_out, kHidden, kFf,
                          nullptr))
      throw std::runtime_error("down GEMM rejected");
  };
  pair();
  down();
  if (verify_pair && preset.pair && wide) {
    // Correctness gate for the paired path: rerun gate/up as separate
    // projections into a second buffer. With uniform counts every bucket
    // row is live under both tile maps, so the F16 outputs must match bit
    // for bit. Q4_K/Q5_K separate projections only accept the 48-row map.
    const std::uint32_t verify_rows =
        preset.gate.type == q::WeightType::kIQ3_S ? 64 : 48;
    std::vector<std::int32_t> verify_tiles;
    for (std::uint32_t e = 0; e < kExperts; ++e)
      for (std::uint32_t j = 0; j < (counts[e] + verify_rows - 1) / verify_rows;
           ++j)
        verify_tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    auto* d_verify_tiles = Upload(verify_tiles);
    auto* d_mid2 = Alloc<__half>(std::size_t{slots} * kFf);
    if (!q::RoutedF16Gemm(d_gate, preset.gate.type, d_x, d_verify_tiles,
                          static_cast<std::uint32_t>(verify_tiles.size()),
                          verify_rows, d_bounds, d_rows_token, d_rows_slot,
                          nullptr, d_gate_out, nullptr, kFf, kHidden,
                          nullptr) ||
        !q::RoutedF16Gemm(d_up, preset.gate.type, d_x, d_verify_tiles,
                          static_cast<std::uint32_t>(verify_tiles.size()),
                          verify_rows, d_bounds, d_rows_token, d_rows_slot,
                          d_gate_out, nullptr, d_mid2, kFf, kHidden, nullptr))
      throw std::runtime_error("verify gate/up GEMM rejected");
    const auto a = Hash(d_mid, std::size_t{slots} * kFf);
    const auto b = Hash(d_mid2, std::size_t{slots} * kFf);
    std::printf(
        "pair verify %s (%016llx vs %016llx)\n", a == b ? "MATCH" : "MISMATCH",
        static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
    // Delta probe: with one nonzero activation k, each output column equals
    // one weight value, so the first differing column names the k position
    // the paired path mis-maps.
    if (const char* probe = std::getenv("IQS_PROBE_K")) {
      const int k_probe = std::atoi(probe);
      std::vector<__half> x_probe(std::size_t{tokens} * kHidden, __half(0.0F));
      for (std::uint32_t t = 0; t < tokens; ++t)
        x_probe[std::size_t{t} * kHidden + k_probe] = __half(1.0F);
      auto* d_x2 = Upload(x_probe);
      auto* d_mid3 = Alloc<__half>(std::size_t{slots} * kFf);
      auto* d_mid4 = Alloc<__half>(std::size_t{slots} * kFf);
      const void* up_probe = std::getenv("IQS_SAME") ? d_gate : d_up;
      if (!q::RoutedGatedF16Gemm(d_gate, up_probe, preset.gate.type, d_x2,
                                 d_tiles + down_tiles, pair_tiles, d_bounds,
                                 d_rows_token, d_rows_slot, d_mid3, kFf,
                                 kHidden, nullptr) ||
          !q::RoutedF16Gemm(d_gate, preset.gate.type, d_x2, d_verify_tiles,
                            static_cast<std::uint32_t>(verify_tiles.size()),
                            verify_rows, d_bounds, d_rows_token, d_rows_slot,
                            nullptr, d_gate_out, nullptr, kFf, kHidden,
                            nullptr) ||
          !q::RoutedF16Gemm(up_probe, preset.gate.type, d_x2, d_verify_tiles,
                            static_cast<std::uint32_t>(verify_tiles.size()),
                            verify_rows, d_bounds, d_rows_token, d_rows_slot,
                            d_gate_out, nullptr, d_mid4, kFf, kHidden, nullptr))
        throw std::runtime_error("probe GEMM rejected");
      const auto pa = Download(d_mid3, std::size_t{slots} * kFf);
      const auto pb = Download(d_mid4, std::size_t{slots} * kFf);
      std::size_t diffs = 0;
      std::vector<int> col_diffs(kFf, 0);
      for (std::size_t i = 0; i < pa.size(); ++i)
        if (std::memcmp(&pa[i], &pb[i], sizeof(__half)) != 0) {
          ++diffs;
          ++col_diffs[i % kFf];
        }
      // Element 0 has exact host-side expectations (MakeBlocks is seeded).
      const float first_p = __half2float(pa[0]);
      const float first_s = __half2float(pb[0]);
      std::printf("probe k=%d: diffs %zu pair %g separate %g\n", k_probe, diffs,
                  first_p, first_s);
      std::printf("matching cols:");
      for (int c = 0; c < static_cast<int>(kFf); ++c)
        if (col_diffs[c] == 0)
          std::printf(" %d", c);
      std::printf("\n");
      CheckHip(hipFree(d_x2), "probe x free");
      CheckHip(hipFree(d_mid3), "probe mid3 free");
      CheckHip(hipFree(d_mid4), "probe mid4 free");
    }
    CheckHip(hipFree(d_verify_tiles), "verify tiles free");
    CheckHip(hipFree(d_mid2), "verify mid free");
  }
  const float pair_ms = Median(reps, pair);
  const float down_ms = Median(reps, down);
  const double pair_gb = 2.0 * active * kFf * kHidden /
                         preset.gate.block_elems * preset.gate.block_bytes /
                         1e9;
  const double down_gb = 1.0 * active * kHidden * kFf /
                         preset.down.block_elems * preset.down.block_bytes /
                         1e9;
  std::printf(
      "gate/up %6.3f ms %5.1f GB/s %016llx  down %6.3f ms %5.1f GB/s "
      "%016llx\n",
      pair_ms, pair_gb / pair_ms * 1e3,
      static_cast<unsigned long long>(Hash(d_mid, std::size_t{slots} * kFf)),
      down_ms, down_gb / down_ms * 1e3,
      static_cast<unsigned long long>(
          Hash(d_out, std::size_t{slots} * kHidden)));
  for (void* p : std::initializer_list<void*>{
           d_ids, d_counts, d_tiles, d_bounds, d_cursors, d_rows_token,
           d_rows_slot, d_x, d_mid, d_gate_out, d_out})
    CheckHip(hipFree(p), "hipFree");
  return {pair_ms, down_ms, pair_gb, down_gb};
}

struct GemvWeight {
  const void* data;
  const WeightFormat* format;
};

/// Decode-tier MoE vector projections at one layer: gated gate/up and the
/// down projection, tokens 1 (single decode) and 8 (speculative verify).
void RunGemv(const Preset& preset, std::uint32_t tokens, int reps,
             const GemvWeight& gate, const GemvWeight& up,
             const GemvWeight& down) {
  const auto ids = SkewedIds(tokens);
  const std::uint32_t slots = tokens * kUsed;
  std::uint32_t seed = 0x5EED0001U;
  std::vector<float> x_f(std::size_t{tokens} * kHidden);
  for (auto& v : x_f)
    v = static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F;
  // The down projection consumes one F32 row per slot (the SwiGLU output).
  std::vector<float> mid_f(std::size_t{slots} * kFf);
  for (auto& v : mid_f)
    v = static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F;
  auto* d_ids = Upload(ids);
  auto* d_x = Upload(x_f);
  auto* d_mid_in = Upload(mid_f);
  auto* d_gate_out = Alloc<float>(std::size_t{slots} * kFf);
  auto* d_out = Alloc<float>(std::size_t{slots} * kHidden);
  CheckHip(hipDeviceSynchronize(), "gemv setup");

  const int gate_m = static_cast<int>(kFf);
  const int gate_k = static_cast<int>(kHidden);
  const auto gated = [&] {
    if (qfn_mmq_moe_gated_vec(static_cast<int>(gate.format->type), gate.data,
                              up.data, d_x, d_ids, d_gate_out, gate_m, gate_k,
                              static_cast<int>(tokens), kExperts, kUsed,
                              nullptr))
      throw std::runtime_error("gated vector launch failed");
  };
  const int down_m = static_cast<int>(kHidden);
  const int down_k = static_cast<int>(kFf);
  const auto down_run = [&] {
    if (qfn_mmq_moe_vec(static_cast<int>(down.format->type), down.data,
                        d_mid_in, d_ids, d_out, down_m, down_k,
                        static_cast<int>(tokens), kExperts, kUsed, nullptr))
      throw std::runtime_error("down vector launch failed");
  };
  gated();
  down_run();
  const float gated_ms = Median(reps, gated);
  const float down_ms = Median(reps, down_run);
  const double gated_gb = 2.0 * kFf * kHidden / gate.format->block_elems *
                          gate.format->block_bytes / 1e9;
  const double down_gb = 1.0 * kHidden * kFf / down.format->block_elems *
                         down.format->block_bytes / 1e9;
  std::printf(
      "gemv tokens %u  gated %7.4f ms %5.1f GB/s %016llx  down %7.4f ms "
      "%5.1f GB/s %016llx\n",
      tokens, gated_ms, gated_gb / gated_ms * 1e3,
      static_cast<unsigned long long>(
          Hash(d_gate_out, std::size_t{slots} * kFf)),
      down_ms, down_gb / down_ms * 1e3,
      static_cast<unsigned long long>(
          Hash(d_out, std::size_t{slots} * kHidden)));
  for (void* p :
       std::initializer_list<void*>{d_ids, d_x, d_mid_in, d_gate_out, d_out})
    CheckHip(hipFree(p), "hipFree");
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
  const std::string source = argc > 2 ? argv[2] : "2048";
  const std::string preset_name = argc > 3 ? argv[3] : "all";
  // "u<n>" builds uniform buckets of n rows: every tile map covers exactly
  // the live rows, which lets the pair path be verified bit-for-bit.
  const bool uniform = source.size() > 1 && source[0] == 'u' &&
                       std::all_of(source.begin() + 1, source.end(),
                                   [](char c) { return c >= '0' && c <= '9'; });
  const bool verify_pair = uniform;
  std::vector<std::vector<std::uint32_t>> layers;
  if (uniform) {
    const auto per_expert =
        static_cast<std::uint32_t>(std::stoul(source.substr(1)));
    layers.push_back(std::vector<std::uint32_t>(kExperts, per_expert));
  } else if (!source.empty() &&
             std::all_of(source.begin(), source.end(),
                         [](char c) { return c >= '0' && c <= '9'; })) {
    layers.push_back(
        SkewedCounts(static_cast<std::uint32_t>(std::stoul(source))));
  } else {
    std::ifstream in(source);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream fields(line);
      std::vector<std::uint32_t> counts;
      std::uint32_t n;
      while (fields >> n)
        counts.push_back(n);
      if (counts.size() == kExperts)
        layers.push_back(std::move(counts));
    }
    if (layers.empty())
      throw std::runtime_error("no 512-expert count lines in " + source);
  }
  for (const auto& preset : kPresets) {
    if (preset_name != "all" && preset_name != preset.name)
      continue;
    std::printf("== preset %s (%s gate/up, %s down)\n", preset.name,
                preset.pair ? "paired" : "separate", "");
    const std::size_t gate_blocks =
        std::size_t{kExperts} * kFf * kHidden / preset.gate.block_elems;
    auto* d_gate = Upload(MakeBlocks(gate_blocks, preset.gate.block_bytes, 1U));
    if (std::getenv("IQS_DUMP")) {
      auto hw = Download(reinterpret_cast<const std::uint8_t*>(d_gate), 256);
      for (std::size_t i = 0; i < hw.size(); ++i)
        std::printf("%02x", hw[i]), std::printf(i % 32 == 31 ? "\n" : " ");
    }
    auto* d_up = Upload(MakeBlocks(gate_blocks, preset.gate.block_bytes, 2U));
    const std::size_t down_blocks =
        std::size_t{kExperts} * kHidden * kFf / preset.down.block_elems;
    auto* d_down = Upload(MakeBlocks(down_blocks, preset.down.block_bytes, 3U));
    Result total{};
    for (const auto& counts : layers) {
      const Result r =
          RunLayer(preset, counts, reps, d_gate, d_up, d_down, verify_pair);
      total.pair_ms += r.pair_ms;
      total.down_ms += r.down_ms;
      total.pair_gb += r.pair_gb;
      total.down_gb += r.down_gb;
    }
    std::printf(
        "total %zu layers: gate/up %.2f ms (%.1f GB/s)  down %.2f ms "
        "(%.1f GB/s)\n",
        layers.size(), total.pair_ms, total.pair_gb / total.pair_ms * 1e3,
        total.down_ms, total.down_gb / total.down_ms * 1e3);

    const GemvWeight gate{d_gate, &preset.gate};
    const GemvWeight up{d_up, &preset.gate};
    const GemvWeight down{d_down, &preset.down};
    RunGemv(preset, 1, reps > 20 ? reps : 50, gate, up, down);
    RunGemv(preset, 8, reps > 20 ? reps : 50, gate, up, down);
    CheckHip(hipFree(d_gate), "gate free");
    CheckHip(hipFree(d_up), "up free");
    CheckHip(hipFree(d_down), "down free");
  }
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
