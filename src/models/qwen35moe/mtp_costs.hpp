#ifndef GUFO_MODELS_QWEN35MOE_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN35MOE_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen35moe {

// gfx1151, Ornith-1.5-35B Q4_K_M target, 2026-10-19. Milliseconds per
// cohort, including catch-up, recursive proposals and target verification.
// Rows are widths 1..8 (width 1 is headless MTP catch-up plus ordinary
// decode). The context-0 C1 row is measured on the chained greedy path
// (device-side draft chain, one full head pass per proposal): width 1-2
// come from capped `gufo bench` medians, width 0 from the AR cycle plus a
// one-row headless catch-up, and the ladder beyond from the width probes at
// --draft-tokens 4/5/7. The context slope and the C2-C8 cohort shape keep
// the Flash-Next measurements, scaled per column by the measured C1 change;
// Ornith concurrency cohorts learn online like Flash-Next's.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {15.5F, 19.5F, 22.5F, 27.5F, 34.0F, 41.0F, 49.0F, 56.5F},      // C1
        {18.6F, 24.6F, 29.3F, 35.8F, 44.2F, 51.2F, 60.5F, 68.7F},      // C2
        {23.3F, 33.3F, 39.9F, 49.8F, 62.4F, 75.1F, 90.1F, 102.2F},     // C4
        {27.6F, 43.2F, 51.4F, 66.7F, 82.3F, 103.8F, 124.6F, 144.4F},   // C6
        {31.8F, 50.3F, 61.8F, 82.1F, 102.9F, 128.5F, 155.4F, 181.2F},  // C8
    },
    {
        // Context 4096
        {16.0F, 20.3F, 23.5F, 28.7F, 35.7F, 42.8F, 51.7F, 59.0F},      // C1
        {19.8F, 26.5F, 31.9F, 40.2F, 49.0F, 56.3F, 66.7F, 76.5F},      // C2
        {25.1F, 36.1F, 43.6F, 54.3F, 68.1F, 82.5F, 98.5F, 112.0F},     // C4
        {30.2F, 47.4F, 57.3F, 74.5F, 92.5F, 116.1F, 140.6F, 163.2F},   // C6
        {35.5F, 56.1F, 69.8F, 92.5F, 116.7F, 144.4F, 177.6F, 208.6F},  // C8
    },
    {
        // Context 32768
        {17.2F, 21.4F, 24.2F, 29.5F, 37.2F, 44.4F, 53.9F, 62.1F},       // C1
        {22.1F, 30.0F, 36.7F, 46.4F, 57.5F, 70.0F, 83.1F, 95.6F},       // C2
        {29.6F, 43.3F, 52.6F, 67.1F, 85.6F, 103.4F, 124.8F, 143.4F},    // C4
        {37.0F, 59.6F, 71.8F, 92.2F, 114.8F, 147.6F, 180.0F, 210.3F},   // C6
        {44.5F, 71.9F, 90.9F, 121.8F, 155.1F, 191.3F, 236.4F, 278.7F},  // C8
    },
};

// Sampled policy uses fixed configured capacity for reproducibility; greedy
// batch policy uses these curves to bootstrap each physical occupancy.
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured QSA slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto interval = context <= 4096 ? 0 : 1;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : static_cast<float>(std::min(context, 262144U) - 4096) / 28672.0F;
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = kMtpCycleMilliseconds[interval][cohort][i];
    const auto high = kMtpCycleMilliseconds[interval + 1][cohort][i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal =
          high - kMtpCycleMilliseconds[interval + 1][cohort][i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  return costs;
}

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_MTP_COSTS_HPP_
