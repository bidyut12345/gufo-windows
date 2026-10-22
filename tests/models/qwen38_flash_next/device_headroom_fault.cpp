// Process-local device headroom limit for snapshot storage tests. Linked only
// into the model-local snapshot test, not the model or serving library.
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <limits>

#include "src/models/qwen38_flash_next/device_headroom.hpp"

namespace {
#if !defined(_WIN32)
std::atomic<std::size_t> free_limit{std::numeric_limits<std::size_t>::max()};
#endif
}  // namespace

/// Caps the free device memory HIP reports; the maximum removes the cap.
void LimitReportedDeviceFree(std::size_t bytes) {
#if defined(_WIN32)
  qfn::testing::reported_device_free_limit.store(bytes,
                                                 std::memory_order_relaxed);
#else
  free_limit = bytes;
#endif
}

#if !defined(_WIN32)
extern "C" hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
  using Function = hipError_t (*)(std::size_t*, std::size_t*);
  static const auto next =
      reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipMemGetInfo"));
  const auto result = next ? next(free, total) : hipErrorUnknown;
  if (result == hipSuccess && free != nullptr && *free > free_limit)
    *free = free_limit;
  return result;
}
#endif  // !defined(_WIN32)
