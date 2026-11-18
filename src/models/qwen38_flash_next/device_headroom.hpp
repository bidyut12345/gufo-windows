#ifndef GUFO_QWEN38_FLASH_NEXT_DEVICE_HEADROOM_HPP_
#define GUFO_QWEN38_FLASH_NEXT_DEVICE_HEADROOM_HPP_

#include <atomic>
#include <cstddef>

namespace qfn::testing {

// Process-local cap on the free device memory the snapshot paths see. Windows
// imports hipMemGetInfo straight from the runtime DLL, so the test's ELF
// interposition cannot apply there; the executor consults this instead. The
// maximum removes the cap. Defined in executor.cpp: a header inline would
// give the test and the library separate copies under MSVC.
extern std::atomic<std::size_t> reported_device_free_limit;

inline std::size_t ClampReportedFree(std::size_t free_bytes) {
  const auto limit =
      reported_device_free_limit.load(std::memory_order_relaxed);
  return free_bytes < limit ? free_bytes : limit;
}

}  // namespace qfn::testing

#endif  // GUFO_QWEN38_FLASH_NEXT_DEVICE_HEADROOM_HPP_
