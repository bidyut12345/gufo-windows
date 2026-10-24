#ifndef GUFO_CORE_PLATFORM_PAGES_HPP_
#define GUFO_CORE_PLATFORM_PAGES_HPP_

#include <cstddef>

namespace gufo::platform {

/// Reserves and commits `bytes` of zero-filled anonymous memory, opting out
/// of transparent huge pages and touching every page before returning, so
/// later promotions cannot invalidate device mappings. Returns nullptr on
/// failure. Pair with ReleaseCommittedPages(); registration with a device
/// runtime stays with the caller.
void* CommitZeroedPages(std::size_t bytes) noexcept;

/// Releases memory obtained from CommitZeroedPages(). `bytes` must match
/// the commit call.
void ReleaseCommittedPages(void* base, std::size_t bytes) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PAGES_HPP_
