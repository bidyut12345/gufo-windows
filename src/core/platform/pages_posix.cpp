#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <thread>

#include "src/core/platform/pages.hpp"

namespace gufo::platform {
namespace {

// Ordinary anonymous pages, populated before use. Opting out of THP before
// population prevents later huge-page promotion from invalidating GPU
// mappings of the same range.
class CommittedRange {
public:
  static void* Commit(std::size_t bytes) noexcept {
    void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED)
      return nullptr;
    // Kernels built without THP reject this advice with EINVAL; their pages
    // cannot be promoted. All other failures must release the mapping.
    if (madvise(data, bytes, MADV_NOHUGEPAGE) != 0 && errno != EINVAL) {
      munmap(data, bytes);
      return nullptr;
    }
    if (!Populate(data, bytes)) {
      munmap(data, bytes);
      return nullptr;
    }
    return data;
  }

private:
  static bool Populate(void* data, std::size_t bytes) noexcept {
    constexpr std::size_t page = 4096;
    const auto workers = std::min<std::size_t>(4, bytes / (128ULL << 20));
    std::atomic<bool> populated{true};
    const auto populate = [&](std::size_t begin, std::size_t end) {
      // A long population call holds the process's mapping lock while
      // faulting pages. Bound each call so request-side heap growth can make
      // progress.
      constexpr std::size_t piece = 64ULL << 10;
      for (auto offset = begin; offset < end; offset += piece) {
        if (!populated.load(std::memory_order_relaxed))
          break;
        if (madvise(static_cast<std::byte*>(data) + offset,
                    std::min(piece, end - offset), MADV_POPULATE_WRITE) != 0) {
          populated.store(false, std::memory_order_relaxed);
          break;
        }
      }
    };
    if (workers >= 2) {
      // Large serial page faults can hold up inference even when growth runs
      // on a background thread. Populate disjoint ordinary-page ranges, then
      // join before exposing any of the backing.
      std::array<std::jthread, 3> faults;
      std::size_t next = 0;
      for (std::size_t i = 0; i + 1 < workers; ++i) {
        const auto end = (bytes / page) * (i + 1) / workers * page;
        try {
          faults[i] = std::jthread(populate, next, end);
        } catch (...) {
          // Thread exhaustion falls back to populating the remaining range
          // here. Every started worker still joins.
          break;
        }
        next = end;
      }
      populate(next, bytes);
    } else {
      populate(0, bytes);
    }
    return populated.load(std::memory_order_relaxed);
  }
};

}  // namespace

void* CommitZeroedPages(std::size_t bytes) noexcept {
  return CommittedRange::Commit(bytes);
}

void ReleaseCommittedPages(void* base, std::size_t bytes) noexcept {
  munmap(base, bytes);
}

}  // namespace gufo::platform
