#include "src/core/platform/pages.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace gufo::platform {

void* CommitZeroedPages(std::size_t bytes) noexcept {
  // Reserving and committing together hands out zero-filled ordinary pages;
  // the OS maps them on first touch.
  return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void ReleaseCommittedPages(void* base, std::size_t) noexcept {
  VirtualFree(base, 0, MEM_RELEASE);
}

}  // namespace gufo::platform
