#include <cassert>
#include <cstddef>
#include <system_error>

#include "src/core/platform/prefault.hpp"

int main() {
  constexpr std::size_t bytes = 32ULL << 20;
#if defined(_WIN32)
  auto* memory = static_cast<char*>(
      ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  assert(memory != nullptr);
  SYSTEM_INFO info{};
  ::GetSystemInfo(&info);
  const auto page_size = info.dwPageSize;
#else
  auto* memory =
      static_cast<char*>(::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(memory != MAP_FAILED);
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  assert(page_size > 0);
#endif
  gufo::platform::PrefaultMappedRange(memory + 1, bytes - 1);
  // A fault in a later chunk must be reported after all workers have joined.
  auto* protected_page = memory + (16ULL << 20) + page_size;
#if defined(_WIN32)
  DWORD old_protection = 0;
  assert(::VirtualProtect(protected_page, page_size, PAGE_NOACCESS,
                          &old_protection));
#else
  assert(::mprotect(protected_page, page_size, PROT_NONE) == 0);
#endif
  bool failed = false;
  try {
    gufo::platform::PrefaultMappedRange(memory, bytes);
  } catch (const std::system_error& error) {
    failed = true;
#if defined(_WIN32)
    assert(error.code() ==
           std::error_code(ERROR_NOACCESS, std::system_category()));
#endif
  }
  assert(failed);
#if defined(_WIN32)
  assert(::VirtualFree(memory, 0, MEM_RELEASE));
#else
  assert(::munmap(memory, bytes) == 0);
#endif
}
