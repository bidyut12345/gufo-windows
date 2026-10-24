#ifndef GUFO_CORE_PLATFORM_OS_INFO_HPP_
#define GUFO_CORE_PLATFORM_OS_INFO_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::platform {

/// OS page size in bytes; never zero.
std::size_t PageSizeBytes();

/// Best-effort physically available memory in bytes. Linux reports
/// sysconf(_SC_AVPHYS_PAGES) times the page size; Windows reports
/// GlobalMemoryStatusEx available physical memory. Callers layer their own
/// refinements (MemAvailable parsing, cgroup limits) on top. Returns 0
/// when the OS cannot answer.
std::uint64_t AvailableMemoryBytes();

/// Id of the running process. Used to make temporary output names
/// collision-free between concurrent processes.
int ProcessId() noexcept;

/// Minor and major page faults of this process since it started.
struct PageFaults {
  std::uint64_t minor{0};
  std::uint64_t major{0};
};

/// Reads the process page-fault counters. Returns zeros when the
/// platform cannot report them; Windows reports every fault as minor.
PageFaults ReadPageFaults() noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_OS_INFO_HPP_
