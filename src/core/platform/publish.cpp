#include "src/core/platform/publish.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace gufo::platform {
namespace {

#if defined(_WIN32)
std::string WindowsErrorText(const char* what) {
  return std::string(what) + " failed with Windows error " +
         std::to_string(GetLastError());
}
#endif

}  // namespace

// --- FileLock ---------------------------------------------------------------

FileLock::FileLock(FileLock&& other) noexcept : handle_(other.handle_) {
  other.handle_ = detail::kInvalidPublicationHandle;
}

FileLock& FileLock::operator=(FileLock&& other) noexcept {
  if (this != &other) {
    Release();
    handle_ = other.handle_;
    other.handle_ = detail::kInvalidPublicationHandle;
  }
  return *this;
}

FileLock::~FileLock() {
  Release();
}

void FileLock::Release() noexcept {
  if (handle_ == detail::kInvalidPublicationHandle)
    return;
#if defined(_WIN32)
  HANDLE file = static_cast<HANDLE>(handle_);
  // Byte-range locks are released by closing any handle to the file.
  CloseHandle(file);
#else
  ::close(handle_);
#endif
  handle_ = detail::kInvalidPublicationHandle;
}

FileLock FileLock::Acquire(const std::string& path) {
#if defined(_WIN32)
  HANDLE file = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::system_error(static_cast<int>(GetLastError()),
                            std::system_category(), "open lock file");
  OVERLAPPED overlapped{};
  if (!LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK, 0, ~0u, ~0u, &overlapped)) {
    const auto error = static_cast<int>(GetLastError());
    CloseHandle(file);
    throw std::system_error(error, std::system_category(), "lock file");
  }
  return FileLock(file);
#else
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0)
    throw std::system_error(errno, std::generic_category(), "open lock file");
  if (::flock(fd, LOCK_EX) != 0) {
    const int error = errno;
    ::close(fd);
    throw std::system_error(error, std::generic_category(), "lock file");
  }
  return FileLock(fd);
#endif
}

// --- AtomicPublishedFile ----------------------------------------------------

AtomicPublishedFile::AtomicPublishedFile(AtomicPublishedFile&& other) noexcept
    : handle_(other.handle_), temporary_(std::move(other.temporary_)) {
  other.handle_ = detail::kInvalidPublicationHandle;
  other.temporary_.clear();
}

AtomicPublishedFile& AtomicPublishedFile::operator=(
    AtomicPublishedFile&& other) noexcept {
  if (this != &other) {
    Discard();
    handle_ = other.handle_;
    temporary_ = std::move(other.temporary_);
    other.handle_ = detail::kInvalidPublicationHandle;
    other.temporary_.clear();
  }
  return *this;
}

AtomicPublishedFile::~AtomicPublishedFile() {
  Discard();
}

void AtomicPublishedFile::Discard() noexcept {
  if (handle_ != detail::kInvalidPublicationHandle) {
#if defined(_WIN32)
    CloseHandle(static_cast<HANDLE>(handle_));
#else
    ::close(handle_);
#endif
    handle_ = detail::kInvalidPublicationHandle;
  }
  std::error_code ignored;
  if (!temporary_.empty())
    std::filesystem::remove(temporary_, ignored);
  temporary_.clear();
}

AtomicPublishedFile AtomicPublishedFile::Create(
    const std::filesystem::path& final_path, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return AtomicPublishedFile();
  };
#if defined(_WIN32)
  for (int attempt = 0; attempt < 32; ++attempt) {
    std::random_device device;
    const auto suffix =
        ".tmp." + std::to_string(static_cast<std::uint32_t>(device()));
    const auto temporary = final_path.string() + suffix;
    HANDLE file = CreateFileA(temporary.c_str(), GENERIC_WRITE, 0, nullptr,
                              CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      if (GetLastError() == ERROR_FILE_EXISTS)
        continue;
      return fail(WindowsErrorText("create temporary file"));
    }
    return AtomicPublishedFile(file, temporary);
  }
  return fail("temporary file name collisions exhausted");
#else
  std::string temporary = final_path.string() + ".tmp.XXXXXX";
  const int fd = ::mkstemp(temporary.data());
  if (fd < 0)
    return fail("cannot create temporary file: " + std::to_string(errno));
  return AtomicPublishedFile(fd, temporary);
#endif
}

bool AtomicPublishedFile::Write(const void* data, std::size_t bytes,
                                std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return false;
  };
  const auto* cursor = static_cast<const char*>(data);
  std::size_t written = 0;
  while (written < bytes) {
#if defined(_WIN32)
    DWORD chunk = 0;
    if (!WriteFile(static_cast<HANDLE>(handle_), cursor + written,
                   static_cast<DWORD>(bytes - written), &chunk, nullptr) ||
        chunk == 0)
      return fail(WindowsErrorText("write temporary file"));
    written += chunk;
#else
    const ssize_t count = ::write(handle_, cursor + written, bytes - written);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return fail("cannot write temporary file: " + std::to_string(errno));
    written += static_cast<std::size_t>(count);
#endif
  }
  return true;
}

bool AtomicPublishedFile::Commit(const std::filesystem::path& final_path,
                                 std::string* error) {
  const auto fail = [&](std::string message) {
    Discard();
    if (error != nullptr)
      *error = std::move(message);
    return false;
  };
#if defined(_WIN32)
  if (!FlushFileBuffers(static_cast<HANDLE>(handle_)))
    return fail(WindowsErrorText("flush temporary file"));
  const std::string source = temporary_.string();
  // Release the handle without deleting so the replace can proceed.
  CloseHandle(static_cast<HANDLE>(handle_));
  handle_ = detail::kInvalidPublicationHandle;
  temporary_.clear();
  if (!MoveFileExA(source.c_str(), final_path.string().c_str(),
                   MOVEFILE_REPLACE_EXISTING)) {
    std::error_code ignored;
    std::filesystem::remove(source, ignored);
    return fail(WindowsErrorText("replace file"));
  }
  return true;
#else
  const int fd = handle_;
  if (::fsync(fd) != 0)
    return fail("cannot flush temporary file: " + std::to_string(errno));
  std::error_code rename_error;
  std::filesystem::rename(temporary_, final_path, rename_error);
  if (rename_error)
    return fail("cannot replace file");
  // Persist the rename itself.
  temporary_.clear();
  handle_ = detail::kInvalidPublicationHandle;
  ::close(fd);
  const auto parent = final_path.has_parent_path() ? final_path.parent_path()
                                                   : std::filesystem::path(".");
  const int directory =
      ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0)
    return fail("file replaced, but cannot open containing directory: " +
                std::to_string(errno));
  const int result = ::fsync(directory);
  const int directory_error = errno;
  ::close(directory);
  if (result != 0)
    return fail("file replaced, but cannot flush containing directory: " +
                std::to_string(directory_error));
  return true;
#endif
}

}  // namespace gufo::platform
