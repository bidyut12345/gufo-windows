#include "src/core/platform/net.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>

namespace gufo::net {
namespace {

#if defined(_WIN32)
void StartWinsock() {
  WSADATA data{};
  const int error = WSAStartup(MAKEWORD(2, 2), &data);
  if (error != 0)
    throw std::system_error(error, std::system_category(),
                            "initialize WinSock 2.2");
}
#endif

}  // namespace

void EnsureSocketRuntimeStarted() {
#if defined(_WIN32)
  static std::once_flag once;
  std::call_once(once, StartWinsock);
#endif
}

void CloseSocket(Socket socket) noexcept {
#if defined(_WIN32)
  closesocket(socket);
#else
  ::close(socket);
#endif
}

Socket CreateStreamSocket(int address_family) noexcept {
  return ::socket(address_family, SOCK_STREAM, 0);
}

int PollSocket(void* descriptors, std::uint32_t count, int timeout_ms) {
#if defined(_WIN32)
  return WSAPoll(static_cast<WSAPOLLFD*>(descriptors), count, timeout_ms);
#else
  return ::poll(static_cast<struct pollfd*>(descriptors), count, timeout_ms);
#endif
}

std::intptr_t SendNoSignal(Socket socket, const void* data, std::size_t bytes) {
#if defined(_WIN32)
  // Winsock's send takes an int length; clamp each call so oversized buffers
  // narrow safely. The send loops handle partial writes.
  const int chunk = static_cast<int>(std::min<std::size_t>(
      bytes, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return ::send(socket, static_cast<const char*>(data), chunk, 0);
#else
#ifdef MSG_NOSIGNAL
  return ::send(socket, data, bytes, MSG_NOSIGNAL);
#else
  return ::send(socket, data, bytes, 0);
#endif
#endif
}

bool SetNonBlocking(Socket socket) noexcept {
#if defined(_WIN32)
  u_long mode = 1;
  return ioctlsocket(socket, FIONBIO, &mode) == 0;
#else
  const int flags = ::fcntl(socket, F_GETFL, 0);
  if (flags < 0)
    return false;
  return ::fcntl(socket, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

std::intptr_t Receive(Socket socket, void* data, std::size_t bytes,
                      int flags) noexcept {
#if defined(_WIN32)
  // Winsock's recv takes an int length; clamp each call so oversized reads
  // narrow safely.
  const int chunk = static_cast<int>(std::min<std::size_t>(
      bytes, static_cast<std::size_t>(std::numeric_limits<int>::max())));
  return ::recv(socket, static_cast<char*>(data), chunk, flags);
#else
  return ::recv(socket, data, bytes, flags);
#endif
}

bool ShutdownSocket(Socket socket, ShutdownHow how) noexcept {
#if defined(_WIN32)
  return ::shutdown(socket, how == ShutdownHow::kWrite ? SD_SEND : SD_BOTH) ==
         0;
#else
  return ::shutdown(socket, how == ShutdownHow::kWrite ? SHUT_WR : SHUT_RDWR) ==
         0;
#endif
}

bool SetBlocking(Socket socket) noexcept {
#if defined(_WIN32)
  u_long mode = 0;
  return ioctlsocket(socket, FIONBIO, &mode) == 0;
#else
  const int flags = ::fcntl(socket, F_GETFL, 0);
  if (flags < 0)
    return false;
  return ::fcntl(socket, F_SETFL, flags & ~O_NONBLOCK) == 0;
#endif
}

bool SetNoDelay(Socket socket) noexcept {
  int value = 1;
  return ::setsockopt(socket, IPPROTO_TCP, TCP_NODELAY,
                      reinterpret_cast<const char*>(&value),
                      sizeof(value)) == 0;
}

bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  const DWORD ms = static_cast<DWORD>(timeout_ms);
  const bool received =
      setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
  const bool sent =
      setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
  return received && sent;
#else
  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  const bool received =
      ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
  const bool sent =
      ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
  return received && sent;
#endif
}

bool SetSendTimeout(Socket socket, int timeout_ms) noexcept {
#if defined(_WIN32)
  DWORD ms = static_cast<DWORD>(timeout_ms);
  return setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO,
                    reinterpret_cast<const char*>(&ms), sizeof(ms)) == 0;
#else
  timeval tv{};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  return ::setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

long long SocketRecvTimeoutMs(Socket socket) noexcept {
#if defined(_WIN32)
  DWORD ms = 0;
  int size = sizeof(ms);
  if (getsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char*>(&ms),
                 &size) != 0)
    return -1;
  return static_cast<long long>(ms);
#else
  timeval tv{};
  socklen_t size = sizeof(tv);
  if (::getsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, &size) != 0)
    return -1;
  return static_cast<long long>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
#endif
}

}  // namespace gufo::net
