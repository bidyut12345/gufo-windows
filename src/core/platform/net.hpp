#ifndef GUFO_CORE_PLATFORM_NET_HPP_
#define GUFO_CORE_PLATFORM_NET_HPP_

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
// clang-format on
#ifndef SHUT_RD
#define SHUT_RD SD_RECEIVE
#endif
#ifndef SHUT_WR
#define SHUT_WR SD_SEND
#endif
#ifndef SHUT_RDWR
#define SHUT_RDWR SD_BOTH
#endif
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace gufo::net {

#if defined(_WIN32)
using Socket = SOCKET;
#else
using Socket = int;
#endif

/// The one-time WinSock startup the HTTP server performs before opening its
/// listener; a no-op on POSIX. Throws std::system_error on failure.
void EnsureSocketRuntimeStarted();

/// Close a socket: closesocket on Windows, close on POSIX.
void CloseSocket(Socket socket) noexcept;

/// poll()/WSAPoll() with the platform's native timeout unit.
int PollSocket(void* descriptors, std::uint32_t count, int timeout_ms);

/// Send that never raises SIGPIPE (MSG_NOSIGNAL on POSIX; default on Winsock).
std::intptr_t SendNoSignal(Socket socket, const void* data, std::size_t bytes);

/// Mark a socket non-blocking; returns false on failure.
bool SetNonBlocking(Socket socket) noexcept;

/// Mark a socket blocking; returns false on failure. Required for sockets
/// accepted from a non-blocking listener on Windows, where they inherit the
/// listener's mode (POSIX accepted sockets never inherit O_NONBLOCK).
bool SetBlocking(Socket socket) noexcept;

/// Disable Nagle on a connected stream socket; returns false on failure.
bool SetNoDelay(Socket socket) noexcept;

/// Set receive/send timeouts in milliseconds; returns false on failure.
bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept;

/// Set only the send timeout in milliseconds; the receive timeout stays at
/// the caller's idle limit. Returns false on failure.
bool SetSendTimeout(Socket socket, int timeout_ms) noexcept;

/// Read the receive timeout back in milliseconds; 0 means no timeout is
/// set, -1 means the query failed.
long long SocketRecvTimeoutMs(Socket socket) noexcept;

}  // namespace gufo::net

#endif  // GUFO_CORE_PLATFORM_NET_HPP_
