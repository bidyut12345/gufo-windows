#include "src/cli/serve/logging.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace gufo::server {
namespace {

std::mutex& LogMutex() {
  static std::mutex mutex;
  return mutex;
}

// Process-wide verbosity threshold. Connection workers, the text scheduler and
// video workers all log through Logger, so the filter has to be shared state
// that is safe to read while another thread is still starting up.
std::atomic<LogLevel>& LevelStorage() {
  static std::atomic<LogLevel> level{LogLevel::kInfo};
  return level;
}

std::string SafeLine(std::string_view input, std::size_t limit = 8192) {
  std::string output;
  for (const unsigned char c : input.substr(0, limit)) {
    if (c < 0x20 || c == 0x7f) {
      constexpr char hex[] = "0123456789abcdef";
      output += "\\x";
      output += hex[c >> 4];
      output += hex[c & 15];
    } else {
      output += static_cast<char>(c);
    }
  }
  if (input.size() > limit)
    output += "...";
  return output;
}

std::string CurrentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm calendar{};
  ::localtime_r(&seconds, &calendar);
  std::array<char, 32> buffer{};
  const auto written = std::strftime(buffer.data(), buffer.size(),
                                     "%Y-%m-%d %H:%M:%S", &calendar);
  return {buffer.data(), written};
}

[[maybe_unused]] std::size_t ReadMemoryKiB(const char* path,
                                           std::string_view field) {
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);) {
    if (line.starts_with(field)) {
      std::istringstream value(line.substr(field.size()));
      std::size_t kib = 0;
      if (value >> kib)
        return kib;
    }
  }
  return 0;
}

}  // namespace

std::optional<LogLevel> LogLevelFromName(std::string_view name) {
  if (name == "debug")
    return LogLevel::kDebug;
  if (name == "info")
    return LogLevel::kInfo;
  if (name == "warn")
    return LogLevel::kWarn;
  if (name == "error")
    return LogLevel::kError;
  return std::nullopt;
}

std::string_view LogLevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "debug";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kError:
      return "error";
  }
  return "info";
}

void Logger::SetLevel(LogLevel level) {
  LevelStorage().store(level, std::memory_order_relaxed);
}

LogLevel Logger::Level() {
  return LevelStorage().load(std::memory_order_relaxed);
}

bool Logger::Enabled(LogLevel level) {
  return level >= Level();
}

void Logger::Log(LogLevel level, std::string_view component,
                 std::string_view message) {
  if (!Enabled(level)) {
    return;
  }
  static const bool color =
      std::getenv("NO_COLOR") == nullptr && ::isatty(STDERR_FILENO) != 0;
  const char* tag = level == LogLevel::kError   ? "ERROR"
                    : level == LogLevel::kWarn  ? "WARN"
                    : level == LogLevel::kDebug ? "DEBUG"
                                                : "INFO";
  const char* tint = level == LogLevel::kError   ? "\033[31m"
                     : level == LogLevel::kWarn  ? "\033[33m"
                     : level == LogLevel::kDebug ? "\033[2;36m"
                                                 : "\033[32m";
  std::ostringstream output;
  output << CurrentTimestamp() << ' ';
  if (color)
    output << tint;
  output << '[' << tag << ']';
  if (color)
    output << "\033[0m";
  output << " [" << SafeLine(component, 64) << "] " << SafeLine(message)
         << '\n';
  const std::lock_guard lock(LogMutex());
  std::clog << output.str() << std::flush;
}

void Logger::LogFormatted(LogLevel level, std::string_view component,
                          const char* format, ...) {
  if (!Enabled(level)) {
    return;
  }
  // A fixed buffer is enough for these process-authored lines and avoids an
  // allocation that a filtered call site would otherwise pay for. vsnprintf
  // reports the untruncated length, so clamp it to what was written.
  char buffer[1024];
  va_list args;
  va_start(args, format);
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  const std::size_t length =
      written < 0 ? 0
                  : std::min<std::size_t>(static_cast<std::size_t>(written),
                                          sizeof(buffer) - 1);
  Log(level, component, std::string_view(buffer, length));
}

std::string Logger::MemoryStatus() {
  std::ostringstream output;
#ifdef _WIN32
  output << "rss_mib=" << gufo_resident_bytes() / (1024 * 1024)
         << " host_available_mib="
         << gufo_available_physical_bytes() / (1024 * 1024);
#else
  output << "rss_mib=" << ReadMemoryKiB("/proc/self/status", "VmRSS:") / 1024
         << " host_available_mib="
         << ReadMemoryKiB("/proc/meminfo", "MemAvailable:") / 1024;
#endif
  return output.str();
}

namespace {

std::size_t VisibleLength(std::string_view s) {
  std::size_t len = 0;
  bool in_escape = false;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\033') {
      in_escape = true;
    } else if (in_escape) {
      if (s[i] == 'm') {
        in_escape = false;
      }
    } else {
      const unsigned char c = static_cast<unsigned char>(s[i]);
      if ((c & 0xc0) != 0x80) {
        ++len;
      }
    }
  }
  return len;
}

std::string CommaNumber(std::string_view s) {
  if (s.empty()) return "0";
  bool all_digits = true;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      all_digits = false;
      break;
    }
  }
  if (!all_digits) return std::string(s);
  std::string res;
  res.reserve(s.size() + s.size() / 3);
  int count = 0;
  for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
    if (count > 0 && count % 3 == 0) {
      res.push_back(',');
    }
    res.push_back(s[i]);
    ++count;
  }
  std::reverse(res.begin(), res.end());
  return res;
}

std::string FriendlyBytes(std::string_view s) {
  if (s.empty()) return "0 B";
  try {
    double bytes = std::stod(std::string(s));
    char buf[64];
    if (bytes >= 1024.0 * 1024.0 * 1024.0) {
      std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
      return buf;
    }
    if (bytes >= 1024.0 * 1024.0) {
      std::snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024.0));
      return buf;
    }
    if (bytes >= 1024.0) {
      std::snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
      return buf;
    }
    std::snprintf(buf, sizeof(buf), "%.0f B", bytes);
    return buf;
  } catch (...) {
    return std::string(s);
  }
}

std::string ExtractField(std::string_view text, std::string_view key) {
  std::string needle = std::string(key) + "=";
  std::size_t pos = text.find(needle);
  if (pos == std::string_view::npos) return "";
  pos += needle.size();
  std::size_t end = text.find(' ', pos);
  if (end == std::string_view::npos) end = text.size();
  return std::string(text.substr(pos, end - pos));
}

std::string FormatRequestTable(
    std::string_view id, std::string_view method, std::string_view path,
    int status_code, double duration_ms, std::string_view outcome,
    std::string_view details, std::string_view mem, bool color) {
  std::string all_details = std::string(details) + " " + std::string(mem);

  std::string prompt_tokens = ExtractField(all_details, "prompt_tokens");
  std::string prefill_tokens = ExtractField(all_details, "prefill_tokens");
  std::string generated_tokens = ExtractField(all_details, "generated_tokens");
  std::string finish = ExtractField(all_details, "finish");
  std::string cache = ExtractField(all_details, "cache");
  std::string cached_tokens = ExtractField(all_details, "cached_tokens");
  std::string cache_restore_ms = ExtractField(all_details, "cache_restore_ms");
  std::string queue_depth = ExtractField(all_details, "queue_depth");
  std::string resident_at_admission = ExtractField(all_details, "resident_at_admission");
  std::string queue_ms = ExtractField(all_details, "queue_ms");
  std::string shared_prefix_wait_ms = ExtractField(all_details, "shared_prefix_wait_ms");
  std::string ttft_ms = ExtractField(all_details, "ttft_ms");
  std::string prefill_tps = ExtractField(all_details, "prefill_tps");
  std::string decode_tps = ExtractField(all_details, "decode_tps");
  std::string batch_width = ExtractField(all_details, "batch_width");
  std::string plan = ExtractField(all_details, "plan");
  std::string draft_accepted = ExtractField(all_details, "draft_accepted");
  std::string draft_proposed = ExtractField(all_details, "draft_proposed");
  std::string draft_rounds = ExtractField(all_details, "draft_rounds");
  std::string acceptance_pct = ExtractField(all_details, "acceptance_pct");
  std::string lookup_accepted = ExtractField(all_details, "lookup_accepted");
  std::string lookup_proposed = ExtractField(all_details, "lookup_proposed");
  std::string cache_reuse = ExtractField(all_details, "cache_reuse");
  std::string cache_captures = ExtractField(all_details, "cache_captures");
  std::string cache_captured_capacity_bytes = ExtractField(all_details, "cache_captured_capacity_bytes");
  std::string cache_global_ledger_bytes = ExtractField(all_details, "cache_global_ledger_bytes");
  std::string cache_snapshot_bytes = ExtractField(all_details, "cache_snapshot_bytes");
  std::string rss_mib = ExtractField(all_details, "rss_mib");
  std::string host_available_mib = ExtractField(all_details, "host_available_mib");

  constexpr int kWidth = 84;
  std::string hline;
  for (int i = 0; i < kWidth - 2; ++i) hline += "-";

  auto make_line = [&](std::string_view left, std::string_view right = "") -> std::string {
    std::string l(left);
    std::string r(right);
    const int total_chars = static_cast<int>(VisibleLength(l) + VisibleLength(r));
    int space_needed = (kWidth - 4) - total_chars;
    if (space_needed < 1) space_needed = 1;
    std::string line = "| ";
    line += l;
    line.append(space_needed, ' ');
    line += r;
    line += " |";
    return line;
  };

  auto make_section = [&](std::string_view title) -> std::string {
    std::string s = "+- " + std::string(title) + " ";
    int fill = (kWidth - 1) - static_cast<int>(title.size() + 4);
    if (fill < 1) fill = 1;
    for (int i = 0; i < fill; ++i) s += "-";
    s += "+";
    return s;
  };

  const char* c_reset = color ? "\033[0m" : "";
  const char* c_green = color ? "\033[1;32m" : "";
  const char* c_yellow = color ? "\033[1;33m" : "";
  const char* c_red = color ? "\033[1;31m" : "";

  std::ostringstream out;
  out << "+" << hline << "+\n";

  // Header Row
  char title_left[128];
  std::snprintf(title_left, sizeof(title_left), "GUFO REQUEST: %.*s (%s%d %.*s%s)",
                static_cast<int>(id.size()), id.data(),
                (status_code < 400 ? c_green : (status_code < 500 ? c_yellow : c_red)),
                status_code,
                static_cast<int>(outcome.size()), outcome.data(),
                c_reset);
  char title_right[128];
  std::snprintf(title_right, sizeof(title_right), "Duration: %.1f ms", duration_ms);
  out << make_line(title_left, title_right) << '\n';

  char route_left[128];
  std::snprintf(route_left, sizeof(route_left), "Endpoint: %.*s %.*s",
                static_cast<int>(method.size()), method.data(),
                static_cast<int>(path.size()), path.data());
  char route_right[128];
  std::snprintf(route_right, sizeof(route_right), "Plan: %s (width %s)",
                plan.empty() ? "default" : plan.c_str(),
                batch_width.empty() ? "1" : batch_width.c_str());
  out << make_line(route_left, route_right) << '\n';

  // Section: Token & Throughput
  out << make_section("Token Metrics & Speed") << '\n';
  char tok1_l[128], tok1_r[128];
  std::snprintf(tok1_l, sizeof(tok1_l), "Prompt Tokens:    %s", CommaNumber(prompt_tokens).c_str());
  std::snprintf(tok1_r, sizeof(tok1_r), "Prefill Speed:  %s tok/s", prefill_tps.empty() ? "0.0" : prefill_tps.c_str());
  out << make_line(tok1_l, tok1_r) << '\n';

  char tok2_l[128], tok2_r[128];
  std::snprintf(tok2_l, sizeof(tok2_l), "Prefill Tokens:   %s", CommaNumber(prefill_tokens).c_str());
  std::snprintf(tok2_r, sizeof(tok2_r), "Decode Speed:   %s tok/s", decode_tps.empty() ? "0.0" : decode_tps.c_str());
  out << make_line(tok2_l, tok2_r) << '\n';

  char tok3_l[128], tok3_r[128];
  std::snprintf(tok3_l, sizeof(tok3_l), "Generated Tokens: %s (finish: %s)",
                CommaNumber(generated_tokens).c_str(), finish.empty() ? "stop" : finish.c_str());
  std::snprintf(tok3_r, sizeof(tok3_r), "Cached Tokens:  %s (%s)",
                CommaNumber(cached_tokens).c_str(), cache.empty() ? "none" : cache.c_str());
  out << make_line(tok3_l, tok3_r) << '\n';

  // Section: Latency & Scheduling
  out << make_section("Latency & Scheduling") << '\n';
  char lat1_l[128], lat1_r[128];
  std::snprintf(lat1_l, sizeof(lat1_l), "Time To First Token: %s ms", ttft_ms.empty() ? "0.0" : ttft_ms.c_str());
  std::snprintf(lat1_r, sizeof(lat1_r), "Queue Wait:     %s ms (depth %s)",
                queue_ms.empty() ? "0.0" : queue_ms.c_str(),
                queue_depth.empty() ? "0" : queue_depth.c_str());
  out << make_line(lat1_l, lat1_r) << '\n';

  char lat2_l[128], lat2_r[128];
  std::snprintf(lat2_l, sizeof(lat2_l), "KV Cache Restore:    %s ms", cache_restore_ms.empty() ? "0.0" : cache_restore_ms.c_str());
  std::snprintf(lat2_r, sizeof(lat2_r), "Shared Prefix:  %s ms (resident %s)",
                shared_prefix_wait_ms.empty() ? "0.0" : shared_prefix_wait_ms.c_str(),
                resident_at_admission.empty() ? "0" : resident_at_admission.c_str());
  out << make_line(lat2_l, lat2_r) << '\n';

  // Section: Speculative & Lookup (if present)
  if (!draft_proposed.empty() || !lookup_proposed.empty()) {
    out << make_section("Speculative Decoding & Lookup") << '\n';
    if (!draft_proposed.empty()) {
      char draft_l[128], draft_r[128];
      std::snprintf(draft_l, sizeof(draft_l), "Draft Tokens:     %s/%s accepted (%s%%)",
                    draft_accepted.c_str(), draft_proposed.c_str(), acceptance_pct.c_str());
      std::snprintf(draft_r, sizeof(draft_r), "Draft Rounds:   %s", draft_rounds.c_str());
      out << make_line(draft_l, draft_r) << '\n';
    }
    if (!lookup_proposed.empty()) {
      char look_l[128];
      std::snprintf(look_l, sizeof(look_l), "Prompt Lookup:    %s/%s accepted",
                    lookup_accepted.c_str(), lookup_proposed.c_str());
      out << make_line(look_l) << '\n';
    }
  }

  // Section: Memory & Cache Ledger
  out << make_section("Memory & Cache Ledger") << '\n';
  char mem1_l[128], mem1_r[128];
  std::snprintf(mem1_l, sizeof(mem1_l), "Cache Ledger:     %s (%s)",
                FriendlyBytes(cache_global_ledger_bytes).c_str(),
                cache_reuse.empty() ? "ram" : cache_reuse.c_str());
  std::snprintf(mem1_r, sizeof(mem1_r), "Captured Cap:   %s (%s caps)",
                FriendlyBytes(cache_captured_capacity_bytes).c_str(),
                cache_captures.empty() ? "0" : cache_captures.c_str());
  out << make_line(mem1_l, mem1_r) << '\n';

  char mem2_l[128], mem2_r[128];
  std::snprintf(mem2_l, sizeof(mem2_l), "Cache Snapshot:   %s",
                FriendlyBytes(cache_snapshot_bytes).c_str());
  std::snprintf(mem2_r, sizeof(mem2_r), "Process RSS:    %s MB (free: %s MB)",
                CommaNumber(rss_mib).c_str(),
                CommaNumber(host_available_mib).c_str());
  out << make_line(mem2_l, mem2_r) << '\n';

  out << "+" << hline << "+";
  return out.str();
}

}  // namespace

void Logger::LogRequest(std::string_view id, std::string_view method,
                        std::string_view path, int status_code,
                        double duration_ms, std::string_view details,
                        std::string_view outcome, LogLevel success_level) {
  // Escalation stays upward-only: a debug-tier poll that failed still logs at
  // WARN/ERROR under the default threshold.
  const LogLevel level =
      status_code >= 500 || outcome == "stream_error"   ? LogLevel::kError
      : status_code >= 400 || outcome == "disconnected" ? LogLevel::kWarn
                                                        : success_level;
  // A filtered line must not pay for MemoryStatus(): two /proc reads per
  // request add up on every quiet poll the tier suppresses.
  if (!Enabled(level)) {
    return;
  }
  const std::string mem = MemoryStatus();
  std::ostringstream message;
  message << "request=" << id << " event=completed method=" << method
          << " path=" << path << " status=" << status_code
          << " duration_ms=" << std::fixed << std::setprecision(1)
          << duration_ms << " outcome=" << outcome;
  if (!details.empty())
    message << ' ' << details;
  message << ' ' << mem;
  Log(level, "http", message.str());

  if (details.find("prompt_tokens=") != std::string_view::npos ||
      details.find("generated_tokens=") != std::string_view::npos) {
    static const bool color =
        std::getenv("NO_COLOR") == nullptr && ::isatty(STDERR_FILENO) != 0;
    const std::string table = FormatRequestTable(
        id, method, path, status_code, duration_ms, outcome, details, mem, color);
    const std::lock_guard lock(LogMutex());
    std::clog << table << '\n' << std::flush;
  }
}

}  // namespace gufo::server
