#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/media.hpp"

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif

#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

// Windows has no pipe:N descriptors beyond stdin, so FFmpeg reads both raw
// streams from overlapped named pipes. Every wait polls cancellation, the
// shared stop flag and the child's exit, so a dead or cancelled FFmpeg never
// leaves a writer blocked.

namespace gufo::minimax_h3 {
namespace {

constexpr std::size_t kIoChunkBytes = 4U << 20U;
constexpr int kAudioChunkSamples = 4096;
constexpr DWORD kPollMilliseconds = 50;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string WindowsMessage(DWORD code) {
  return std::system_category().message(static_cast<int>(code));
}

const char* FfmpegProgram() {
  const char* override = std::getenv("GUFO_FFMPEG");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFMPEG_EXECUTABLE;
}

const char* FfprobeProgram() {
  const char* override = std::getenv("GUFO_FFPROBE");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFPROBE_EXECUTABLE;
}

std::wstring Widen(std::string_view text) {
  if (text.empty()) {
    return {};
  }
  const int length = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(std::max(length, 0)), L'\0');
  if (length > 0) {
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        wide.data(), length);
  }
  return wide;
}

class Handle {
public:
  Handle() = default;
  explicit Handle(HANDLE handle)
      : handle_(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
  ~Handle() { Reset(); }
  Handle(Handle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      Reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  [[nodiscard]] HANDLE get() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }
  void Reset() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
      handle_ = nullptr;
    }
  }

private:
  HANDLE handle_{nullptr};
};

std::filesystem::path PartialMediaPath(
    const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> sequence{0};
  std::wstring name =
      destination.filename().wstring() + L".gufo-partial-" +
      std::to_wstring(GetCurrentProcessId()) + L"-" +
      std::to_wstring(sequence.fetch_add(1, std::memory_order_relaxed)) +
      L".mp4";
  return destination.parent_path() / name;
}

std::wstring PipeName(std::wstring_view stream) {
  static std::atomic<std::uint64_t> sequence{0};
  return L"\\\\.\\pipe\\gufo-h3-" + std::to_wstring(GetCurrentProcessId()) +
         L"-" +
         std::to_wstring(sequence.fetch_add(1, std::memory_order_relaxed)) +
         L"-" + std::wstring(stream);
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

// CommandLineToArgvW quoting, so FFmpeg sees each argument unchanged.
void AppendArgument(std::wstring* line, std::wstring_view argument) {
  if (!line->empty()) {
    line->push_back(L' ');
  }
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring_view::npos) {
    line->append(argument);
    return;
  }
  line->push_back(L'"');
  for (auto it = argument.begin();; ++it) {
    std::size_t backslashes = 0;
    while (it != argument.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == argument.end()) {
      line->append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      line->append(backslashes * 2 + 1, L'\\');
    } else {
      line->append(backslashes, L'\\');
    }
    line->push_back(*it);
  }
  line->push_back(L'"');
}

Handle InheritableNul(DWORD access) {
  SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
  return Handle(CreateFileW(L"NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &attributes, OPEN_EXISTING, 0, nullptr));
}

// The parent's standard handle as an inheritable copy, or NUL without one.
Handle InheritableStandard(DWORD which) {
  const HANDLE source = GetStdHandle(which);
  HANDLE copy = nullptr;
  if (source != nullptr && source != INVALID_HANDLE_VALUE &&
      DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &copy,
                      0, TRUE, DUPLICATE_SAME_ACCESS)) {
    return Handle(copy);
  }
  return InheritableNul(GENERIC_WRITE);
}

// Starts `program` with `arguments` (argv[1..]); errors name it `label`. stdin
// is NUL; stdout is `output` when given, else the parent's. Only these three
// handles are inherited.
bool Spawn(const char* program, std::string_view label,
           const std::vector<std::wstring>& arguments, HANDLE output,
           Handle* process, std::string* error) {
  std::wstring line;
  AppendArgument(&line, Widen(program));
  for (const std::wstring& argument : arguments) {
    AppendArgument(&line, argument);
  }
  Handle input = InheritableNul(GENERIC_READ);
  Handle parent_output;
  if (output == nullptr) {
    parent_output = InheritableStandard(STD_OUTPUT_HANDLE);
    output = parent_output.get();
  }
  Handle errors = InheritableStandard(STD_ERROR_HANDLE);
  if (!input || output == nullptr || !errors) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  std::array<HANDLE, 3> inherited = {input.get(), output, errors.get()};
  SIZE_T attribute_bytes = 0;
  (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
  std::vector<unsigned char> attribute_storage(attribute_bytes);
  auto* attributes =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.data());
  if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes)) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.get();
  startup.StartupInfo.hStdOutput = output;
  startup.StartupInfo.hStdError = errors.get();
  startup.lpAttributeList = attributes;
  PROCESS_INFORMATION info{};
  const bool created =
      UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
          sizeof(inherited), nullptr, nullptr) &&
      CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                     EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                     nullptr, &startup.StartupInfo, &info);
  const DWORD failure = created ? ERROR_SUCCESS : GetLastError();
  DeleteProcThreadAttributeList(attributes);
  if (!created) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(failure));
    return false;
  }
  CloseHandle(info.hThread);
  *process = Handle(info.hProcess);
  return true;
}

bool WaitForChild(HANDLE child, int* status, std::string* error) {
  DWORD code = 0;
  if (WaitForSingleObject(child, INFINITE) != WAIT_OBJECT_0 ||
      !GetExitCodeProcess(child, &code)) {
    SetError(error, "cannot wait for media process: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  *status = static_cast<int>(code);
  return true;
}

Handle CreateOutboundPipe(const std::wstring& name) {
  return Handle(
      CreateNamedPipeW(name.c_str(),
                       PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED |
                           FILE_FLAG_FIRST_PIPE_INSTANCE,
                       PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                       1, static_cast<DWORD>(kIoChunkBytes), 0, 0, nullptr));
}

// Completes an overlapped pipe operation, or cancels it when the stream is
// stopped, the job is cancelled or FFmpeg has exited.
bool AwaitOverlapped(HANDLE pipe, OVERLAPPED* overlapped, HANDLE child,
                     const CancellationToken* cancellation,
                     const std::atomic<bool>& stop, DWORD* transferred,
                     DWORD* failure) {
  const std::array<HANDLE, 2> waits = {overlapped->hEvent, child};
  while (true) {
    const DWORD result =
        WaitForMultipleObjects(static_cast<DWORD>(waits.size()), waits.data(),
                               FALSE, kPollMilliseconds);
    if (result == WAIT_OBJECT_0) {
      if (GetOverlappedResult(pipe, overlapped, transferred, FALSE)) {
        return true;
      }
      *failure = GetLastError();
      return false;
    }
    const DWORD wait_failure =
        result == WAIT_FAILED ? GetLastError() : ERROR_SUCCESS;
    const bool child_exited = result == WAIT_OBJECT_0 + 1;
    if (child_exited || result == WAIT_FAILED ||
        stop.load(std::memory_order_acquire) || IsCancelled(cancellation)) {
      CancelIoEx(pipe, overlapped);
      if (GetOverlappedResult(pipe, overlapped, transferred, TRUE)) {
        return true;
      }
      *failure = child_exited            ? ERROR_BROKEN_PIPE
                 : result == WAIT_FAILED ? wait_failure
                                         : ERROR_OPERATION_ABORTED;
      return false;
    }
  }
}

bool ConnectPipe(HANDLE pipe, HANDLE child,
                 const CancellationToken* cancellation,
                 const std::atomic<bool>& stop, DWORD* failure) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) {
    *failure = GetLastError();
    return false;
  }
  OVERLAPPED overlapped{};
  overlapped.hEvent = event.get();
  if (ConnectNamedPipe(pipe, &overlapped)) {
    return true;
  }
  const DWORD code = GetLastError();
  if (code == ERROR_PIPE_CONNECTED) {
    return true;
  }
  if (code != ERROR_IO_PENDING) {
    *failure = code;
    return false;
  }
  DWORD ignored = 0;
  return AwaitOverlapped(pipe, &overlapped, child, cancellation, stop, &ignored,
                         failure);
}

bool WriteAll(HANDLE pipe, const std::uint8_t* data, std::size_t bytes,
              HANDLE child, const CancellationToken* cancellation,
              const std::atomic<bool>& stop, std::uint64_t* written,
              DWORD* failure) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) {
    *failure = GetLastError();
    return false;
  }
  while (bytes != 0) {
    if (stop.load(std::memory_order_acquire) || IsCancelled(cancellation)) {
      *failure = ERROR_OPERATION_ABORTED;
      return false;
    }
    OVERLAPPED overlapped{};
    overlapped.hEvent = event.get();
    ResetEvent(event.get());
    const auto request =
        static_cast<DWORD>(std::min<std::size_t>(bytes, kIoChunkBytes));
    if (!WriteFile(pipe, data, request, nullptr, &overlapped)) {
      const DWORD code = GetLastError();
      if (code != ERROR_IO_PENDING) {
        *failure = code;
        return false;
      }
    }
    DWORD amount = 0;
    if (!AwaitOverlapped(pipe, &overlapped, child, cancellation, stop, &amount,
                         failure)) {
      return false;
    }
    if (amount == 0) {
      *failure = ERROR_BROKEN_PIPE;
      return false;
    }
    data += amount;
    bytes -= amount;
    *written += amount;
  }
  return true;
}

bool ParseInteger(std::string_view text, int* output) {
  if (text.empty()) {
    return false;
  }
  int value = 0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseDouble(std::string_view text, double* output) {
  if (text.empty() || text == "N/A") {
    *output = 0.0;
    return true;
  }
  double value = 0.0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(value)) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseRate(std::string_view text, double* output) {
  const std::size_t slash = text.find('/');
  if (slash == std::string_view::npos) {
    return ParseDouble(text, output);
  }
  double numerator = 0.0;
  double denominator = 0.0;
  if (!ParseDouble(text.substr(0, slash), &numerator) ||
      !ParseDouble(text.substr(slash + 1), &denominator) ||
      denominator == 0.0) {
    return false;
  }
  *output = numerator / denominator;
  return std::isfinite(*output);
}

bool RunCapture(const char* program, const std::vector<std::wstring>& arguments,
                std::string* output, std::string* error) {
  SECURITY_ATTRIBUTES inheritable{sizeof(inheritable), nullptr, TRUE};
  HANDLE read_end = nullptr;
  HANDLE write_end = nullptr;
  if (!CreatePipe(&read_end, &write_end, &inheritable, 0)) {
    SetError(error, "cannot create media inspection pipe: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  Handle reader(read_end);
  Handle writer(write_end);
  SetHandleInformation(reader.get(), HANDLE_FLAG_INHERIT, 0);
  Handle child;
  const bool started =
      Spawn(program, program, arguments, writer.get(), &child, error);
  writer.Reset();
  if (!started) {
    return false;
  }
  output->clear();
  std::array<char, 4096> buffer{};
  while (true) {
    DWORD amount = 0;
    if (!ReadFile(reader.get(), buffer.data(),
                  static_cast<DWORD>(buffer.size()), &amount, nullptr) ||
        amount == 0) {
      break;
    }
    if (output->size() + amount > (1U << 20U)) {
      reader.Reset();
      TerminateProcess(child.get(), 1);
      int ignored = 0;
      (void)WaitForChild(child.get(), &ignored, nullptr);
      SetError(error, "media inspection output exceeded 1 MiB");
      return false;
    }
    output->append(buffer.data(), amount);
  }
  reader.Reset();
  int status = 0;
  if (!WaitForChild(child.get(), &status, error)) {
    return false;
  }
  if (status != 0) {
    SetError(error, std::string(program) + " exited with status " +
                        std::to_string(status));
    return false;
  }
  return true;
}

using KeyValues = std::vector<std::pair<std::string, std::string>>;

KeyValues ParseKeyValues(std::string_view text) {
  KeyValues result;
  while (!text.empty()) {
    const std::size_t newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    const std::size_t equals = line.find('=');
    if (equals != std::string_view::npos) {
      result.emplace_back(line.substr(0, equals), line.substr(equals + 1));
    }
    if (newline == std::string_view::npos) {
      break;
    }
    text.remove_prefix(newline + 1);
  }
  return result;
}

const std::string* FindValue(const KeyValues& values, std::string_view key) {
  for (const auto& [candidate, value] : values) {
    if (candidate == key) {
      return &value;
    }
  }
  return nullptr;
}

}  // namespace

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          const MediaVideoOptions& video,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  if (path.empty() || frame_count < 1 || video.input_width < 2 ||
      video.input_height < 2 || video.output_width < 2 ||
      video.output_height < 2 || (video.input_width & 1) != 0 ||
      (video.input_height & 1) != 0 || (video.output_width & 1) != 0 ||
      (video.output_height & 1) != 0 || video.fps < 1 || samples < 1 ||
      channels < 1 || sample_rate < 1) {
    SetError(error, "invalid MiniMax H3 media mux arguments");
    return false;
  }
  const std::uint64_t frame_bytes =
      static_cast<std::uint64_t>(video.input_width) * video.input_height * 3U;
  const std::uint64_t video_bytes =
      frame_bytes * static_cast<std::uint64_t>(frame_count);
  const std::uint64_t audio_elements =
      static_cast<std::uint64_t>(samples) * channels;
  if (video_bytes != rgb24.size() ||
      audio_elements != channel_major_pcm.size() ||
      video_bytes > std::numeric_limits<std::size_t>::max() ||
      audio_elements >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    SetError(error, "MiniMax H3 media stream geometry differs from buffers");
    return false;
  }
  if (IsCancelled(cancellation)) {
    if (telemetry != nullptr) {
      telemetry->cancelled = true;
    }
    SetError(error, "MiniMax H3 media mux cancelled before start");
    return false;
  }
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create media output directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial_path = PartialMediaPath(path);
  std::filesystem::remove(partial_path, filesystem_error);
  filesystem_error.clear();

  const std::wstring video_name = PipeName(L"video");
  const std::wstring audio_name = PipeName(L"audio");
  Handle video_pipe = CreateOutboundPipe(video_name);
  Handle audio_pipe = CreateOutboundPipe(audio_name);
  if (!video_pipe || !audio_pipe) {
    SetError(error, "cannot create FFmpeg media pipes: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  const std::wstring size = std::to_wstring(video.input_width) + L"x" +
                            std::to_wstring(video.input_height);
  const std::wstring output_size = std::to_wstring(video.output_width) + L":" +
                                   std::to_wstring(video.output_height);
  const std::vector<std::wstring> arguments = {
      L"-y",
      L"-nostdin",
      L"-loglevel",
      L"error",
      L"-f",
      L"rawvideo",
      L"-pixel_format",
      L"rgb24",
      L"-video_size",
      size,
      L"-framerate",
      std::to_wstring(video.fps),
      L"-i",
      video_name,
      L"-f",
      L"f32le",
      L"-ar",
      std::to_wstring(sample_rate),
      L"-ac",
      std::to_wstring(channels),
      L"-i",
      audio_name,
      L"-map",
      L"0:v:0",
      L"-map",
      L"1:a:0",
      L"-vf",
      L"scale=" + output_size + L":flags=lanczos,setsar=1",
      L"-c:v",
      L"libx264",
      L"-preset",
      L"fast",
      L"-crf",
      L"18",
      L"-pix_fmt",
      L"yuv420p",
      L"-c:a",
      L"aac",
      L"-b:a",
      L"192k",
      L"-movflags",
      L"+faststart",
      partial_path.wstring(),
  };
  Handle child;
  if (!Spawn(FfmpegProgram(), "FFmpeg", arguments, nullptr, &child, error)) {
    std::filesystem::remove(partial_path, filesystem_error);
    return false;
  }

  std::atomic<bool> stop{false};
  std::atomic<DWORD> video_error{ERROR_SUCCESS};
  std::atomic<DWORD> audio_error{ERROR_SUCCESS};
  std::uint64_t written_video = 0;
  std::uint64_t written_audio = 0;
  std::size_t maximum_audio_staging = 0;

  // A writer error counts only when nothing else stopped the mux first.
  const auto record = [&](std::atomic<DWORD>* slot, DWORD failure) {
    if (!IsCancelled(cancellation) && !stop.load(std::memory_order_acquire)) {
      slot->store(failure, std::memory_order_release);
    }
    stop.store(true, std::memory_order_release);
  };
  std::thread video_thread([&] {
    DWORD failure = ERROR_SUCCESS;
    if (!ConnectPipe(video_pipe.get(), child.get(), cancellation, stop,
                     &failure) ||
        !WriteAll(video_pipe.get(), rgb24.data(), rgb24.size(), child.get(),
                  cancellation, stop, &written_video, &failure)) {
      record(&video_error, failure);
    }
    video_pipe.Reset();
  });
  std::thread audio_thread([&] {
    std::vector<float> interleaved(
        static_cast<std::size_t>(kAudioChunkSamples) * channels);
    maximum_audio_staging = interleaved.size() * sizeof(float);
    DWORD failure = ERROR_SUCCESS;
    if (!ConnectPipe(audio_pipe.get(), child.get(), cancellation, stop,
                     &failure)) {
      record(&audio_error, failure);
      audio_pipe.Reset();
      return;
    }
    int offset = 0;
    while (offset < samples && !stop.load(std::memory_order_acquire)) {
      const int chunk = std::min(kAudioChunkSamples, samples - offset);
      for (int sample = 0; sample < chunk; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
          interleaved[static_cast<std::size_t>(sample) * channels + channel] =
              channel_major_pcm[static_cast<std::size_t>(channel) * samples +
                                offset + sample];
        }
      }
      if (!WriteAll(audio_pipe.get(),
                    reinterpret_cast<const std::uint8_t*>(interleaved.data()),
                    static_cast<std::size_t>(chunk) * channels * sizeof(float),
                    child.get(), cancellation, stop, &written_audio,
                    &failure)) {
        record(&audio_error, failure);
        break;
      }
      offset += chunk;
    }
    audio_pipe.Reset();
  });
  video_thread.join();
  audio_thread.join();

  const bool cancelled = IsCancelled(cancellation);
  if (stop.load(std::memory_order_acquire) || cancelled) {
    TerminateProcess(child.get(), 1);
  }
  int status = 0;
  const bool waited = WaitForChild(child.get(), &status, error);
  if (telemetry != nullptr) {
    telemetry->video_bytes = written_video;
    telemetry->audio_bytes = written_audio;
    telemetry->maximum_audio_staging_bytes = maximum_audio_staging;
    telemetry->child_status = waited ? status : -1;
    telemetry->cancelled = cancelled;
  }
  if (!waited) {
    std::filesystem::remove(partial_path, filesystem_error);
    return false;
  }
  if (cancelled) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "MiniMax H3 media mux cancelled");
    return false;
  }
  const DWORD writer_error =
      video_error.load(std::memory_order_acquire) != ERROR_SUCCESS
          ? video_error.load(std::memory_order_relaxed)
          : audio_error.load(std::memory_order_acquire);
  if (writer_error != ERROR_SUCCESS) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot stream generated media to FFmpeg: " +
                        WindowsMessage(writer_error));
    return false;
  }
  if (status != 0) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "FFmpeg exited with status " + std::to_string(status));
    return false;
  }
  if (written_video != video_bytes ||
      written_audio != audio_elements * sizeof(float)) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "FFmpeg media streams ended before all bytes were sent");
    return false;
  }
  MediaProbe probe;
  const double expected_video_duration =
      static_cast<double>(frame_count) / video.fps;
  const double expected_audio_duration =
      static_cast<double>(samples) / sample_rate;
  if (!ProbeMediaFile(partial_path, &probe, error) ||
      probe.video_codec != "h264" || probe.audio_codec != "aac" ||
      probe.width != video.output_width ||
      probe.height != video.output_height ||
      std::abs(probe.video_frame_rate - video.fps) > 1.0e-6 ||
      probe.audio_channels != channels ||
      probe.audio_sample_rate != sample_rate ||
      std::abs(probe.video_start_seconds - probe.audio_start_seconds) >
          1.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - expected_video_duration) >
          1.0 / video.fps ||
      std::abs(probe.audio_duration_seconds - expected_audio_duration) >
          2048.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - probe.audio_duration_seconds) >
          0.075) {
    std::filesystem::remove(partial_path, filesystem_error);
    if (error != nullptr && error->empty()) {
      *error = "generated MiniMax H3 media violates the delivery contract";
    }
    return false;
  }
  std::filesystem::rename(partial_path, path, filesystem_error);
  if (filesystem_error) {
    const std::string message = filesystem_error.message();
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot publish generated media: " + message);
    return false;
  }
  return true;
}

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          int width, int height, int fps,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  return WriteSynchronizedMp4(path, rgb24, frame_count,
                              {.input_width = width,
                               .input_height = height,
                               .output_width = width,
                               .output_height = height,
                               .fps = fps},
                              channel_major_pcm, samples, channels, sample_rate,
                              cancellation, telemetry, error);
}

bool ProbeMediaFile(const std::filesystem::path& path, MediaProbe* probe,
                    std::string* error) {
  if (path.empty() || probe == nullptr) {
    SetError(error, "invalid MiniMax H3 media probe arguments");
    return false;
  }
  *probe = {};
  std::string video_output;
  if (!RunCapture(
          FfprobeProgram(),
          {L"-v", L"error", L"-select_streams", L"v:0", L"-show_entries",
           L"stream=codec_name,width,height,r_frame_rate,start_time,"
           L"duration",
           L"-of", L"default=noprint_wrappers=1", path.wstring()},
          &video_output, error)) {
    return false;
  }
  const KeyValues video = ParseKeyValues(video_output);
  const std::string* video_codec = FindValue(video, "codec_name");
  const std::string* width = FindValue(video, "width");
  const std::string* height = FindValue(video, "height");
  const std::string* frame_rate = FindValue(video, "r_frame_rate");
  const std::string* video_start = FindValue(video, "start_time");
  const std::string* video_duration = FindValue(video, "duration");
  if (video_codec == nullptr || width == nullptr || height == nullptr ||
      frame_rate == nullptr || video_start == nullptr ||
      video_duration == nullptr || !ParseInteger(*width, &probe->width) ||
      !ParseInteger(*height, &probe->height) ||
      !ParseRate(*frame_rate, &probe->video_frame_rate) ||
      !ParseDouble(*video_start, &probe->video_start_seconds) ||
      !ParseDouble(*video_duration, &probe->video_duration_seconds)) {
    SetError(error, "FFprobe returned malformed MiniMax H3 video metadata");
    return false;
  }
  probe->video_codec = *video_codec;

  std::string audio_output;
  if (!RunCapture(
          FfprobeProgram(),
          {L"-v", L"error", L"-select_streams", L"a:0", L"-show_entries",
           L"stream=codec_name,channels,sample_rate,start_time,"
           L"duration",
           L"-of", L"default=noprint_wrappers=1", path.wstring()},
          &audio_output, error)) {
    return false;
  }
  const KeyValues audio = ParseKeyValues(audio_output);
  const std::string* audio_codec = FindValue(audio, "codec_name");
  const std::string* audio_channels = FindValue(audio, "channels");
  const std::string* audio_rate = FindValue(audio, "sample_rate");
  const std::string* audio_start = FindValue(audio, "start_time");
  const std::string* audio_duration = FindValue(audio, "duration");
  if (audio_codec == nullptr || audio_channels == nullptr ||
      audio_rate == nullptr || audio_start == nullptr ||
      audio_duration == nullptr ||
      !ParseInteger(*audio_channels, &probe->audio_channels) ||
      !ParseInteger(*audio_rate, &probe->audio_sample_rate) ||
      !ParseDouble(*audio_start, &probe->audio_start_seconds) ||
      !ParseDouble(*audio_duration, &probe->audio_duration_seconds)) {
    SetError(error, "FFprobe returned malformed MiniMax H3 audio metadata");
    return false;
  }
  probe->audio_codec = *audio_codec;
  return true;
}

}  // namespace gufo::minimax_h3
