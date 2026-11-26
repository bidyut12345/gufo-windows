#include "src/models/minimax_h3/generation.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gufo/ext/platform/os_info.hpp>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>

#include "src/models/minimax_h3/conditioning.hpp"
#include "src/models/minimax_h3/image_processor.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/sha256.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

namespace gufo::minimax_h3 {

std::optional<int> ResolveDurationFrames(std::string_view seconds,
                                         std::string* error) {
  double duration = 0.0;
  const auto [end, status] = std::from_chars(
      seconds.data(), seconds.data() + seconds.size(), duration);
  if (status == std::errc{} && end == seconds.data() + seconds.size() &&
      std::isfinite(duration)) {
    if (duration == 1.0)
      return 22;
    if (duration >= 5.0 && duration <= 15.0) {
      const int frames = AlignFrameCount(
          static_cast<int>(std::ceil(duration * kH3FramesPerSecond)));
      if (frames <= kH3MaximumFrames)
        return frames;
    }
  }
  if (error != nullptr)
    *error =
        "MiniMax H3 seconds must be 1 (diagnostic) or 5..15 with "
        "24-fps VAE alignment fitting within 345 frames (14.375 seconds)";
  return std::nullopt;
}

namespace {

constexpr std::string_view kModelRevision =
    "42ed227ee7df40d41602854ae760620d6eb651fe";
constexpr std::string_view kReferenceRevision =
    "8974cc055ea9c02fcd14cc27dfda3e1027c05153";

std::uint64_t NextOutputNonce() {
  static std::atomic_uint64_t nonce{0};
  return nonce.fetch_add(1, std::memory_order_relaxed);
}

std::uint8_t QuantizeRgb(float source) {
  const float value =
      std::clamp(std::isfinite(source) ? source : 0.0F, 0.0F, 1.0F);
  const float scaled = value * 255.0F;
  const float lower = std::floor(scaled);
  const float fraction = scaled - lower;
  const bool round_up =
      fraction > 0.5F ||
      (fraction == 0.5F && static_cast<unsigned int>(lower) % 2U != 0U);
  return static_cast<std::uint8_t>(lower + (round_up ? 1.0F : 0.0F));
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

#if defined(ENGINE_ENABLE_HIP)
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}
#endif

std::string EscapeJson(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(byte) << std::dec;
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  return output.str();
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents,
                 std::string* error) {
  if (path.empty()) {
    SetError(error, "MiniMax H3 output path is empty");
    return false;
  }
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 output directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial = path.string() + ".gufo-partial-" +
                                        std::to_string(platform::ProcessId()) +
                                        "-" + std::to_string(NextOutputNonce());
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::filesystem::remove(partial, filesystem_error);
      SetError(error, "cannot write MiniMax H3 output file");
      return false;
    }
  }
  std::filesystem::rename(partial, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(partial, filesystem_error);
    SetError(error, "cannot publish MiniMax H3 output file: " +
                        filesystem_error.message());
    return false;
  }
  return true;
}

bool WriteFramesAtomic(const std::filesystem::path& directory,
                       const VideoFrames& frames,
                       const CancellationToken* cancellation,
                       std::string* error) {
  if (directory.empty() || frames.width < 1 || frames.height < 1 ||
      frames.frame_indices.empty()) {
    SetError(error, "invalid MiniMax H3 frame output");
    return false;
  }
  const std::size_t frame_elements =
      static_cast<std::size_t>(frames.width) * frames.height * 3U;
  if (frames.rgb.size() != frame_elements * frames.frame_indices.size()) {
    SetError(error, "MiniMax H3 frame output geometry differs");
    return false;
  }
  std::error_code filesystem_error;
  const bool exists = std::filesystem::exists(directory, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot inspect MiniMax H3 frame directory: " +
                        filesystem_error.message());
    return false;
  }
  if (exists) {
    SetError(error,
             "MiniMax H3 frame directory already exists; refusing to replace "
             "it");
    return false;
  }
  const std::filesystem::path parent = directory.parent_path().empty()
                                           ? std::filesystem::path(".")
                                           : directory.parent_path();
  std::filesystem::create_directories(parent, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 frame parent directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial =
      parent / (directory.filename().string() + ".gufo-partial-" +
                std::to_string(platform::ProcessId()) + "-" +
                std::to_string(NextOutputNonce()));
  std::filesystem::remove_all(partial, filesystem_error);
  filesystem_error.clear();
  std::filesystem::create_directories(partial, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 partial frame directory: " +
                        filesystem_error.message());
    return false;
  }
  for (std::size_t frame = 0; frame < frames.frame_indices.size(); ++frame) {
    if (IsCancelled(cancellation)) {
      std::filesystem::remove_all(partial, filesystem_error);
      SetError(error, "MiniMax H3 frame publication cancelled");
      return false;
    }
    std::string ppm = "P6\n" + std::to_string(frames.width) + " " +
                      std::to_string(frames.height) + "\n255\n";
    ppm.resize(ppm.size() + frame_elements);
    unsigned char* pixels = reinterpret_cast<unsigned char*>(
        ppm.data() + ppm.size() - frame_elements);
    const float* source = frames.rgb.data() + frame * frame_elements;
    for (std::size_t index = 0; index < frame_elements; ++index) {
      pixels[index] = QuantizeRgb(source[index]);
    }
    std::ostringstream name;
    name << "frame-" << std::setw(4) << std::setfill('0')
         << frames.frame_indices[frame] << ".ppm";
    if (!WriteAtomic(partial / name.str(), ppm, error)) {
      std::filesystem::remove_all(partial, filesystem_error);
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    std::filesystem::remove_all(partial, filesystem_error);
    SetError(error, "MiniMax H3 frame publication cancelled");
    return false;
  }
  std::filesystem::rename(partial, directory, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove_all(partial, filesystem_error);
    SetError(error, "cannot publish MiniMax H3 frame directory: " +
                        filesystem_error.message());
    return false;
  }
  return true;
}

struct LatentManifestContext {
  const GenerationParameters* parameters{nullptr};
  std::uint64_t seed{0};
  std::string prompt_sha256;
  std::string conditioning_sha256;
};

bool WriteLatentsAtomic(const std::filesystem::path& directory,
                        const GenerationGeometry& geometry,
                        std::span<const float> video,
                        std::span<const float> audio,
                        const LatentManifestContext* context,
                        const CancellationToken* cancellation,
                        std::string* error) {
  const std::size_t expected_video =
      kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry.temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry.latent_height) *
      static_cast<std::size_t>(geometry.latent_width);
  const std::size_t expected_audio =
      kH3AudioLatentChannels * kH3AudioTracks *
      static_cast<std::size_t>(geometry.temporal.audio_latent_frames);
  if (directory.empty() || video.size() != expected_video ||
      audio.size() != expected_audio) {
    SetError(error, "invalid MiniMax H3 latent output");
    return false;
  }
  std::error_code filesystem_error;
  const bool exists = std::filesystem::exists(directory, filesystem_error);
  if (filesystem_error || exists) {
    SetError(error, filesystem_error
                        ? "cannot inspect MiniMax H3 latent directory: " +
                              filesystem_error.message()
                        : "MiniMax H3 latent directory already exists; "
                          "refusing to replace it");
    return false;
  }
  const std::filesystem::path parent = directory.parent_path().empty()
                                           ? std::filesystem::path(".")
                                           : directory.parent_path();
  std::filesystem::create_directories(parent, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 latent parent directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial =
      parent / (directory.filename().string() + ".gufo-partial-" +
                std::to_string(platform::ProcessId()) + "-" +
                std::to_string(NextOutputNonce()));
  std::filesystem::remove_all(partial, filesystem_error);
  filesystem_error.clear();
  std::filesystem::create_directories(partial, filesystem_error);
  if (filesystem_error) {
    SetError(error, "cannot create MiniMax H3 partial latent directory: " +
                        filesystem_error.message());
    return false;
  }
  const auto bytes = [](std::span<const float> values) {
    return std::span<const unsigned char>(
        reinterpret_cast<const unsigned char*>(values.data()),
        values.size_bytes());
  };
  const auto write = [&](std::string_view name, std::span<const float> values) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 latent publication cancelled");
      return false;
    }
    const auto payload = bytes(values);
    return WriteAtomic(
        partial / name,
        std::string_view(reinterpret_cast<const char*>(payload.data()),
                         payload.size()),
        error);
  };
  if (!write("video_final.f32", video) || !write("audio_final.f32", audio)) {
    std::filesystem::remove_all(partial, filesystem_error);
    return false;
  }
  std::ostringstream manifest;
  manifest << "{\n"
           << "  \"schema\": \"gufo.minimax-h3-final-latents.v1\",\n"
           << "  \"video\": {\"dtype\": \"F32\", \"shape\": ["
           << kH3VideoLatentChannels << ", "
           << geometry.temporal.video_latent_frames << ", "
           << geometry.latent_height << ", " << geometry.latent_width
           << "], \"bytes\": " << video.size_bytes() << ", \"sha256\": \""
           << Sha256(bytes(video)) << "\"},\n"
           << "  \"audio\": {\"dtype\": \"F32\", \"shape\": ["
           << kH3AudioLatentChannels << ", " << kH3AudioTracks << ", "
           << geometry.temporal.audio_latent_frames
           << "], \"bytes\": " << audio.size_bytes() << ", \"sha256\": \""
           << Sha256(bytes(audio)) << "\"}";
  if (context != nullptr && context->parameters != nullptr) {
    const GenerationParameters& parameters = *context->parameters;
    manifest << ",\n"
             << "  \"model_revision\": \"" << kModelRevision << "\",\n"
             << "  \"reference_revision\": \"" << kReferenceRevision << "\",\n"
             << "  \"seed\": " << context->seed << ",\n"
             << "  \"noise_mode\": \"seeded\",\n"
             << "  \"prompt_sha256\": \"" << EscapeJson(context->prompt_sha256)
             << "\",\n"
             << "  \"conditioning_sha256\": \""
             << EscapeJson(context->conditioning_sha256) << "\",\n"
             << "  \"geometry\": {\"width\": " << geometry.width
             << ", \"height\": " << geometry.height
             << ", \"frames\": " << geometry.temporal.frames << "},\n"
             << "  \"steps\": " << parameters.evaluations << ",\n"
             << "  \"blocks\": " << parameters.active_blocks << ",\n"
             << "  \"reuse_interval\": " << parameters.reuse_interval << ",\n"
             << "  \"attention_kernel\": \""
             << (parameters.row_parallel_attention ? "row_parallel" : "scalar")
             << "\"";
  }
  manifest << "\n}\n";
  if (IsCancelled(cancellation) ||
      !WriteAtomic(partial / "manifest.json", manifest.str(), error)) {
    std::filesystem::remove_all(partial, filesystem_error);
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 latent publication cancelled");
    }
    return false;
  }
  std::filesystem::rename(partial, directory, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove_all(partial, filesystem_error);
    SetError(error, "cannot publish MiniMax H3 latent directory: " +
                        filesystem_error.message());
    return false;
  }
  return true;
}

#if defined(ENGINE_ENABLE_HIP)
std::vector<std::uint8_t> ToRgb24(const VideoFrames& frames) {
  std::vector<std::uint8_t> result(frames.rgb.size());
  std::transform(frames.rgb.begin(), frames.rgb.end(), result.begin(),
                 [](float source) { return QuantizeRgb(source); });
  return result;
}

struct ProgressBridge {
  GenerationProgress callback{nullptr};
  void* opaque{nullptr};
  std::string_view phase;
};

void ForwardDenoiserProgress(int completed, int total, void* opaque) {
  auto* bridge = static_cast<ProgressBridge*>(opaque);
  if (bridge != nullptr && bridge->callback != nullptr) {
    bridge->callback(bridge->phase, completed, total, bridge->opaque);
  }
}

void ForwardVideoProgress(int completed, int total, void* opaque) {
  ForwardDenoiserProgress(completed, total, opaque);
}

void ForwardAudioProgress(int completed, int total, void* opaque) {
  ForwardDenoiserProgress(completed, total, opaque);
}

void Report(GenerationProgress progress, void* opaque, std::string_view phase,
            int completed, int total) {
  if (progress != nullptr) {
    progress(phase, completed, total, opaque);
  }
}

enum class CheckpointStage : std::uint8_t {
  kNone = 0,
  kPrompt = 1,
  kDenoise = 2,
  kVideoVae = 3,
  kAudioVae = 4,
};

std::string_view CheckpointStageName(CheckpointStage stage) noexcept {
  switch (stage) {
    case CheckpointStage::kNone:
      return "none";
    case CheckpointStage::kPrompt:
      return "prompt";
    case CheckpointStage::kDenoise:
      return "denoise";
    case CheckpointStage::kVideoVae:
      return "video_vae";
    case CheckpointStage::kAudioVae:
      return "audio_vae";
  }
  return "none";
}

struct CheckpointMeta {
  char magic[8]{'G', 'U', 'F', 'O', 'H', '3', 'C', 'P'};
  std::uint32_t version{1};
  std::uint8_t stage{0};
  std::uint8_t decode_audio{0};
  std::uint8_t mux{0};
  std::uint8_t row_parallel_attention{1};
  std::int32_t completed_steps{0};
  std::int32_t evaluations{0};
  std::int32_t internal_width{0};
  std::int32_t internal_height{0};
  std::int32_t output_width{0};
  std::int32_t output_height{0};
  std::int32_t frames{0};
  std::int32_t active_blocks{0};
  std::int32_t reuse_interval{0};
  std::uint64_t seed{0};
  char prompt_sha256[64]{};
};

CheckpointMeta MakeCheckpointMeta(const GenerationParameters& parameters,
                                  std::uint64_t seed,
                                  std::string_view prompt_sha256,
                                  CheckpointStage stage,
                                  int completed_steps) {
  CheckpointMeta meta;
  meta.stage = static_cast<std::uint8_t>(stage);
  meta.decode_audio = parameters.decode_audio ? 1U : 0U;
  meta.mux = parameters.mux ? 1U : 0U;
  meta.row_parallel_attention = parameters.row_parallel_attention ? 1U : 0U;
  meta.completed_steps = completed_steps;
  meta.evaluations = parameters.evaluations;
  meta.internal_width = parameters.internal_width;
  meta.internal_height = parameters.internal_height;
  meta.output_width = parameters.output_width;
  meta.output_height = parameters.output_height;
  meta.frames = parameters.frames;
  meta.active_blocks = parameters.active_blocks;
  meta.reuse_interval = parameters.reuse_interval;
  meta.seed = seed;
  const std::size_t n = std::min(prompt_sha256.size(), sizeof(meta.prompt_sha256));
  std::copy_n(prompt_sha256.data(), n, meta.prompt_sha256);
  return meta;
}

bool CheckpointMetaMatches(const CheckpointMeta& meta,
                           const GenerationParameters& parameters,
                           std::uint64_t seed,
                           std::string_view prompt_sha256) {
  constexpr char kMagic[8]{'G', 'U', 'F', 'O', 'H', '3', 'C', 'P'};
  if (!std::equal(std::begin(meta.magic), std::end(meta.magic),
                  std::begin(kMagic)) ||
      meta.version != 1U) {
    return false;
  }
  const std::size_t n = std::min(prompt_sha256.size(), sizeof(meta.prompt_sha256));
  return meta.decode_audio == (parameters.decode_audio ? 1U : 0U) &&
         meta.mux == (parameters.mux ? 1U : 0U) &&
         meta.row_parallel_attention ==
             (parameters.row_parallel_attention ? 1U : 0U) &&
         meta.evaluations == parameters.evaluations &&
         meta.internal_width == parameters.internal_width &&
         meta.internal_height == parameters.internal_height &&
         meta.output_width == parameters.output_width &&
         meta.output_height == parameters.output_height &&
         meta.frames == parameters.frames &&
         meta.active_blocks == parameters.active_blocks &&
         meta.reuse_interval == parameters.reuse_interval &&
         meta.seed == seed &&
         std::equal(meta.prompt_sha256, meta.prompt_sha256 + n,
                    prompt_sha256.begin());
}

template <typename Fn>
bool WriteBinaryFileAtomic(const std::filesystem::path& path, Fn&& writer,
                           std::string* error) {
  std::error_code fs_err;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), fs_err);
    if (fs_err) {
      SetError(error, "cannot create checkpoint directory: " + fs_err.message());
      return false;
    }
  }
  const std::filesystem::path partial =
      path.parent_path() /
      (path.filename().string() + ".gufo-partial-" +
       std::to_string(platform::ProcessId()) + "-" +
       std::to_string(NextOutputNonce()));
  {
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      SetError(error, "cannot open checkpoint file " + partial.string());
      return false;
    }
    writer(out);
    out.flush();
    if (!out.good()) {
      out.close();
      std::filesystem::remove(partial, fs_err);
      SetError(error, "cannot write checkpoint file " + partial.string());
      return false;
    }
  }
  std::filesystem::rename(partial, path, fs_err);
  if (fs_err) {
    std::filesystem::remove(partial, fs_err);
    SetError(error, "cannot publish checkpoint file: " + fs_err.message());
    return false;
  }
  return true;
}

template <typename T>
void WritePod(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void WriteVec(std::ostream& out, const std::vector<T>& vec) {
  const std::uint64_t count = static_cast<std::uint64_t>(vec.size());
  WritePod(out, count);
  if (!vec.empty()) {
    out.write(reinterpret_cast<const char*>(vec.data()),
              static_cast<std::streamsize>(vec.size() * sizeof(T)));
  }
}

template <typename T>
bool ReadPod(std::istream& in, T* value) {
  return static_cast<bool>(
      in.read(reinterpret_cast<char*>(value), sizeof(T)));
}

template <typename T>
bool ReadVec(std::istream& in, std::vector<T>* vec) {
  std::uint64_t count = 0;
  if (!ReadPod(in, &count)) {
    return false;
  }
  vec->resize(static_cast<std::size_t>(count));
  if (count == 0U) {
    return true;
  }
  return static_cast<bool>(
      in.read(reinterpret_cast<char*>(vec->data()),
              static_cast<std::streamsize>(vec->size() * sizeof(T))));
}

bool SaveCheckpointMeta(const std::filesystem::path& dir,
                        const CheckpointMeta& meta, std::string* error) {
  if (!WriteBinaryFileAtomic(
          dir / "meta.bin", [&](std::ostream& out) { WritePod(out, meta); },
          error)) {
    return false;
  }
  std::ostringstream json;
  json << "{\n"
       << "  \"schema\": \"gufo.minimax-h3-checkpoint.v1\",\n"
       << "  \"stage\": \""
       << CheckpointStageName(static_cast<CheckpointStage>(meta.stage))
       << "\",\n"
       << "  \"step\": " << meta.completed_steps << ",\n"
       << "  \"total_steps\": " << meta.evaluations << "\n"
       << "}\n";
  return WriteAtomic(dir / "checkpoint.json", json.str(), error);
}

bool SavePromptCheckpoint(const std::filesystem::path& dir,
                          const PromptEmbedding& embedding,
                          const std::vector<float>& condition_rows,
                          const std::vector<ConditionBlock>& condition_blocks,
                          const std::vector<double>& anchor_temporal,
                          double target_temporal_origin,
                          const InitialNoise& noise, std::string* error) {
  return WriteBinaryFileAtomic(
      dir / "prompt.bin",
      [&](std::ostream& out) {
        WritePod(out, embedding.tokens);
        WriteVec(out, embedding.values);
        WriteVec(out, embedding.tags);
        WriteVec(out, condition_rows);
        WriteVec(out, condition_blocks);
        WriteVec(out, anchor_temporal);
        WritePod(out, target_temporal_origin);
        WriteVec(out, noise.video);
        WriteVec(out, noise.audio);
      },
      error);
}

bool LoadPromptCheckpoint(const std::filesystem::path& dir,
                          PromptEmbedding* embedding,
                          std::vector<float>* condition_rows,
                          std::vector<ConditionBlock>* condition_blocks,
                          std::vector<double>* anchor_temporal,
                          double* target_temporal_origin,
                          InitialNoise* noise, std::string* error) {
  std::ifstream in(dir / "prompt.bin", std::ios::binary);
  if (!in.is_open() || !ReadPod(in, &embedding->tokens) ||
      !ReadVec(in, &embedding->values) || !ReadVec(in, &embedding->tags) ||
      !ReadVec(in, condition_rows) || !ReadVec(in, condition_blocks) ||
      !ReadVec(in, anchor_temporal) ||
      !ReadPod(in, target_temporal_origin) || !ReadVec(in, &noise->video) ||
      !ReadVec(in, &noise->audio)) {
    SetError(error, "cannot read MiniMax H3 prompt checkpoint");
    return false;
  }
  return true;
}

bool SaveDenoiseCheckpoint(const std::filesystem::path& dir,
                           const DenoiserCheckpointState& state,
                           std::string* error) {
  return WriteBinaryFileAtomic(
      dir / "denoise.bin",
      [&](std::ostream& out) {
        const std::uint8_t hv = state.have_velocity ? 1U : 0U;
        WritePod(out, state.completed_steps);
        WritePod(out, hv);
        WriteVec(out, state.video_latent);
        WriteVec(out, state.audio_latent);
        WriteVec(out, state.video_velocity);
        WriteVec(out, state.audio_velocity);
        WriteVec(out, state.previous_video_velocity);
        WriteVec(out, state.previous_audio_velocity);
      },
      error);
}

bool LoadDenoiseCheckpoint(const std::filesystem::path& dir,
                           DenoiserCheckpointState* state, std::string* error) {
  std::ifstream in(dir / "denoise.bin", std::ios::binary);
  std::uint8_t hv = 0;
  if (!in.is_open() || !ReadPod(in, &state->completed_steps) ||
      !ReadPod(in, &hv) || !ReadVec(in, &state->video_latent) ||
      !ReadVec(in, &state->audio_latent) ||
      !ReadVec(in, &state->video_velocity) ||
      !ReadVec(in, &state->audio_velocity) ||
      !ReadVec(in, &state->previous_video_velocity) ||
      !ReadVec(in, &state->previous_audio_velocity)) {
    SetError(error, "cannot read MiniMax H3 denoise checkpoint");
    return false;
  }
  state->have_velocity = hv != 0U;
  return true;
}

bool SaveVideoVaeCheckpoint(const std::filesystem::path& dir,
                            const VideoFrames& frames, std::string* error) {
  return WriteBinaryFileAtomic(
      dir / "video_vae.bin",
      [&](std::ostream& out) {
        WritePod(out, frames.width);
        WritePod(out, frames.height);
        WriteVec(out, frames.frame_indices);
        WriteVec(out, frames.rgb);
      },
      error);
}

bool LoadVideoVaeCheckpoint(const std::filesystem::path& dir,
                            VideoFrames* frames, std::string* error) {
  std::ifstream in(dir / "video_vae.bin", std::ios::binary);
  if (!in.is_open() || !ReadPod(in, &frames->width) ||
      !ReadPod(in, &frames->height) || !ReadVec(in, &frames->frame_indices) ||
      !ReadVec(in, &frames->rgb)) {
    SetError(error, "cannot read MiniMax H3 video VAE checkpoint");
    return false;
  }
  return true;
}

bool SaveAudioVaeCheckpoint(const std::filesystem::path& dir,
                            const AudioWaveform& waveform, std::string* error) {
  return WriteBinaryFileAtomic(
      dir / "audio_vae.bin",
      [&](std::ostream& out) {
        WritePod(out, waveform.sample_rate);
        WritePod(out, waveform.channels);
        WritePod(out, waveform.samples);
        WriteVec(out, waveform.pcm);
      },
      error);
}

bool LoadAudioVaeCheckpoint(const std::filesystem::path& dir,
                            AudioWaveform* waveform, std::string* error) {
  std::ifstream in(dir / "audio_vae.bin", std::ios::binary);
  if (!in.is_open() || !ReadPod(in, &waveform->sample_rate) ||
      !ReadPod(in, &waveform->channels) || !ReadPod(in, &waveform->samples) ||
      !ReadVec(in, &waveform->pcm)) {
    SetError(error, "cannot read MiniMax H3 audio VAE checkpoint");
    return false;
  }
  return true;
}

struct DenoiseCheckpointContext {
  std::filesystem::path dir;
  GenerationParameters parameters;
  std::uint64_t seed{0};
  std::string prompt_sha256;
};

bool OnDenoiseStepCheckpoint(const DenoiserCheckpointState& state,
                             void* opaque, std::string* error) {
  auto* ctx = static_cast<DenoiseCheckpointContext*>(opaque);
  if (ctx == nullptr || ctx->dir.empty()) {
    return true;
  }
  if (!SaveDenoiseCheckpoint(ctx->dir, state, error)) {
    return false;
  }
  return SaveCheckpointMeta(
      ctx->dir,
      MakeCheckpointMeta(ctx->parameters, ctx->seed, ctx->prompt_sha256,
                         CheckpointStage::kDenoise, state.completed_steps),
      error);
}
#endif

}  // namespace

std::optional<GenerationParameters> ResolveGenerationPreset(
    std::string_view name, std::string* error) {
  GenerationParameters parameters;
  if (name == "exact" || name == "exact-512") {
    parameters = {.preset = "exact-512",
                  .internal_width = 512,
                  .internal_height = 512,
                  .output_width = 512,
                  .output_height = 512,
                  .frames = 22,
                  .evaluations = 49,
                  .active_blocks = 50,
                  .reuse_interval = 1,
                  .selected_frames = {}};
  } else if (name == "exact-1344x768" || name == "fullres") {
    parameters = {.preset = "exact-1344x768",
                  .internal_width = 1344,
                  .internal_height = 768,
                  .output_width = 1344,
                  .output_height = 768,
                  .frames = 124,
                  .evaluations = 49,
                  .active_blocks = 50,
                  .reuse_interval = 1,
                  .selected_frames = {}};
  } else if (name == "fast" || name == "fast-384") {
    parameters = {.preset = "fast-384",
                  .internal_width = 384,
                  .internal_height = 384,
                  .output_width = 512,
                  .output_height = 512,
                  .frames = 22,
                  .evaluations = 19,
                  .active_blocks = 45,
                  .reuse_interval = 2,
                  .selected_frames = {}};
  } else if (name == "aggressive" || name == "aggressive-320") {
    parameters = {.preset = "aggressive-320",
                  .internal_width = 320,
                  .internal_height = 320,
                  .output_width = 512,
                  .output_height = 512,
                  .frames = 22,
                  .evaluations = 19,
                  .active_blocks = 40,
                  .reuse_interval = 3,
                  .selected_frames = {}};
  } else if (name == "dev" || name == "development-256") {
    parameters = {.preset = "development-256",
                  .internal_width = 256,
                  .internal_height = 256,
                  .output_width = 256,
                  .output_height = 256,
                  .frames = 22,
                  .evaluations = 4,
                  .active_blocks = 50,
                  .reuse_interval = 1,
                  .decode_audio = false,
                  .mux = false,
                  .selected_frames = {0, 11, 21}};
  } else {
    SetError(error,
             "unknown MiniMax H3 preset; expected exact, exact-1344x768, "
             "fast, aggressive, or dev");
    return std::nullopt;
  }
  return parameters;
}

std::optional<GenerationParameters> ResolveKeyframeParameters(
    const GenerationParameters& parameters, int keyframe_width,
    int keyframe_height, std::string* error) {
  if (parameters.internal_width < kH3CanvasMultiple ||
      parameters.internal_height < kH3CanvasMultiple ||
      parameters.output_width < 2 || parameters.output_height < 2 ||
      keyframe_width <= 0 || keyframe_height <= 0) {
    SetError(error, "invalid MiniMax H3 keyframe canvas request");
    return std::nullopt;
  }
  const int short_edge =
      std::min(parameters.internal_width, parameters.internal_height);
  const std::int64_t budget =
      static_cast<std::int64_t>(parameters.internal_width) *
      parameters.internal_height;
  const auto canvas = ResolveCanvasSize(keyframe_width, keyframe_height,
                                        short_edge, budget, error);
  if (!canvas.has_value()) {
    return std::nullopt;
  }
  // The released processor would upscale such a keyframe before the vision
  // tower; only the small `dev` budget and custom canvases reach this.
  if (static_cast<std::int64_t>(canvas->width) * canvas->height <
      image_processor::kVisionMinimumPixels) {
    SetError(error,
             "MiniMax H3 keyframe canvas " + std::to_string(canvas->width) +
                 "x" + std::to_string(canvas->height) +
                 " is below the 65536-pixel vision minimum; use a larger "
                 "preset or size, or a keyframe closer to square");
    return std::nullopt;
  }
  const auto scale_axis = [](int value, int preset_output,
                             int preset_internal) {
    if (preset_output == preset_internal) {
      return value;
    }
    const double scaled =
        static_cast<double>(value) * preset_output / (2.0 * preset_internal);
    return std::max(1, static_cast<int>(std::lround(scaled))) * 2;
  };
  GenerationParameters resolved = parameters;
  resolved.internal_width = canvas->width;
  resolved.internal_height = canvas->height;
  resolved.output_width = scale_axis(canvas->width, parameters.output_width,
                                     parameters.internal_width);
  resolved.output_height = scale_axis(canvas->height, parameters.output_height,
                                      parameters.internal_height);
  return resolved;
}

bool ValidateGenerationParameters(const GenerationParameters& parameters,
                                  std::string* error) {
  if (parameters.preset.empty() || parameters.output_width < 2 ||
      parameters.output_height < 2 || (parameters.output_width & 1) != 0 ||
      (parameters.output_height & 1) != 0 || parameters.evaluations < 2 ||
      parameters.evaluations > kH3MaximumEvaluations ||
      parameters.active_blocks < 3 || parameters.active_blocks > 50 ||
      parameters.reuse_interval < 1 || parameters.reuse_interval > 32) {
    SetError(error, "invalid MiniMax H3 generation parameters");
    return false;
  }
  std::string geometry_error;
  const auto geometry = ResolveGenerationGeometry(
      parameters.internal_width, parameters.internal_height, parameters.frames,
      &geometry_error);
  if (!geometry.has_value() || geometry->temporal.frames != parameters.frames) {
    SetError(error, geometry_error.empty()
                        ? "MiniMax H3 generation requires a legal frame count"
                        : geometry_error);
    return false;
  }
  if (!ResolveVideoVaePlan(*geometry, &geometry_error).has_value()) {
    SetError(error, geometry_error.empty()
                        ? "MiniMax H3 generation requires decodable VisualVAE "
                          "geometry"
                        : geometry_error);
    return false;
  }
  int previous = -1;
  for (const int frame : parameters.selected_frames) {
    if (frame <= previous || frame < 0 || frame >= parameters.frames) {
      SetError(error,
               "MiniMax H3 selected frames must be increasing and in range");
      return false;
    }
    previous = frame;
  }
  if (parameters.mux &&
      (!parameters.decode_audio || !parameters.selected_frames.empty())) {
    SetError(error,
             "MiniMax H3 MP4 output requires audio and every video frame");
    return false;
  }
  return true;
}

std::string GenerationParametersJson(const GenerationParameters& parameters,
                                     std::uint64_t seed,
                                     std::string_view prompt_sha256) {
  std::ostringstream output;
  output << "{\n"
         << "  \"schema\": \"" << kH3GenerationSchema << "\",\n"
         << "  \"backend\": \"rocm-hip-gfx1151\",\n"
         << "  \"precision\": \"bf16-f32\",\n"
         << "  \"model_repository\": \"MiniMaxAI/MiniMax-H3\",\n"
         << "  \"model_revision\": \"" << kModelRevision << "\",\n"
         << "  \"reference_repository\": \"antirez/h3.c\",\n"
         << "  \"reference_revision\": \"" << kReferenceRevision << "\",\n"
         << "  \"preset\": \"" << EscapeJson(parameters.preset) << "\",\n"
         << "  \"prompt_sha256\": \"" << EscapeJson(prompt_sha256) << "\",\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"internal_width\": " << parameters.internal_width << ",\n"
         << "  \"internal_height\": " << parameters.internal_height << ",\n"
         << "  \"output_width\": " << parameters.output_width << ",\n"
         << "  \"output_height\": " << parameters.output_height << ",\n"
         << "  \"fps\": " << kH3FramesPerSecond << ",\n"
         << "  \"frames\": " << parameters.frames << ",\n"
         << "  \"evaluations\": " << parameters.evaluations << ",\n"
         << "  \"active_blocks\": " << parameters.active_blocks << ",\n"
         << "  \"reuse_interval\": " << parameters.reuse_interval << ",\n"
         << "  \"attention_kernel\": \""
         << (parameters.row_parallel_attention ? "row_parallel" : "scalar")
         << "\",\n"
         << "  \"token_reduction\": false,\n"
         << "  \"decode_audio\": "
         << (parameters.decode_audio ? "true" : "false") << ",\n"
         << "  \"mux\": " << (parameters.mux ? "true" : "false") << ",\n"
         << "  \"selected_frames\": [";
  for (std::size_t index = 0; index < parameters.selected_frames.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << parameters.selected_frames[index];
  }
  output << "]\n}\n";
  return output.str();
}

bool WriteDiagnosticFrames(const std::filesystem::path& directory,
                           const VideoFrames& frames,
                           const CancellationToken* cancellation,
                           std::string* error) {
  return WriteFramesAtomic(directory, frames, cancellation, error);
}

bool WriteDiagnosticLatents(const std::filesystem::path& directory,
                            const GenerationGeometry& geometry,
                            std::span<const float> video,
                            std::span<const float> audio,
                            const CancellationToken* cancellation,
                            std::string* error) {
  return WriteLatentsAtomic(directory, geometry, video, audio, nullptr,
                            cancellation, error);
}

bool GenerateTextVideo(const GenerationRequest& request,
                       const CancellationToken* cancellation,
                       GenerationProgress progress, void* progress_opaque,
                       GenerationTelemetry* telemetry, std::string* error) {
  if (telemetry != nullptr) {
    *telemetry = {};
  }
#if !defined(ENGINE_ENABLE_HIP)
  (void)request;
  (void)cancellation;
  (void)progress;
  (void)progress_opaque;
  SetError(error,
           "MiniMax H3 generation requires a ROCm HIP-enabled build; no CPU "
           "inference fallback is provided");
  return false;
#else
  if (request.model_root.empty() || request.source_manifest.empty() ||
      request.prompt.empty() ||
      !ValidateGenerationParameters(request.parameters, error) ||
      (request.parameters.mux && request.output_path.empty()) ||
      (!request.parameters.mux && request.frames_directory.empty()) ||
      IsCancelled(cancellation)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "invalid MiniMax H3 text generation request");
    }
    return false;
  }
  const bool has_keyframes =
      !request.first_frame_path.empty() || !request.last_frame_path.empty();
  const bool has_references = !request.reference_images.empty();
  const bool has_videos = !request.reference_videos.empty();
  const bool has_ref2va = has_references || has_videos;
  if (request.reference_images.size() > 9U) {
    SetError(error, "MiniMax H3 Ref2VA accepts at most 9 reference images");
    return false;
  }
  if (request.reference_videos.size() > 3U) {
    SetError(error, "MiniMax H3 Ref2VA accepts at most 3 reference videos");
    return false;
  }
  if (request.reference_images.size() + request.reference_videos.size() > 12U) {
    SetError(error, "MiniMax H3 Ref2VA accepts at most 12 references in total");
    return false;
  }
  // The geometry anchor (packed index 0: the first keyframe, or a lone last
  // one) selects the canvas before any model work unless
  // lock_canvas_to_preset is set or Ref2VA references are present; Ref2VA
  // references never bind the output canvas.
  GenerationParameters parameters = request.parameters;
  core::Image first_image;
  core::Image last_image;
  if (has_keyframes) {
    if ((!request.first_frame_path.empty() &&
         !DecodeKeyframe(request.first_frame_path, &first_image, error)) ||
        (!request.last_frame_path.empty() &&
         !DecodeKeyframe(request.last_frame_path, &last_image, error))) {
      return false;
    }
    if (!request.lock_canvas_to_preset && !has_ref2va) {
      const core::Image& anchor =
          request.first_frame_path.empty() ? last_image : first_image;
      auto resolved = ResolveKeyframeParameters(
          request.parameters, static_cast<int>(anchor.width),
          static_cast<int>(anchor.height), error);
      if (!resolved.has_value() ||
          !ValidateGenerationParameters(*resolved, error)) {
        return false;
      }
      parameters = std::move(*resolved);
    }
  }
  std::vector<core::Image> ref_images;
  if (has_references) {
    ref_images.resize(request.reference_images.size());
    for (std::size_t i = 0; i < request.reference_images.size(); ++i) {
      if (!DecodeKeyframe(request.reference_images[i], &ref_images[i], error)) {
        return false;
      }
      const std::uint64_t w = ref_images[i].width;
      const std::uint64_t h = ref_images[i].height;
      if (w > 4U * h || h > 4U * w) {
        SetError(error,
                 "MiniMax H3 reference image aspect ratio must be within 1:4 "
                 "to 4:1");
        return false;
      }
    }
  }
  std::vector<DecodedReferenceVideo> ref_videos;
  if (has_videos) {
    ref_videos.resize(request.reference_videos.size());
    for (std::size_t i = 0; i < request.reference_videos.size(); ++i) {
      if (!DecodeReferenceVideo(request.reference_videos[i], parameters.frames,
                                &ref_videos[i], error)) {
        return false;
      }
    }
  }
  GenerationTelemetry local;
  local.internal_width = parameters.internal_width;
  local.internal_height = parameters.internal_height;
  local.output_width = parameters.output_width;
  local.output_height = parameters.output_height;
  const auto total_begin = Clock::now();
  Report(progress, progress_opaque, "inventory", 0, 1);
  auto begin = Clock::now();
  auto inventory = ModelInventory::Inspect(request.model_root,
                                           request.source_manifest, error);
  local.inventory_ms = Milliseconds(begin, Clock::now());
  if (!inventory.has_value()) {
    return false;
  }
  if (has_ref2va && !inventory->is_ref2va()) {
    SetError(
        error,
        "MiniMax H3 reference images/videos require the Ref2VA model checkpoint");
    return false;
  }
  local.inventory = inventory->telemetry();
  Report(progress, progress_opaque, "inventory", 1, 1);

  const std::string prompt_sha256 = Sha256(std::span<const unsigned char>(
      reinterpret_cast<const unsigned char*>(request.prompt.data()),
      request.prompt.size()));
  const bool has_checkpoint = !request.checkpoint_directory.empty();
  CheckpointStage resumed_stage = CheckpointStage::kNone;
  DenoiserCheckpointState denoise_cp;
  if (has_checkpoint && request.resume_from_checkpoint) {
    std::ifstream meta_in(request.checkpoint_directory / "meta.bin",
                          std::ios::binary);
    CheckpointMeta meta;
    if (meta_in.is_open() && ReadPod(meta_in, &meta) &&
        CheckpointMetaMatches(meta, parameters, request.seed, prompt_sha256)) {
      resumed_stage = static_cast<CheckpointStage>(meta.stage);
    }
  }

  const auto geometry = ResolveGenerationGeometry(parameters.internal_width,
                                                  parameters.internal_height,
                                                  parameters.frames, error);
  if (!geometry.has_value()) {
    return false;
  }
  const std::size_t video_elements =
      kH3VideoLatentChannels *
      static_cast<std::size_t>(geometry->temporal.video_latent_frames) *
      static_cast<std::size_t>(geometry->latent_height) *
      static_cast<std::size_t>(geometry->latent_width);

  PromptEmbedding embedding;
  std::vector<float> condition_rows;
  std::vector<double> anchor_temporal;
  std::vector<ConditionBlock> condition_blocks;
  double target_temporal_origin = 0.0;
  std::optional<InitialNoise> noise;

  if (resumed_stage >= CheckpointStage::kPrompt) {
    InitialNoise loaded_noise;
    if (!LoadPromptCheckpoint(request.checkpoint_directory, &embedding,
                              &condition_rows, &condition_blocks,
                              &anchor_temporal, &target_temporal_origin,
                              &loaded_noise, error)) {
      return false;
    }
    noise = std::move(loaded_noise);
    local.prompt_tokens = embedding.tokens;
    Report(progress, progress_opaque, "prompt", 1, 1);
  } else {
    begin = Clock::now();
    Tokenizer tokenizer;
    std::vector<std::uint32_t> tokens;
    if (!Tokenizer::Load(
            request.model_root / inventory->variant_dir() /
                "tokenizer/tokenizer.json",
            &tokenizer, error) ||
        !tokenizer.EncodePrompt(request.prompt, &tokens, error) ||
        tokens.empty()) {
      return false;
    }
    local.tokenizer_ms = Milliseconds(begin, Clock::now());
    local.prompt_tokens = tokens.size();

    Report(progress, progress_opaque, "prompt", 0, 1);
  // Reference workflow order (modular_blocks_minimax_h3.py): resize onto the
  // canvas by the index rule, present keyframes/references to the prompt
  // encoder, then VisualVAE-encode. Anchor latents depend only on pixels (fixed
  // posterior seed), so their order against the prompt stream is numerically
  // neutral.
  std::vector<PreparedKeyframe> keyframes;
  if (has_references) {
    constexpr double kReferenceImageShortEdge = 2048.0;
    for (const core::Image& ref : ref_images) {
      const double w = static_cast<double>(ref.width);
      const double h = static_cast<double>(ref.height);
      const double scale = kReferenceImageShortEdge / std::min(w, h);
      const int target_h = std::max(
          kH3CanvasMultiple,
          static_cast<int>(std::llround(
              std::nearbyint(h * scale / kH3CanvasMultiple))) *
              kH3CanvasMultiple);
      const int target_w = std::max(
          kH3CanvasMultiple,
          static_cast<int>(std::llround(
              std::nearbyint(w * scale / kH3CanvasMultiple))) *
              kH3CanvasMultiple);
      keyframes.emplace_back();
      if (!PrepareKeyframe(ref, target_w, target_h, false, &keyframes.back(),
                           error)) {
        return false;
      }
    }
  }
  if (has_keyframes) {
    const int canvas_width = parameters.internal_width;
    const int canvas_height = parameters.internal_height;
    const bool lock_fit = request.lock_canvas_to_preset || has_ref2va;
    if (!request.first_frame_path.empty()) {
      keyframes.emplace_back();
      if (!PrepareKeyframe(first_image, canvas_width, canvas_height,
                           lock_fit, &keyframes.back(), error)) {
        return false;
      }
    }
    if (!request.last_frame_path.empty()) {
      // A lone last keyframe is packed index 0: the released geometry
      // anchor stretches it (before_encoder.py), while a follower beside a
      // first keyframe (or a locked canvas / Ref2VA combined) takes cover-crop.
      const bool follower = lock_fit || !request.first_frame_path.empty();
      keyframes.emplace_back();
      if (!PrepareKeyframe(last_image, canvas_width, canvas_height, follower,
                           &keyframes.back(), error)) {
        return false;
      }
    }
  }
  begin = Clock::now();
  if (keyframes.empty() && ref_videos.empty()) {
    if (!EncodePromptLayer50(*inventory, tokens, {}, cancellation, &embedding,
                             &local.prompt_encoder, error)) {
      return false;
    }
  } else {
    std::vector<MultimodalBlock> vision_blocks;
    std::vector<KeyframeVision> visions;
    for (std::size_t i = 0; i < keyframes.size(); ++i) {
      KeyframeVision vision;
      if (!image_processor::BuildKeyframeVision(
              keyframes[i].rgb, keyframes[i].width, keyframes[i].height,
              &vision, error)) {
        return false;
      }
      vision_blocks.push_back(
          {.prefix_label = "<Picture " + std::to_string(i + 1) + ">: ",
           .is_video = false,
           .grid_h = vision.grid_h,
           .grid_w = vision.grid_w});
      visions.push_back(std::move(vision));
    }
    for (std::size_t v = 0; v < ref_videos.size(); ++v) {
      const DecodedReferenceVideo& vid = ref_videos[v];
      // Sample at 2 fps from 24 fps (stride = 12.0), merge in pairs.
      constexpr double kSampleFps = 2.0;
      constexpr double kStride = 24.0 / kSampleFps;
      std::vector<int> indices;
      double cursor = 0.0;
      while (static_cast<int>(std::llround(cursor)) < vid.frame_count) {
        const int idx = static_cast<int>(std::llround(cursor));
        if (indices.empty() || idx > indices.back()) {
          indices.push_back(idx);
        }
        cursor += kStride;
      }
      if (indices.size() < 2U) {
        SetError(error,
                 "MiniMax H3 reference video is too short for 2 fps sampling");
        return false;
      }
      std::vector<double> timestamps;
      timestamps.reserve(indices.size() + 1U);
      for (std::size_t k = 0; k < indices.size(); ++k) {
        timestamps.push_back(static_cast<double>(k) / kSampleFps);
      }
      if (timestamps.size() % 2U != 0U) {
        indices.push_back(indices.back());
        timestamps.push_back(timestamps.back());
      }
      const std::size_t frame_bytes =
          static_cast<std::size_t>(vid.width) * vid.height * 3U;
      for (std::size_t p = 0; p < indices.size(); p += 2U) {
        KeyframeVision vision;
        const std::uint8_t* f0 =
            vid.rgb24.data() + static_cast<std::size_t>(indices[p]) * frame_bytes;
        const std::uint8_t* f1 =
            vid.rgb24.data() +
            static_cast<std::size_t>(indices[p + 1U]) * frame_bytes;
        if (!image_processor::BuildVideoPairVision(
                f0, f1, vid.width, vid.height, &vision, error)) {
          if (error != nullptr && error->empty()) {
            SetError(error, "MiniMax H3 BuildVideoPairVision failed (" +
                                std::to_string(vid.width) + "x" +
                                std::to_string(vid.height) + ", frames=" +
                                std::to_string(vid.frame_count) + ")");
          }
          return false;
        }
        const double mean_ts = (timestamps[p] + timestamps[p + 1U]) * 0.5;
        std::ostringstream label;
        if (p == 0U) {
          label << "<Video " << (v + 1) << ">: ";
        }
        label << "<" << std::fixed << std::setprecision(1) << mean_ts
              << " seconds>";
        vision_blocks.push_back({.prefix_label = label.str(),
                                 .is_video = true,
                                 .grid_h = vision.grid_h,
                                 .grid_w = vision.grid_w});
        visions.push_back(std::move(vision));
      }
    }
    if (!EncodePromptWithVisions(*inventory, tokenizer, request.prompt,
                                 vision_blocks, visions, cancellation,
                                 &embedding, &local.prompt_encoder, error)) {
      if (error != nullptr && error->empty()) {
        SetError(error, "MiniMax H3 EncodePromptWithVisions failed");
      }
      return false;
    }
  }
  local.prompt_ms = Milliseconds(begin, Clock::now());
  Report(progress, progress_opaque, "prompt", 1, 1);

  if (!has_keyframes && !has_references && !has_videos) {
    noise =
        BuildInitialNoise(request.seed, video_elements, kH3AudioLatentChannels,
                          geometry->temporal.audio_latent_frames, error);
    if (!noise.has_value()) {
      return false;
    }
  } else {
    const std::size_t ref_count = ref_images.size();
    std::vector<std::size_t> cond_latent_sizes;
    cond_latent_sizes.reserve(keyframes.size() + ref_videos.size());
    for (std::size_t i = 0; i < ref_count; ++i) {
      if (!EncodeKeyframeLatent(*inventory, keyframes[i], cancellation, nullptr,
                                error)) {
        return false;
      }
      const std::size_t lh =
          static_cast<std::size_t>(keyframes[i].height / kH3VaeSpatialRatio);
      const std::size_t lw =
          static_cast<std::size_t>(keyframes[i].width / kH3VaeSpatialRatio);
      cond_latent_sizes.push_back(kH3VideoVaeLatentChannels * lh * lw);
    }
    struct EncodedVideoRef {
      std::vector<float> latent;
      int latent_frames{0};
      int latent_height{0};
      int latent_width{0};
    };
    std::vector<EncodedVideoRef> encoded_videos(ref_videos.size());
    for (std::size_t v = 0; v < ref_videos.size(); ++v) {
      Report(progress, progress_opaque, "video_ref_vae", static_cast<int>(v),
             static_cast<int>(ref_videos.size()));
      const DecodedReferenceVideo& vid = ref_videos[v];
      if (!EncodeVideoReferenceLatent(
              *inventory, vid.rgb24, vid.frame_count, vid.width, vid.height,
              cancellation, nullptr, &encoded_videos[v].latent,
              &encoded_videos[v].latent_frames, error)) {
        return false;
      }
      Report(progress, progress_opaque, "video_ref_vae",
             static_cast<int>(v + 1), static_cast<int>(ref_videos.size()));
      encoded_videos[v].latent_height = vid.height / kH3VaeSpatialRatio;
      encoded_videos[v].latent_width = vid.width / kH3VaeSpatialRatio;
      cond_latent_sizes.push_back(encoded_videos[v].latent.size());
    }
    for (std::size_t i = ref_count; i < keyframes.size(); ++i) {
      if (!EncodeKeyframeLatent(*inventory, keyframes[i], cancellation, nullptr,
                                error)) {
        return false;
      }
      const std::size_t lh =
          static_cast<std::size_t>(keyframes[i].height / kH3VaeSpatialRatio);
      const std::size_t lw =
          static_cast<std::size_t>(keyframes[i].width / kH3VaeSpatialRatio);
      cond_latent_sizes.push_back(kH3VideoVaeLatentChannels * lh * lw);
    }
    // Request-stream consumption order (before_denoise.py): one augmentation
    // tensor per condition in packed order, then video, then audio.
    noise = BuildInitialNoise(
        request.seed, cond_latent_sizes, video_elements,
        kH3AudioLatentChannels, geometry->temporal.audio_latent_frames, error);
    if (!noise.has_value()) {
      return false;
    }
    std::size_t noise_idx = 0;
    double rotary_time = static_cast<double>(embedding.tokens);
    // 1. Image references
    for (std::size_t i = 0; i < ref_count; ++i, ++noise_idx) {
      std::vector<float> mixed;
      if (!MixKeyframeNoise(keyframes[i], noise->conditions[noise_idx], &mixed,
                            error)) {
        return false;
      }
      const int lh = keyframes[i].height / kH3VaeSpatialRatio;
      const int lw = keyframes[i].width / kH3VaeSpatialRatio;
      const std::size_t rows_for_cond =
          static_cast<std::size_t>(lh / 2) * static_cast<std::size_t>(lw / 2);
      const std::size_t offset = condition_rows.size();
      condition_rows.resize(offset + rows_for_cond * kH3VideoPatchWidth);
      if (!PatchifyVideo(mixed, kH3VideoLatentChannels, 1, lh, lw,
                         std::span<float>(condition_rows).subspan(offset),
                         error)) {
        return false;
      }
      condition_blocks.push_back({lh, lw, rotary_time, 1});
      rotary_time += 1.0;
    }
    // 2. Video references
    for (std::size_t v = 0; v < encoded_videos.size(); ++v, ++noise_idx) {
      const EncodedVideoRef& ev = encoded_videos[v];
      PreparedKeyframe tmp;
      tmp.latent = ev.latent;
      std::vector<float> mixed;
      if (!MixKeyframeNoise(tmp, noise->conditions[noise_idx], &mixed, error)) {
        return false;
      }
      const std::size_t rows_for_cond =
          static_cast<std::size_t>(ev.latent_frames) *
          static_cast<std::size_t>(ev.latent_height / 2) *
          static_cast<std::size_t>(ev.latent_width / 2);
      const std::size_t offset = condition_rows.size();
      condition_rows.resize(offset + rows_for_cond * kH3VideoPatchWidth);
      if (!PatchifyVideo(mixed, kH3VideoLatentChannels, ev.latent_frames,
                         ev.latent_height, ev.latent_width,
                         std::span<float>(condition_rows).subspan(offset),
                         error)) {
        return false;
      }
      condition_blocks.push_back(
          {ev.latent_height, ev.latent_width, rotary_time, ev.latent_frames});
      rotary_time += VideoConditionRotarySpan(ev.latent_frames);
    }
    // 3. Keyframes (FL2VA or combined with Ref2VA)
    if (has_ref2va) {
      target_temporal_origin = rotary_time;
      for (std::size_t i = ref_count; i < keyframes.size(); ++i, ++noise_idx) {
        std::vector<float> mixed;
        if (!MixKeyframeNoise(keyframes[i], noise->conditions[noise_idx],
                              &mixed, error)) {
          return false;
        }
        const int lh = keyframes[i].height / kH3VaeSpatialRatio;
        const int lw = keyframes[i].width / kH3VaeSpatialRatio;
        const std::size_t rows_for_cond =
            static_cast<std::size_t>(lh / 2) * static_cast<std::size_t>(lw / 2);
        const std::size_t offset = condition_rows.size();
        condition_rows.resize(offset + rows_for_cond * kH3VideoPatchWidth);
        if (!PatchifyVideo(mixed, kH3VideoLatentChannels, 1, lh, lw,
                           std::span<float>(condition_rows).subspan(offset),
                           error)) {
          return false;
        }
      }
      if (has_keyframes) {
        const std::size_t base_tokens =
            static_cast<std::size_t>(std::llround(target_temporal_origin));
        auto anchors = KeyframeAnchorTimes(
            base_tokens, geometry->temporal.video_latent_frames,
            !request.first_frame_path.empty(), !request.last_frame_path.empty(),
            error);
        if (!anchors.has_value()) {
          return false;
        }
        for (const double anchor_time : *anchors) {
          condition_blocks.push_back(
              {geometry->latent_height, geometry->latent_width, anchor_time, 1});
        }
      }
    } else if (has_keyframes) {
      for (std::size_t i = 0; i < keyframes.size(); ++i, ++noise_idx) {
        std::vector<float> mixed;
        if (!MixKeyframeNoise(keyframes[i], noise->conditions[noise_idx],
                              &mixed, error)) {
          return false;
        }
        const int lh = keyframes[i].height / kH3VaeSpatialRatio;
        const int lw = keyframes[i].width / kH3VaeSpatialRatio;
        const std::size_t rows_for_cond =
            static_cast<std::size_t>(lh / 2) * static_cast<std::size_t>(lw / 2);
        const std::size_t offset = condition_rows.size();
        condition_rows.resize(offset + rows_for_cond * kH3VideoPatchWidth);
        if (!PatchifyVideo(mixed, kH3VideoLatentChannels, 1, lh, lw,
                           std::span<float>(condition_rows).subspan(offset),
                           error)) {
          return false;
        }
      }
      auto anchors = KeyframeAnchorTimes(
          embedding.tokens, geometry->temporal.video_latent_frames,
          !request.first_frame_path.empty(), !request.last_frame_path.empty(),
          error);
      if (!anchors.has_value()) {
        return false;
      }
      anchor_temporal = std::move(*anchors);
    }
    }
    if (has_checkpoint) {
      if (!SavePromptCheckpoint(request.checkpoint_directory, embedding,
                                condition_rows, condition_blocks,
                                anchor_temporal, target_temporal_origin,
                                *noise, error) ||
          !SaveCheckpointMeta(
              request.checkpoint_directory,
              MakeCheckpointMeta(parameters, request.seed, prompt_sha256,
                                 CheckpointStage::kPrompt, 0),
              error)) {
        return false;
      }
    }
  }

  VideoFrames frames;
  if (resumed_stage >= CheckpointStage::kVideoVae) {
    if (!LoadVideoVaeCheckpoint(request.checkpoint_directory, &frames, error)) {
      return false;
    }
    if (resumed_stage == CheckpointStage::kVideoVae &&
        has_checkpoint &&
        std::filesystem::exists(request.checkpoint_directory / "denoise.bin")) {
      if (LoadDenoiseCheckpoint(request.checkpoint_directory, &denoise_cp,
                                error)) {
        noise->video = denoise_cp.video_latent;
        noise->audio = denoise_cp.audio_latent;
      } else {
        return false;
      }
    }
    Report(progress, progress_opaque, "denoiser", parameters.evaluations,
           parameters.evaluations);
    Report(progress, progress_opaque, "video_vae", 1, 1);
  } else {
    int start_step = 0;
    if (resumed_stage == CheckpointStage::kDenoise &&
        LoadDenoiseCheckpoint(request.checkpoint_directory, &denoise_cp, error)) {
      start_step = std::clamp(denoise_cp.completed_steps, 0,
                              parameters.evaluations);
      if (start_step > 0) {
        noise->video = denoise_cp.video_latent;
        noise->audio = denoise_cp.audio_latent;
      }
    }
    if (start_step < parameters.evaluations) {
      std::vector<std::uint8_t> tags =
          !embedding.tags.empty()
              ? embedding.tags
              : std::vector<std::uint8_t>(embedding.tokens, 1);
      std::shared_ptr<DitLoras> dit_loras;
      if (!request.loras.empty()) {
        dit_loras = std::make_shared<DitLoras>();
        const auto base_shape = [&](std::string_view name) -> std::vector<int> {
          const TensorDescriptor* desc = inventory->FindTensor(name);
          if (desc == nullptr) {
            return {};
          }
          std::vector<int> shape;
          shape.reserve(desc->shape.size());
          for (const std::uint64_t dim : desc->shape) {
            shape.push_back(static_cast<int>(dim));
          }
          return shape;
        };
        for (const auto& req_lora : request.loras) {
          try {
            auto loaded = LoadMinimaxAdapter(req_lora.path, base_shape);
            if (loaded.lora) {
              for (const auto& patch : loaded.lora->patches()) {
                dit_loras->terms[patch.target].push_back(
                    {&patch, req_lora.scale});
              }
              dit_loras->adapters.push_back(std::move(loaded.lora));
            }
            if (!loaded.lokr_patches.empty()) {
              const std::size_t base_idx = dit_loras->lokr_patches.size();
              for (auto& lp : loaded.lokr_patches) {
                dit_loras->lokr_patches.push_back(std::move(lp));
              }
              for (std::size_t idx = base_idx;
                   idx < dit_loras->lokr_patches.size(); ++idx) {
                const auto& lp = dit_loras->lokr_patches[idx];
                dit_loras->lokr_terms[lp.target].push_back(
                    {&lp, req_lora.scale});
              }
            }
          } catch (const std::exception& exc) {
            if (error != nullptr) {
              *error = "failed to load MiniMax H3 LoRA '" + req_lora.name +
                       "': " + exc.what();
            }
            return false;
          }
        }
      }
      auto denoiser = DenoiserSession::Create(
          *inventory,
          {.geometry = *geometry,
           .evaluations = parameters.evaluations,
           .text_rows = embedding.tokens,
           .active_blocks = parameters.active_blocks,
           .reuse_interval = parameters.reuse_interval,
           .row_parallel_attention = parameters.row_parallel_attention,
           .condition_video_rows = condition_rows.size() / kH3VideoPatchWidth,
           .loras = std::move(dit_loras)},
          {.layer50 = embedding.values,
           .tags = tags,
           .condition_rows = condition_rows,
           .anchor_temporal = anchor_temporal,
           .condition_blocks = condition_blocks,
           .target_temporal_origin = target_temporal_origin},
          cancellation, &local.denoiser, error);
      if (denoiser == nullptr) {
        return false;
      }
      ProgressBridge denoiser_progress{progress, progress_opaque, "denoiser"};
      DenoiseCheckpointContext cp_ctx{request.checkpoint_directory, parameters,
                                      request.seed, prompt_sha256};
      begin = Clock::now();
      if (!denoiser->DenoiseWithCheckpoints(
              noise->video, noise->audio, start_step,
              start_step > 0 ? &denoise_cp : nullptr, cancellation,
              ForwardDenoiserProgress, &denoiser_progress,
              has_checkpoint ? OnDenoiseStepCheckpoint : nullptr, &cp_ctx,
              &local.denoiser, error)) {
        return false;
      }
      local.denoiser_ms = Milliseconds(begin, Clock::now());
      denoiser.reset();
    } else {
      Report(progress, progress_opaque, "denoiser", parameters.evaluations,
             parameters.evaluations);
    }
    if (!request.latents_directory.empty()) {
      Report(progress, progress_opaque, "latents", 0, 1);
      const std::span<const unsigned char> prompt_bytes(
          reinterpret_cast<const unsigned char*>(request.prompt.data()),
          request.prompt.size());
      const std::span<const unsigned char> conditioning_bytes(
          reinterpret_cast<const unsigned char*>(embedding.values.data()),
          embedding.values.size() * sizeof(std::uint16_t));
      const LatentManifestContext context{
          .parameters = &parameters,
          .seed = request.seed,
          .prompt_sha256 = Sha256(prompt_bytes),
          .conditioning_sha256 = Sha256(conditioning_bytes),
      };
      if (!WriteLatentsAtomic(request.latents_directory, *geometry, noise->video,
                              noise->audio, &context, cancellation, error)) {
        return false;
      }
      Report(progress, progress_opaque, "latents", 1, 1);
    }

    begin = Clock::now();
    auto video_decoder =
        VideoVaeDecoder::Create(*inventory, {.geometry = *geometry},
                                cancellation, &local.video_vae, error);
    if (video_decoder == nullptr) {
      return false;
    }
    ProgressBridge video_progress{progress, progress_opaque, "video_vae"};
    const bool video_ok =
        parameters.selected_frames.empty()
            ? video_decoder->Decode(noise->video, cancellation,
                                    ForwardVideoProgress, &video_progress,
                                    &frames, &local.video_vae, error)
            : video_decoder->DecodeSelected(
                  noise->video, parameters.selected_frames, cancellation,
                  ForwardVideoProgress, &video_progress, &frames,
                  &local.video_vae, error);
    if (!video_ok) {
      return false;
    }
    local.video_vae_ms = Milliseconds(begin, Clock::now());
    local.first_preview_ms = Milliseconds(total_begin, Clock::now());
    video_decoder.reset();
    if (has_checkpoint) {
      if (!SaveVideoVaeCheckpoint(request.checkpoint_directory, frames, error) ||
          !SaveCheckpointMeta(
              request.checkpoint_directory,
              MakeCheckpointMeta(parameters, request.seed, prompt_sha256,
                                 CheckpointStage::kVideoVae,
                                 parameters.evaluations),
              error)) {
        return false;
      }
    }
  }

  if (!request.frames_directory.empty() &&
      !WriteDiagnosticFrames(request.frames_directory, frames, cancellation,
                             error)) {
    return false;
  }

  if (parameters.decode_audio) {
    AudioWaveform waveform;
    if (resumed_stage >= CheckpointStage::kAudioVae) {
      if (!LoadAudioVaeCheckpoint(request.checkpoint_directory, &waveform,
                                  error)) {
        return false;
      }
      Report(progress, progress_opaque, "audio_vae", 1, 1);
    } else {
      begin = Clock::now();
      auto audio_decoder = AudioVaeDecoder::Create(
          *inventory, geometry->temporal.audio_latent_frames, cancellation,
          &local.audio_vae, error);
      if (audio_decoder == nullptr) {
        return false;
      }
      ProgressBridge audio_progress{progress, progress_opaque, "audio_vae"};
      if (!audio_decoder->Decode(noise->audio, cancellation,
                                 ForwardAudioProgress, &audio_progress,
                                 &waveform, &local.audio_vae, error)) {
        return false;
      }
      local.audio_vae_ms = Milliseconds(begin, Clock::now());
      audio_decoder.reset();
      if (has_checkpoint) {
        if (!SaveAudioVaeCheckpoint(request.checkpoint_directory, waveform,
                                    error) ||
            !SaveCheckpointMeta(
                request.checkpoint_directory,
                MakeCheckpointMeta(parameters, request.seed, prompt_sha256,
                                   CheckpointStage::kAudioVae,
                                   parameters.evaluations),
                error)) {
          return false;
        }
      }
    }
    if (parameters.mux) {
      begin = Clock::now();
      const std::vector<std::uint8_t> rgb24 = ToRgb24(frames);
      if (!WriteSynchronizedMp4(request.output_path, rgb24,
                                static_cast<int>(frames.frame_indices.size()),
                                {.input_width = frames.width,
                                 .input_height = frames.height,
                                 .output_width = parameters.output_width,
                                 .output_height = parameters.output_height,
                                 .fps = kH3FramesPerSecond},
                                waveform.pcm, waveform.samples,
                                waveform.channels, waveform.sample_rate,
                                cancellation, &local.media, error)) {
        return false;
      }
      local.media_ms = Milliseconds(begin, Clock::now());
    }
  }
  if (has_checkpoint) {
    std::error_code ignored;
    std::filesystem::remove_all(request.checkpoint_directory, ignored);
  }
  local.total_ms = Milliseconds(total_begin, Clock::now());
  if (telemetry != nullptr) {
    *telemetry = std::move(local);
  }
  return true;
#endif
}

std::string GenerationTelemetryJson(const GenerationTelemetry& telemetry) {
  std::uint64_t prompt_weight_bytes = 0;
  double prompt_load_ms = 0.0;
  double prompt_prefetch_wait_ms = 0.0;
  double prompt_submit_ms = 0.0;
  double prompt_gpu_ms = 0.0;
  for (const PromptEncoderLayerTelemetry& layer :
       telemetry.prompt_encoder.layers) {
    prompt_weight_bytes += layer.weight_bytes;
    prompt_load_ms += layer.load_ms;
    prompt_prefetch_wait_ms += layer.prefetch_wait_ms;
    prompt_submit_ms += layer.submit_ms;
    prompt_gpu_ms += layer.gpu_ms;
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << "{\n"
         << "  \"schema\": \"gufo.minimax-h3-generation-telemetry.v1\",\n"
         << "  \"prompt_tokens\": " << telemetry.prompt_tokens << ",\n"
         << "  \"internal_width\": " << telemetry.internal_width << ",\n"
         << "  \"internal_height\": " << telemetry.internal_height << ",\n"
         << "  \"output_width\": " << telemetry.output_width << ",\n"
         << "  \"output_height\": " << telemetry.output_height << ",\n"
         << "  \"inventory_ms\": " << telemetry.inventory_ms << ",\n"
         << "  \"inventory_json_bytes\": "
         << telemetry.inventory.json_bytes_read << ",\n"
         << "  \"inventory_header_bytes\": "
         << telemetry.inventory.safetensors_header_bytes_read << ",\n"
         << "  \"inventory_payload_bytes\": "
         << telemetry.inventory.payload_bytes_read << ",\n"
         << "  \"tokenizer_ms\": " << telemetry.tokenizer_ms << ",\n"
         << "  \"prompt_ms\": " << telemetry.prompt_ms << ",\n"
         << "  \"prompt_weight_bytes\": " << prompt_weight_bytes << ",\n"
         << "  \"prompt_load_ms\": " << prompt_load_ms << ",\n"
         << "  \"prompt_prefetch_wait_ms\": " << prompt_prefetch_wait_ms
         << ",\n"
         << "  \"prompt_submit_ms\": " << prompt_submit_ms << ",\n"
         << "  \"prompt_gpu_ms\": " << prompt_gpu_ms << ",\n"
         << "  \"prompt_peak_device_bytes\": "
         << telemetry.prompt_encoder.peak_device_bytes << ",\n"
         << "  \"prompt_peak_registered_host_bytes\": "
         << telemetry.prompt_encoder.peak_registered_host_bytes << ",\n"
         << "  \"prompt_live_registered_host_bytes_after\": "
         << telemetry.prompt_encoder.live_registered_host_bytes_after << ",\n"
         << "  \"prompt_layers\": [";
  for (std::size_t index = 0; index < telemetry.prompt_encoder.layers.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    const PromptEncoderLayerTelemetry& layer =
        telemetry.prompt_encoder.layers[index];
    output << "{\"layer\": " << layer.layer
           << ", \"weight_bytes\": " << layer.weight_bytes
           << ", \"load_ms\": " << layer.load_ms
           << ", \"prefetch_wait_ms\": " << layer.prefetch_wait_ms
           << ", \"submit_ms\": " << layer.submit_ms
           << ", \"gpu_ms\": " << layer.gpu_ms
           << ", \"dispatches\": " << layer.dispatches << "}";
  }
  output << "],\n"
         << "  \"denoiser_ms\": " << telemetry.denoiser_ms << ",\n"
         << "  \"denoiser_setup_ms\": " << telemetry.denoiser.setup_ms << ",\n"
         << "  \"denoiser_text_refiner_ms\": "
         << telemetry.denoiser.text_refiner_ms << ",\n"
         << "  \"denoiser_adaln_precompute_ms\": "
         << telemetry.denoiser.adaln_precompute_ms << ",\n"
         << "  \"denoiser_core_load_ms\": " << telemetry.denoiser.core_load_ms
         << ",\n"
         << "  \"denoiser_sampler_ms\": " << telemetry.denoiser.sampler_ms
         << ",\n"
         << "  \"denoiser_evaluations\": "
         << telemetry.denoiser.forward_ms.size() << ",\n"
         << "  \"denoiser_forward_ms\": [";
  for (std::size_t index = 0; index < telemetry.denoiser.forward_ms.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << telemetry.denoiser.forward_ms[index];
  }
  output << "],\n"
         << "  \"denoiser_core_weight_bytes\": "
         << telemetry.denoiser.core_weight_bytes << ",\n"
         << "  \"denoiser_persistent_bytes\": "
         << telemetry.denoiser.persistent_bytes << ",\n"
         << "  \"denoiser_scratch_bytes\": " << telemetry.denoiser.scratch_bytes
         << ",\n"
         << "  \"denoiser_cumulative_allocation_bytes\": "
         << telemetry.denoiser.cumulative_allocation_bytes << ",\n"
         << "  \"denoiser_core_gemm_plan_validations\": "
         << telemetry.denoiser.core_gemm_plan_validations << ",\n"
         << "  \"video_vae_ms\": " << telemetry.video_vae_ms << ",\n"
         << "  \"video_vae_load_ms\": " << telemetry.video_vae.load_ms << ",\n"
         << "  \"video_vae_decode_ms\": " << telemetry.video_vae.decode_ms
         << ",\n"
         << "  \"video_vae_tile_ms\": [";
  for (std::size_t index = 0; index < telemetry.video_vae.tile_ms.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << telemetry.video_vae.tile_ms[index];
  }
  output << "],\n"
         << "  \"first_preview_ms\": " << telemetry.first_preview_ms << ",\n"
         << "  \"audio_vae_ms\": " << telemetry.audio_vae_ms << ",\n"
         << "  \"audio_vae_load_ms\": " << telemetry.audio_vae.load_ms << ",\n"
         << "  \"audio_vae_decode_ms\": " << telemetry.audio_vae.decode_ms
         << ",\n"
         << "  \"audio_vae_stage_ms\": [";
  for (std::size_t index = 0; index < telemetry.audio_vae.stage_ms.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << telemetry.audio_vae.stage_ms[index];
  }
  output
      << "],\n"
      << "  \"media_ms\": " << telemetry.media_ms << ",\n"
      << "  \"media_maximum_audio_staging_bytes\": "
      << telemetry.media.maximum_audio_staging_bytes << ",\n"
      << "  \"media_child_status\": " << telemetry.media.child_status << ",\n"
      << "  \"media_cancelled\": "
      << (telemetry.media.cancelled ? "true" : "false") << ",\n"
      << "  \"total_ms\": " << telemetry.total_ms << ",\n"
      << "  \"peak_denoiser_bytes\": " << telemetry.denoiser.peak_live_bytes
      << ",\n"
      << "  \"peak_video_vae_bytes\": " << telemetry.video_vae.peak_live_bytes
      << ",\n"
      << "  \"peak_audio_vae_bytes\": " << telemetry.audio_vae.peak_live_bytes
      << ",\n"
      << "  \"prompt_dispatches\": "
      << telemetry.prompt_encoder.total_dispatches << ",\n"
      << "  \"denoiser_dispatches\": " << telemetry.denoiser.dispatches << ",\n"
      << "  \"denoiser_attention_kernel\": \""
      << (telemetry.denoiser.row_parallel_attention ? "row_parallel" : "scalar")
      << "\",\n"
      << "  \"video_vae_dispatches\": " << telemetry.video_vae.dispatches
      << ",\n"
      << "  \"audio_vae_dispatches\": " << telemetry.audio_vae.dispatches
      << ",\n"
      << "  \"media_video_bytes\": " << telemetry.media.video_bytes << ",\n"
      << "  \"media_audio_bytes\": " << telemetry.media.audio_bytes << ",\n"
      << "  \"minor_page_faults\": "
      << telemetry.denoiser.minor_page_faults +
             telemetry.video_vae.minor_page_faults +
             telemetry.audio_vae.minor_page_faults
      << ",\n"
      << "  \"major_page_faults\": "
      << telemetry.denoiser.major_page_faults +
             telemetry.video_vae.major_page_faults +
             telemetry.audio_vae.major_page_faults
      << ",\n"
      << "  \"swap_bytes\": "
      << std::max({telemetry.denoiser.swap_bytes,
                   telemetry.video_vae.swap_bytes,
                   telemetry.audio_vae.swap_bytes})
      << "\n}\n";
  return output.str();
}

}  // namespace gufo::minimax_h3
