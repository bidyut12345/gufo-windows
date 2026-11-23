#ifndef GUFO_MODELS_MINIMAX_H3_MEDIA_HPP_
#define GUFO_MODELS_MINIMAX_H3_MEDIA_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

struct MediaMuxTelemetry {
  std::uint64_t video_bytes{0};
  std::uint64_t audio_bytes{0};
  std::size_t maximum_audio_staging_bytes{0};
  int child_status{-1};
  bool cancelled{false};
};

struct MediaProbe {
  std::string video_codec;
  int width{0};
  int height{0};
  double video_frame_rate{0.0};
  double video_start_seconds{0.0};
  double video_duration_seconds{0.0};

  std::string audio_codec;
  int audio_channels{0};
  int audio_sample_rate{0};
  double audio_start_seconds{0.0};
  double audio_duration_seconds{0.0};
};

struct MediaVideoOptions {
  int input_width{0};
  int input_height{0};
  int output_width{0};
  int output_height{0};
  int fps{0};
};

// RGB is frame-major, row-major, interleaved RGB24. PCM is channel-major F32.
// Both streams are written concurrently in bounded chunks; no second complete
// uncompressed media buffer or temporary media file is created.
[[nodiscard]] bool WriteSynchronizedMp4(
    const std::filesystem::path& path, std::span<const std::uint8_t> rgb24,
    int frame_count, const MediaVideoOptions& video,
    std::span<const float> channel_major_pcm, int samples, int channels,
    int sample_rate, const CancellationToken* cancellation,
    MediaMuxTelemetry* telemetry, std::string* error = nullptr);

[[nodiscard]] bool WriteSynchronizedMp4(
    const std::filesystem::path& path, std::span<const std::uint8_t> rgb24,
    int frame_count, int width, int height, int fps,
    std::span<const float> channel_major_pcm, int samples, int channels,
    int sample_rate, const CancellationToken* cancellation,
    MediaMuxTelemetry* telemetry, std::string* error = nullptr);

struct DecodedReferenceVideo {
  int width{0};
  int height{0};
  int frame_count{0};
  std::vector<std::uint8_t> rgb24;  // frame-major RGB24 at 24 fps
};

[[nodiscard]] bool DecodeReferenceVideo(const std::filesystem::path& path,
                                        int max_frames,
                                        DecodedReferenceVideo* out,
                                        std::string* error = nullptr);

[[nodiscard]] bool ProbeMediaFile(const std::filesystem::path& path,
                                  MediaProbe* probe,
                                  std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_MEDIA_HPP_
