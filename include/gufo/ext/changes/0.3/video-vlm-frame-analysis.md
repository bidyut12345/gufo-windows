# Video VLM Frame Analysis

Additive change to `src/ext/video.hpp` (forwarded by `include/gufo/ext/video.hpp`):

- Added `gufo::ext::VideoAnalyzeRequest` for multimodal frame-analysis requests.
- Added optional `analyze_frames` callback on `gufo::ext::VideoProvider` (defaults to `nullptr`), implemented by `minimax-h3` using its 64-layer Qwen3-VL text encoder and vision tower.
