# Video Checkpoint and Resume Fields

Add optional `checkpoint_directory` and `resume_from_checkpoint` fields to `gufo::ext::VideoRequest` in `src/ext/video.hpp` so callers can persist stage/step checkpoints and resume paused video generation.
