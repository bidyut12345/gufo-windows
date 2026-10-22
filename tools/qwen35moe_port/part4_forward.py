"""Executor transform part 4: Forward / ForwardBody rewrite."""

PATH = "src/models/qwen35moe/kernels/rocm/executor.cpp"

text = open(PATH, "rb").read().decode("utf8").replace("\r\n", "\n")


def cut_span(text, start_marker):
    start = text.index(start_marker)
    line_start = text.rfind("\n", 0, start) + 1
    while True:
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        prev = text[prev_start:line_start - 1]
        if prev.lstrip().startswith("///") or prev.lstrip().startswith("//"):
            line_start = prev_start
        else:
            break
    brace = text.index("{", start)
    depth = 0
    i = brace
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return line_start, i + 1


new_forward = '''bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, ForwardMode mode,
                       std::string* error_msg) const {
  const bool speculative = mode == ForwardMode::kVerify;
  PrefillPhase phase(mode == ForwardMode::kPrefill);
  selected_logits_ = nullptr;
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > options_.max_batch ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  if (n_logits > n || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  if (!session.CheckCancellation(error_msg))
    return false;
  if (session.mtp_enabled_ && prefill_phase &&
      session.mtp_.position != start_pos) {
    AssignError(error_msg,
                "MTP must consume the preceding frontier before prefill");
    return false;
  }
  if (speculative && !EnsureRollback(session, n - 1, error_msg))
    return false;
  ++session.mutation_epoch_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens and the control block.
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->mtp_position = session.mtp_.position;
  control_host_->hidden_row = -1;
  // Decode-sized batches replay as graphs.
  const bool graph =
      !prefill_phase && n <= kVecBatch &&
      session.position_ >= session.VisionLayout().PrefixLength();
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (std::uint64_t{logits != nullptr} << 34);
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, logits != nullptr, speculative,
                       start_pos, 0, c.num_layers, error_msg);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (n_logits > 0 && logits != nullptr) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }

  session.position_ += n;
  if (session.mtp_enabled_ && prefill_phase && n > 1) {
    // Every successor except the last one is already known. Consume it while
    // the trunk residual is still in shared scratch. MtpBody reads that
    // residual into mtp_h before reusing res for the predictor output.
    PrefillPhase draft_phase(false);
    if (!MtpForward(session, tokens.subspan(1), 0, {}, error_msg, s_.res))
      return false;
  }
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool download_logits,
                           bool speculative, std::uint32_t start_pos,
                           std::uint32_t first_layer, std::uint32_t end_layer,
                           std::string* error_msg) const {
  const Config& c = config();
  if (first_layer == 0) {
    if (!Check(hipMemcpyAsync(session.control_, control_host_,
                              sizeof(Session::Control), hipMemcpyHostToDevice,
                              stream_),
               "control upload", error_msg) ||
        !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "token upload", error_msg)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                s_.tokens, s_.res, n, c.hidden_size, 1, stream_);
    session.vision_input_.Inject(s_.res, start_pos, n, c.hidden_size, 1,
                                 stream_);
  }
  const auto& layers = model_->layers();
  for (std::uint32_t il = first_layer; il < end_layer; ++il) {
    if (!session.CheckCancellation(error_msg))
      return false;
    const DeviceLayer& l = layers[il];
    RmsNormRows(s_.res, l.attn_norm.f32(), s_.mixed, n, c.hidden_size, 1,
                c.rms_eps, stream_);
    if (l.linear) {
      if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           speculative, error_msg)) {
        return false;
      }
    } else if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                          &session.control_->position, start_pos, error_msg)) {
      return false;
    }
    Combine(s_.res, nullptr, n);

    RmsNormRows(s_.res, l.post_attention_norm.f32(), s_.mixed, n,
                c.hidden_size, 1, c.rms_eps, stream_);
    if (!Moe(l, s_.mixed, s_.block_out, n, error_msg)) {
      return false;
    }
    Combine(s_.res, nullptr, n);
  }
  if (end_layer < c.num_layers) {
    return true;
  }
  // Prefill catches up immediately from shared scratch. Only the short tail
  // needed across calls belongs to this session.
  const auto kept = std::min(n, options_.max_speculative);
  if (session.mtp_enabled_ &&
      !Check(hipMemcpyAsync(
                 session.mtp_.target_hidden,
                 s_.res + static_cast<std::size_t>(n - kept) * c.hidden_size,
                 static_cast<std::size_t>(kept) * c.hidden_size * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    PrefillPhase head_phase(false);
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    RmsNormRows(s_.res + skip * c.hidden_size, model_->output_norm().f32(),
                s_.mixed, n_logits, c.hidden_size, 1, c.rms_eps, stream_);
    if (!Dense(model_->output(), s_.mixed, s_.logits, n_logits, error_msg)) {
      return false;
    }
    if (download_logits &&
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) *
                                  c.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  return true;
}
'''

ls, end = cut_span(text, "bool Executor::Forward(")
_, end_fb = cut_span(text, "bool Executor::ForwardBody(")
assert end <= end_fb
text = text[:ls] + new_forward + text[end_fb:]
open(PATH, "wb").write(text.replace("\n", "\r\n").encode("utf8"))
print("part 4 ok")
