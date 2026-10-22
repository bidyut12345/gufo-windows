"""Executor transform part 6: MtpBody rewrite for the plain-residual draft."""

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


new_body = '''bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       bool token, bool candidates, std::string* error_msg,
                       const float* hidden_source, MtpTrace* trace) const {
  const Config& c = config();
  const DeviceLayer& l = model_->mtp();
  const std::size_t final_row = static_cast<std::size_t>(n - 1) * c.hidden_size;
  const auto copy_trace = [&](const float* source,
                              std::span<float> destination) {
    return destination.empty() ||
           Check(hipMemcpyAsync(destination.data(), source,
                                destination.size_bytes(), hipMemcpyDeviceToHost,
                                stream_),
                 "MTP trace download", error_msg);
  };
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
              s_.tokens, s_.mtp_embd, n, c.hidden_size, 1, stream_);
  // MTP embeds shifted token IDs. Visual information arrives in the trunk
  // hidden stream; image embeddings belong only to the target input.
  RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n, c.hidden_size,
              1, c.rms_eps, stream_);
  // The hidden input: kept trunk rows from `hidden_row`, or the block's
  // own carried residual.
  MtpHidden(hidden_source ? hidden_source : session.mtp_.target_hidden,
            session.mtp_.h, &session.control_->hidden_row, s_.mtp_h, n,
            c.hidden_size, stream_);
  RmsNormRows(s_.mtp_h, l.nextn_hnorm.f32(), s_.mtp_h, n, c.hidden_size, 1,
              c.rms_eps, stream_);
  if (trace && !copy_trace(s_.mtp_h + final_row, trace->normalized_hidden))
    return false;
  if (!Dense(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n, error_msg) ||
      !Dense(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n, error_msg)) {
    return false;
  }
  MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, n, c.hidden_size, 1, stream_);
  if (trace && !copy_trace(s_.mtp_res + final_row, trace->fused))
    return false;
  Session::AttentionState attn;
  attn.rope = session.vision_input_.rope();
  attn.k_cache = session.mtp_.k_cache;
  attn.v_cache = session.mtp_.v_cache;
  // The draft block's attention runs at its own position.
  // Catch-up exports KV for every row but only carries its final residual.
  // Preserve that row's original query tile; earlier attention results
  // never contribute to the carried state.
  const bool last_only =
      !token && !candidates && n > 32 && control_host_->hidden_row >= 0;
  // Keep the final 128-column tile (and its predecessor for short tails).
  // At least 96 rows retain the wide projection arithmetic. Attention
  // still writes every KV row before the scratch view is narrowed.
  const auto skipped = last_only && n >= 224 ? (n - 96) / 128 * 128 : 0U;
  RmsNormRows(s_.mtp_res, l.attn_norm.f32(), s_.mixed, n, c.hidden_size, 1,
              c.rms_eps, stream_);
  if (!Attention(l, attn, s_.mixed, s_.block_out, n,
                 &session.control_->mtp_position, pos, error_msg, last_only,
                 false, skipped == 0)) {
    return false;
  }
  struct RestoreScratch {
    const Executor* executor;
    Scratch scratch;
    bool changed;
    ~RestoreScratch() {
      if (changed)
        executor->UseScratch(scratch);
    }
  } restore{this, s_, skipped != 0};
  const auto tail_rows = n - skipped;
  const auto tail_last = static_cast<std::size_t>(tail_rows - 1) * c.hidden_size;
  if (skipped != 0) {
    auto tail = RowScratch(s_, skipped);
    UseScratch(tail);
    if (!Dense(l.attn_out, s_.ctx, s_.block_out, tail_rows, error_msg))
      return false;
  }
  Combine(s_.mtp_res, nullptr, tail_rows);
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->attention))
    return false;
  if (!session.CheckCancellation(error_msg))
    return false;
  RmsNormRows(s_.mtp_res, l.post_attention_norm.f32(), s_.mixed, tail_rows,
              c.hidden_size, 1, c.rms_eps, stream_);
  if ((trace && !copy_trace(s_.mixed + static_cast<std::size_t>(tail_rows - 1) *
                                          c.hidden_size,
                            trace->ffn_input)) ||
      !Moe(l, s_.mixed, s_.block_out, tail_rows, error_msg, last_only)) {
    return false;
  }
  Combine(s_.mtp_res, nullptr, tail_rows);
  if (trace &&
      !copy_trace(s_.block_out +
                      static_cast<std::size_t>(tail_rows - 1) * c.hidden_size,
                  trace->ffn_output))
    return false;
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->hidden))
    return false;
  if (trace && !trace->head.empty()) {
    RmsNormRows(s_.mtp_res + tail_last, l.nextn_head_norm.f32(), s_.mixed, 1,
                c.hidden_size, 1, c.rms_eps, stream_);
    if (!copy_trace(s_.mixed, trace->head))
      return false;
  }
  const float* last = s_.mtp_res + tail_last;
  if (!Check(hipMemcpyAsync(session.mtp_.h, last, c.hidden_size * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return (!token && !candidates) ||
         MtpHead(l.nextn_head_norm, last, token, candidates, error_msg);
}
'''

ls, end = cut_span(text, "bool Executor::MtpBody(")
text = text[:ls] + new_body + text[end:]
open(PATH, "wb").write(text.replace("\n", "\r\n").encode("utf8"))
print("part 6 ok")
