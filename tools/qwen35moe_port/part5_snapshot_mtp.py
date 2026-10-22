"""Executor transform part 5: Rollback, snapshots, MtpForward/MtpBody."""

PATH = "src/models/qwen35moe/kernels/rocm/executor.cpp"

text = open(PATH, "rb").read().decode("utf8").replace("\r\n", "\n")


def replace_one(text, old, new):
    if old not in text:
        print("MISSING:\n" + old[:300])
        raise SystemExit(1)
    return text.replace(old, new, 1)


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


# ---- Rollback: recurrent state only ----
text = replace_one(
    text,
    """  if (keep < n) {
    const std::size_t conv_elems =
        static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
    const std::size_t slot = keep - 1;
    for (auto& l : session.linear_) {
      if (l.state == nullptr) {
        continue;
      }
      RestoreGdnState(l.state, l.state_snapshots, keep, c.ssm_num_k_heads,
                      c.ssm_num_v_heads, stream_);
      if (!Check(hipGetLastError(), "state rollback", error_msg) ||
          !Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots.rows[slot],
                                conv_elems * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 "conv rollback", error_msg)) {
        return false;
      }
    }
    if (session.ple_history_ != nullptr) {
      const std::size_t hist =
          static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
      if (!Check(hipMemcpyAsync(
                     session.ple_history_, session.ple_snapshots_.rows[slot],
                     hist * sizeof(float), hipMemcpyDeviceToDevice, stream_),
                 "PLE rollback", error_msg)) {
        return false;
      }
      session.ngram_ = session.ngram_snapshots_[slot];
    }
    session.position_ = session.spec_base_ + keep;
    // Pooled block keys past the kept prefix are stale; they are rebuilt
    // from the raw keys when needed.
    session.blocks_ =
        c.compress_ratio == 0
            ? 0
            : std::min(session.blocks_, session.position_ / c.compress_ratio);
  }""",
    """  if (keep < n) {
    const std::size_t conv_elems =
        static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
    const std::size_t slot = keep - 1;
    for (auto& l : session.linear_) {
      if (l.state == nullptr) {
        continue;
      }
      RestoreGdnState(l.state, l.state_snapshots, keep, c.ssm_num_k_heads,
                      c.ssm_num_v_heads, stream_);
      if (!Check(hipGetLastError(), "state rollback", error_msg) ||
          !Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots.rows[slot],
                                conv_elems * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 "conv rollback", error_msg)) {
        return false;
      }
    }
    session.position_ = session.spec_base_ + keep;
  }""",
)

# ---- Snapshot header, maker, geometry check ----
text = replace_one(
    text,
    """constexpr std::array<char, 8> kSnapshotMagic{'Q', 'F', 'N', 'S',
                                             'N', 'A', 'P', '4'};""",
    """constexpr std::array<char, 8> kSnapshotMagic{'Q', '3', '5', 'M',
                                             'S', 'N', 'A', 'P'};""",
)
ls, end = cut_span(text, "struct SnapshotHeader {")
text = text[:ls] + """/// Fixed header ahead of the section bytes. It carries every geometry
/// value the section sizes derive from, so a payload of another artifact
/// or executor configuration is rejected before anything is copied.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t num_layers;
  std::uint32_t full_attention_interval;
  std::uint32_t conv_elems;
  std::uint32_t state_elems;
  std::uint32_t kv_row;
  std::uint32_t hidden;
  std::uint32_t has_mtp;
  std::uint32_t position;
  std::uint32_t mtp_position;
  std::uint32_t hidden_rows;
  std::uint32_t image_count;
  std::uint64_t payload_bytes;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);
""" + text[end:]

ls, end = cut_span(text, "SnapshotHeader MakeSnapshotHeader(")
text = text[:ls] + """SnapshotHeader MakeSnapshotHeader(const Config& c, bool has_mtp,
                                  const Session& session) {
  SnapshotHeader h{};
  h.magic = kSnapshotMagic;
  h.num_layers = c.num_layers;
  h.full_attention_interval = c.full_attention_interval;
  h.conv_elems = (c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  h.state_elems = c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim;
  h.kv_row = c.AttentionKvDim();
  h.hidden = c.hidden_size;
  h.has_mtp = has_mtp ? 1 : 0;
  h.position = session.position();
  h.image_count = std::ranges::count_if(
      session.VisionLayout().images,
      [&](const auto& image) { return image.offset < h.position; });
  return h;
}

/// Whether `h` describes this executor's geometry (positions aside).
bool SameGeometry(const SnapshotHeader& h, const SnapshotHeader& mine) {
  return h.magic == mine.magic && h.num_layers == mine.num_layers &&
         h.full_attention_interval == mine.full_attention_interval &&
         h.conv_elems == mine.conv_elems && h.state_elems == mine.state_elems &&
         h.kv_row == mine.kv_row && h.hidden == mine.hidden &&
         h.has_mtp == mine.has_mtp;
}
""" + text[end:]

# ---- WalkSnapshot ----
ls, end = cut_span(text, "template<typename Visit>\nstd::uint64_t Executor::WalkSnapshot(")
text = text[:ls] + """template<typename Visit>
std::uint64_t Executor::WalkSnapshot(const SnapshotHeader& h,
                                     const Session* session, Visit&& visit) {
  std::uint64_t offset = sizeof(SnapshotHeader);
  const auto region = [&](void* device, std::uint64_t bytes, const char* what) {
    if (bytes != 0 && !visit(device, offset, bytes, what)) {
      return false;
    }
    offset += bytes;
    return true;
  };
  const auto linear = [&](std::uint32_t il) -> const Session::LinearState* {
    return session != nullptr ? &session->linear_[il] : nullptr;
  };
  const auto attention =
      [&](std::uint32_t il) -> const Session::AttentionState* {
    return session != nullptr ? &session->attention_[il] : nullptr;
  };
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) != 0) {
      const auto* l = linear(il);
      if (!region(l != nullptr ? l->conv_state : nullptr,
                  std::uint64_t{h.conv_elems} * sizeof(float), "conv state") ||
          !region(l != nullptr ? l->state : nullptr,
                  std::uint64_t{h.state_elems} * sizeof(float),
                  "recurrent state")) {
        return 0;
      }
    }
  }
  const std::uint64_t kv_bytes =
      std::uint64_t{h.position} * h.kv_row * sizeof(__half);
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) == 0) {
      const auto* at = attention(il);
      if (!region(at != nullptr ? at->k_cache : nullptr, kv_bytes, "K cache") ||
          !region(at != nullptr ? at->v_cache : nullptr, kv_bytes, "V cache")) {
        return 0;
      }
    }
  }
  if (h.has_mtp != 0) {
    const auto* mtp = session != nullptr ? &session->mtp_ : nullptr;
    const std::uint64_t mtp_kv_bytes =
        std::uint64_t{h.mtp_position} * h.kv_row * sizeof(__half);
    if (!region(mtp != nullptr ? mtp->k_cache : nullptr, mtp_kv_bytes,
                "draft K cache") ||
        !region(mtp != nullptr ? mtp->v_cache : nullptr, mtp_kv_bytes,
                "draft V cache") ||
        !region(mtp != nullptr ? mtp->h : nullptr,
                std::uint64_t{h.hidden} * sizeof(float), "draft residual") ||
        !region(mtp != nullptr ? mtp->target_hidden : nullptr,
                std::uint64_t{h.hidden_rows} * h.hidden * sizeof(float),
                "kept trunk rows")) {
      return 0;
    }
  }
  return offset +
         std::uint64_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
}
""" + text[end:]

# ---- SnapshotBytes / SaveSnapshot header fills ----
text = replace_one(
    text,
    """  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(""",
    """  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(""",
)
text = replace_one(
    text,
    """  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  h.ngram_prev = session.ngram_.prev;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);""",
    """  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);""",
)

# ---- RestoreSnapshot validation ----
text = replace_one(
    text,
    """  const std::uint32_t max_blocks =
      h.compress_ratio == 0 ? 0 : h.position / h.compress_ratio;
  if (h.image_count > 256 || h.position == 0 ||
      h.position > session.max_context_ || h.blocks > max_blocks ||
      h.mtp_position > h.position ||
      h.mtp_blocks > h.mtp_position / h.compress_ratio ||
      h.mtp_position - h.mtp_blocks * h.compress_ratio >
          session.index_capacity_ ||
      h.hidden_rows > h.position || h.hidden_rows > options_.max_speculative ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  if (h.position - h.blocks * h.compress_ratio > session.index_capacity_) {
    AssignError(error_msg, "unpooled indexer rows exceed the ring capacity");
    return false;
  }
  if (h.payload_bytes != payload.size() ||""",
    """  if (h.image_count > 256 || h.position == 0 ||
      h.position > session.max_context_ || h.mtp_position > h.position ||
      h.hidden_rows > h.position || h.hidden_rows > options_.max_speculative ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  if (h.payload_bytes != payload.size() ||""",
)
text = replace_one(
    text,
    """  session.RestoreVisionLayout(layout, stream_);
  session.position_ = h.position;
  session.blocks_ = h.blocks;
  session.mtp_.position = h.mtp_position;
  session.mtp_.blocks = h.mtp_blocks;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;
  session.ngram_.prev = h.ngram_prev;""",
    """  session.RestoreVisionLayout(layout, stream_);
  session.position_ = h.position;
  session.mtp_.position = h.mtp_position;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;""",
)

# ---- MtpForward: no sparse plumbing ----
text = replace_one(
    text,
    """  ++session.mutation_epoch_;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  control_host_->mtp_blocks = session.mtp_.blocks;
  const auto& c = config();
  const bool sparse = pos + n > c.indexer_top_k;
  const auto complete = (pos + n) / c.compress_ratio;
  const auto pool = sparse ? complete - session.mtp_.blocks : 0;
  const auto graph_pool = n / c.compress_ratio + 1;
  const bool graph = output.trace == nullptr && hidden_source == nullptr &&
                     n <= kVecBatch && pool <= graph_pool &&
                     pos + 1 >= session.VisionLayout().PrefixLength();
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43) |
      (std::uint64_t{sparse} << 44);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg,
                   graph ? graph_pool : pool, hidden_source, output.trace);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (output.token != nullptr) {
    *output.token = *mtp_token_host_;
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
  }
  session.mtp_.position = pos + n;
  if (sparse)
    session.mtp_.blocks = complete;
  return true;
}""",
    """  ++session.mutation_epoch_;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  const auto& c = config();
  const bool graph = output.trace == nullptr && hidden_source == nullptr &&
                     n <= kVecBatch &&
                     pos + 1 >= session.VisionLayout().PrefixLength();
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg, hidden_source,
                   output.trace);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (output.token != nullptr) {
    *output.token = *mtp_token_host_;
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
  }
  session.mtp_.position = pos + n;
  return true;
}""",
)

open(PATH, "wb").write(text.replace("\n", "\r\n").encode("utf8"))
print("part 5 ok")
