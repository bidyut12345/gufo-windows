"""Transform qwen38_flash_next executor.cpp to qwen35moe.

Cut/rewrite list: PLE and hyper-connection stages go; the layer loop norms
into `mixed` and adds block outputs back; attention runs dense only; the
snapshot header shrinks to the plain geometry. Run from the repo root.
"""
import sys

SRC = "src/models/qwen38_flash_next/kernels/rocm/executor.cpp"
DST = "src/models/qwen35moe/kernels/rocm/executor.cpp"


def cut_span(text, start_marker, end_marker):
    """Removes [start_marker line .. line before end_marker], with the
    start's preceding /// and // comment lines."""
    start = text.index(start_marker)
    line_start = text.rfind("\n", 0, start) + 1
    while True:
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        prev = text[prev_start:line_start - 1]
        if prev.lstrip().startswith("///") or prev.lstrip().startswith("//"):
            line_start = prev_start
        else:
            break
    end = text.index(end_marker)
    end_line = text.rfind("\n", 0, end) + 1
    return text[:line_start] + text[end_line:]


def replace_one(text, old, new):
    if old not in text:
        print("MISSING ANCHOR:\n" + old[:400])
        sys.exit(1)
    return text.replace(old, new, 1)


def main():
    text = open(SRC, "rb").read().decode("utf8").replace("\r\n", "\n")
    text = text.replace("qwen38_flash_next", "qwen35moe")

    # ---- Session::Reset: no PLE history, no n-gram window, no blocks ----
    text = replace_one(
        text,
        """void Session::Reset() {
  TrimRollback(0);
  position_ = 0;
  spec_tokens_ = 0;
  ngram_.Reset();
  mtp_.position = 0;
  mtp_.blocks = 0;
  const Config& c = owner_->config();
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      (void)hipMemset(l.conv_state, 0,
                      static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                          c.SsmConvChannels() * sizeof(float));
      (void)hipMemset(l.state, 0,
                      static_cast<std::size_t>(c.ssm_num_v_heads) *
                          c.ssm_head_dim * c.ssm_head_dim * sizeof(float));
    }
  }
  blocks_ = 0;
  if (ple_history_ != nullptr) {
    (void)hipMemset(ple_history_, 0,
                    static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim() *
                        sizeof(float));
  }
}""",
        """void Session::Reset() {
  TrimRollback(0);
  position_ = 0;
  spec_tokens_ = 0;
  mtp_.position = 0;
  const Config& c = owner_->config();
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      (void)hipMemset(l.conv_state, 0,
                      static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                          c.SsmConvChannels() * sizeof(float));
      (void)hipMemset(l.state, 0,
                      static_cast<std::size_t>(c.ssm_num_v_heads) *
                          c.ssm_head_dim * c.ssm_head_dim * sizeof(float));
    }
  }
}""",
    )

    # ---- TrimRollback: GDN snapshots only ----
    text = replace_one(
        text,
        """    for (auto& layer : linear_) {
      layer.conv_snapshots.rows[row] = nullptr;
      layer.state_snapshots.rows[row] = nullptr;
    }
    ple_snapshots_.rows[row] = nullptr;
  }
  rollback_allocations_.resize(depth);
  rollback_depth_ = depth;
  rollback_bytes_ =
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_,
                           depth) -
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_, 0);
  ngram_snapshots_.resize(depth);
}""",
        """    for (auto& layer : linear_) {
      layer.conv_snapshots.rows[row] = nullptr;
      layer.state_snapshots.rows[row] = nullptr;
    }
  }
  rollback_allocations_.resize(depth);
  rollback_depth_ = depth;
  rollback_bytes_ =
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_,
                           depth) -
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_, 0);
}""",
    )

    # ---- Executor::Create: signature, scratch, allocations ----
    text = replace_one(
        text,
        """std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           NgramTable* ngram, Options options,
                                           std::string* error_msg) {""",
        """std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           Options options,
                                           std::string* error_msg) {""",
    )
    text = replace_one(
        text,
        """  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->ngram_ = ngram;
  e->options_ = options;""",
        """  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->options_ = options;""",
    )
    text = replace_one(
        text,
        """  const std::size_t hc_dim = c.HcDim();
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;""",
        """  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;""",
    )
    text = replace_one(
        text,
        """  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hc_dim);
  s.xn = f32(T * hc_dim);
  s.xn_half = Alloc<__half>(a, T * hc_dim, error_msg);
  s.xn_q8t = Alloc<std::uint8_t>(a, Q8TiledBytes(T, hc_dim), error_msg);
  s.lo = f32(T * c.hc_low_rank);
  s.hc_gate = f32(T * hc_dim);
  s.mixed = f32(T * hidden);
  s.inject = f32(T * c.hc_count * HcInjectParts(hidden));
  s.block_out = f32(T * hidden);
  // Stacked and separate SSM projections are mutually exclusive. MTP's
  // projected embedding is consumed before attention and reuses this space.
  s.qkvz = f32(T * std::max<std::size_t>({c.SsmConvChannels() + c.SsmValueDim(),
                                          c.ple_layer >= 0 ? hc_dim : 0}));""",
        """  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hidden);
  s.mixed = f32(T * hidden);
  s.block_out = f32(T * hidden);
  // Stacked and separate SSM projections are mutually exclusive. MTP's
  // projected embedding is consumed before attention and reuses this space.
  s.qkvz = f32(T * (c.SsmConvChannels() + c.SsmValueDim()));""",
    )
    text = replace_one(
        text,
        """  s.qg = f32(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = f32(T * c.AttentionQDim());
  s.attn_gate = f32(T * c.AttentionQDim());
  s.k = f32(T * c.AttentionKvDim());
  s.v = f32(T * c.AttentionKvDim());
  s.iq = f32(T * c.indexer_heads * c.indexer_head_dim);
  s.ik = f32(T * c.indexer_head_dim);
  const std::uint32_t max_blocks =
      (c.context_length + c.compress_ratio - 1) / c.compress_ratio;
  e->mask_words_ = (max_blocks + 31) / 32;
  s.mask = Alloc<std::uint32_t>(a, T * e->mask_words_, error_msg);
  s.scores =
      f32(static_cast<std::size_t>(e->select_chunk_) * e->mask_words_ * 32);
  s.ctx = f32(T * c.AttentionQDim());
  s.attn_partials = f32(static_cast<std::size_t>(kVecBatch) * c.num_heads *
                        kAttnSplits * (c.head_dim + 2));
  if (c.ple_layer >= 0) {
    s.ple_emb = f32(T * c.PleEmbeddingDim());
    // PLE finishes before this layer's mixer/SSM. Its gate consumes key
    // and query before normalization/convolution reuse those two buffers.
    // The gated input stays separate until PleInject consumes both outputs.
    s.ple_key = s.hc_gate;
    s.ple_value = s.block_out;
    s.ple_query = s.xn;
    s.ple_gated = s.qkvz;
    s.ple_norm = s.hc_gate;
    s.ple_conv = s.xn;
    s.ple_history_scratch =
        f32(static_cast<std::size_t>(c.PleConvHistory()) * hc_dim);
    void* pinned = nullptr;
    if (!Check(hipHostMalloc(&pinned, T * c.PleEmbeddingDim() * sizeof(float)),
               "pinned n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_emb_ = static_cast<float*>(pinned);
    e->host_rows_.resize(T * c.ple_heads);
  }
  s.router = f32(T * (c.num_experts + 1));""",
        """  s.qg = f32(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = f32(T * c.AttentionQDim());
  s.attn_gate = f32(T * c.AttentionQDim());
  s.k = f32(T * c.AttentionKvDim());
  s.v = f32(T * c.AttentionKvDim());
  s.ctx = f32(T * c.AttentionQDim());
  s.attn_partials = f32(static_cast<std::size_t>(kVecBatch) * c.num_heads *
                        kAttnSplits * (c.head_dim + 2));
  s.router = f32(T * (c.num_experts + 1));""",
    )
    text = replace_one(
        text,
        """  // The wide mixer route (F16 norm for the epilogue, tiled Q8 norm for the
  // W8A8 down projection) needs the four-stream geometry, 32-wide blocks and
  // a Q8_0 down projection.
  e->wide_mixer_ = c.hc_count == 4 && c.hidden_size % 32 == 0 &&
                   !model.layers().empty() &&
                   model.layers()[0].hc_ffn.down.type == GgmlType::kQ8_0;
  if (model.has_mtp()) {""",
        """  if (model.has_mtp()) {""",
    )
    text = replace_one(
        text,
        """    // The trunk's kept rows are session-owned. Its transient residual and
    // mixer buffers are free while MTP constructs its input. The split
    // projections consume h/embd before the first mixer rewrites xn/mixed.
    s.mtp_h = s.xn;
    s.mtp_embd = s.mixed;""",
        """    // The trunk's kept rows are session-owned. Its transient residual and
    // mixed buffers are free while MTP constructs its input. The split
    // projections consume h/embd before attention rewrites res/mixed.
    s.mtp_h = s.mixed;
    s.mtp_embd = s.block_out;""",
    )

    # ---- CreateSession: caches without indexer rings or PLE history ----
    text = replace_one(
        text,
        """  s->max_context_ = max_context;
  // Before sparse attention starts, pooling can lag by indexer_top_k
  // rows. Afterwards only an incomplete block precedes the current batch.
  // Completed block keys remain in block_k; their raw rows are dead.
  s->index_capacity_ = IndexerCapacity(c, options_.max_batch, max_context);
  s->linear_.resize(c.num_layers);""",
        """  s->max_context_ = max_context;
  s->linear_.resize(c.num_layers);""",
    )
    text = replace_one(
        text,
        """      at.v_cache =
          Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                        error_msg, &s->allocated_bytes_);
      at.index_k = Alloc<float>(
          a, static_cast<std::size_t>(s->index_capacity_) * c.indexer_head_dim,
          error_msg, &s->allocated_bytes_);
      at.block_k = Alloc<__half>(
          a,
          static_cast<std::size_t>(max_context / c.compress_ratio + 1) *
              c.indexer_head_dim,
          error_msg, &s->allocated_bytes_);
    }
  }
  if (c.ple_layer >= 0) {
    const std::size_t hist =
        static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    s->ple_history_ = Alloc<float>(a, hist, error_msg, &s->allocated_bytes_);
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);""",
        """      at.v_cache =
          Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                        error_msg, &s->allocated_bytes_);
    }
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);""",
    )
    text = replace_one(
        text,
        """    s->mtp_.v_cache =
        Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                      error_msg, &s->allocated_bytes_);
    s->mtp_.index_k =
        Alloc<float>(a, std::size_t{s->index_capacity_} * c.indexer_head_dim,
                     error_msg, &s->allocated_bytes_);
    s->mtp_.block_k = Alloc<__half>(
        a, std::size_t{max_context / c.compress_ratio + 1} * c.indexer_head_dim,
        error_msg, &s->allocated_bytes_);
    s->mtp_.h = Alloc<float>(a, c.HcDim(), error_msg, &s->allocated_bytes_);
  }""",
        """    s->mtp_.v_cache =
        Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                      error_msg, &s->allocated_bytes_);
    s->mtp_.h = Alloc<float>(a, c.hidden_size, error_msg, &s->allocated_bytes_);
  }""",
    )

    # ---- EnsureRollback / SessionBytes: recurrent state only ----
    text = replace_one(
        text,
        """  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t linear =
      c.num_layers - c.num_layers / c.full_attention_interval;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  session.rollback_allocations_.reserve(depth);
  session.ngram_snapshots_.resize(depth);
  for (auto row = session.rollback_depth_; row < depth; ++row) {""",
        """  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t linear =
      c.num_layers - c.num_layers / c.full_attention_interval;
  session.rollback_allocations_.reserve(depth);
  for (auto row = session.rollback_depth_; row < depth; ++row) {""",
    )
    text = replace_one(
        text,
        """    const auto state = GdnRollbackRowFloats(row, c.ssm_num_k_heads,
                                            c.ssm_num_v_heads, c.ssm_head_dim);
    const auto bytes = (linear * (conv + state) + ple) * sizeof(float);""",
        """    const auto state = GdnRollbackRowFloats(row, c.ssm_num_k_heads,
                                            c.ssm_num_v_heads, c.ssm_head_dim);
    const auto bytes = linear * (conv + state) * sizeof(float);""",
    )
    text = replace_one(
        text,
        """      session.linear_[il].conv_snapshots.rows[row] = next;
      next += conv;
      session.linear_[il].state_snapshots.rows[row] = next;
      next += state;
    }
    if (ple != 0) {
      session.ple_snapshots_.rows[row] = next;
    }
    session.rollback_depth_ = row + 1;""",
        """      session.linear_[il].conv_snapshots.rows[row] = next;
      next += conv;
      session.linear_[il].state_snapshots.rows[row] = next;
      next += state;
    }
    session.rollback_depth_ = row + 1;""",
    )
    text = replace_one(
        text,
        """  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const std::size_t kv =
      2 * std::size_t{max_context} * c.AttentionKvDim() * sizeof(__half);
  const std::size_t index =
      std::size_t{IndexerCapacity(c, options_.max_batch, max_context)} *
          c.indexer_head_dim * sizeof(float) +
      std::size_t{max_context / c.compress_ratio + 1} * c.indexer_head_dim *
          sizeof(__half);
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : state + (rollback_depth - 1) *
                        GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                             c.ssm_num_v_heads, c.ssm_head_dim);
  return ((linear * conv + ple) * (rollback_depth + 1) +
          linear * (state + rollback_state)) *
             sizeof(float) +
         attention * (kv + index) + sizeof(Session::Control) +
         (mode == core::SessionMode::kSpeculative
              ? kv + index +
                    std::size_t{options_.max_speculative + 1} * c.HcDim() *
                        sizeof(float)
              : 0);
}""",
        """  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t kv =
      2 * std::size_t{max_context} * c.AttentionKvDim() * sizeof(__half);
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : state + (rollback_depth - 1) *
                        GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                             c.ssm_num_v_heads, c.ssm_head_dim);
  return (linear * conv * (rollback_depth + 1) +
          linear * (state + rollback_state)) *
             sizeof(float) +
         attention * kv + sizeof(Session::Control) +
         (mode == core::SessionMode::kSpeculative
              ? kv + std::size_t{options_.max_speculative + 1} *
                         c.hidden_size * sizeof(float)
              : 0);
}""",
    )

    open(DST, "wb").write(text.replace("\n", "\r\n").encode("utf8"))
    print("part 1 ok")


if __name__ == "__main__":
    main()

def part2():
    text = open(DST, "rb").read().decode("utf8").replace("\r\n", "\n")

    # ---- HcMix and the PLE stages: gone (the layer loop norms inline) ----
    text = cut_span(text, "bool Executor::HcMix(", "bool Executor::PleFetch(")
    text = cut_span(text, "bool Executor::PleFetch(", "bool Executor::LinearAttention(")

    # ---- Attention: dense gated GQA only ----
    start = text.index("bool Executor::Attention(")
    # walk to the function end by brace matching
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
    new_attention = '''bool Executor::Attention(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos, std::string* error_msg,
                         bool last_only, bool projections_ready,
                         bool project_output) const {
  const Config& c = config();
  const std::uint32_t kv_row = c.AttentionKvDim();
  bool prepared = false;
  if (!l.attn_qkv.empty()) {
    if (!projections_ready &&
        !Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg)) {
      return false;
    }
    prepared = PrepareAttention(
        s_.qg, l.attn_qkv.rows, l.attn_q_norm.f32(), l.attn_k_norm.f32(),
        s_.q, s_.attn_gate, s.k_cache, s.v_cache, n_tokens, c.num_heads,
        c.num_kv_heads, c.head_dim, c.rotary_dim, pos, c.rope_theta,
        c.rms_eps, stream_, s.rope, prefill_phase);
    if (!prepared) {
      UnpackQGate(s_.qg, l.attn_qkv.rows, s_.q, s_.attn_gate, s_.k, s_.v,
                  n_tokens, c.num_heads, c.head_dim, kv_row, stream_);
    }
  } else {
    Q8Input xq;
    if (!projections_ready &&
        (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
         !Dense(l.attn_q, xq, s_.qg, error_msg) ||
         !Dense(l.attn_k, xq, s_.k, error_msg) ||
         !Dense(l.attn_v, xq, s_.v, error_msg))) {
      return false;
    }
    UnpackQGate(s_.qg, 2 * c.AttentionQDim(), s_.q, s_.attn_gate, nullptr,
                nullptr, n_tokens, c.num_heads, c.head_dim, 0, stream_);
  }
  if (!prepared) {
    RmsNormRows(s_.q, l.attn_q_norm.f32(), s_.q, n_tokens * c.num_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    RmsNormRows(s_.k, l.attn_k_norm.f32(), s_.k, n_tokens * c.num_kv_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    Rope(s_.q, n_tokens, c.num_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope);
    Rope(s_.k, n_tokens, c.num_kv_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope);
    StoreKv(s_.k, s.k_cache, n_tokens, kv_row, pos, stream_);
    StoreKv(s_.v, s.v_cache, n_tokens, kv_row, pos, stream_);
  }
  if (last_only && !Check(hipMemsetAsync(s_.ctx, 0,
                                         static_cast<std::size_t>(n_tokens) *
                                             c.AttentionQDim() * sizeof(float),
                                         stream_),
                          "draft attention output initialization", error_msg)) {
    return false;
  }
  // Wide batches run the fused WMMA kernel (never inside a graph: the kv
  // extent is a host value); the per-token kernel covers the rest.
  if (MatrixRows(n_tokens) &&
      WmmaCausalAttention(s_.q, s_.attn_gate, s.k_cache, s.v_cache, nullptr, 0,
                          s_.ctx, n_tokens, pos ? *pos : 0, c.num_heads,
                          c.num_kv_heads, c.head_dim, 0, stream_, last_only)) {
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  // Narrow batches split each row's key tiles over kAttnSplits blocks so a
  // decode step at depth fills the device.
  const bool split = !MatrixRows(n_tokens);
  rocm::Attention(s_.q, s.k_cache, s.v_cache, nullptr, 0, s_.ctx,
                  split ? s_.attn_partials : nullptr, kAttnSplits, n_tokens,
                  pos, c.num_heads, c.num_kv_heads, c.head_dim, 1, stream_);
  SigmoidMul(s_.ctx, s_.attn_gate,
             static_cast<std::size_t>(n_tokens) * c.AttentionQDim(), stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}
'''
    # The start line plus any /// comment lines above it
    line_start = text.rfind("\n", 0, start) + 1
    while True:
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        prev = text[prev_start:line_start - 1]
        if prev.lstrip().startswith("///") or prev.lstrip().startswith("//"):
            line_start = prev_start
        else:
            break
    text = text[:line_start] + new_attention + text[i + 3:]
    open(DST, "wb").write(text.replace("\n", "\r\n").encode("utf8"))
    print("part 2 ok")


part2()
print("parts 1-2 complete (part 3 follows)")
