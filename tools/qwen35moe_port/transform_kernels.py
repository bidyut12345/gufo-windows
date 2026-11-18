"""One-shot transform of qwen38_flash_next kernels to qwen35moe.

Cuts the hyper-connection, PLE and indexer kernels, retargets the WMMA
causal-attention kernel to both 16- and 24-head geometry, and adds the plain
residual combine kernel. Run from the repo root: python tools/qwen35moe_port/transform_kernels.py
"""
import re
import sys

SRC = "src/models/qwen38_flash_next"
DST = "src/models/qwen35moe"


def read(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode("utf8")


def write(path: str, text: str) -> None:
    data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf8")
    with open(path, "wb") as f:
        f.write(data)


def cut_function(text: str, start_marker: str, comment_ok: bool = True) -> str:
    """Removes the function opened at start_marker through its closing brace."""
    idx = text.index(start_marker)
    # Include immediately preceding /// comment lines and blank separation.
    line_start = text.rfind("\n", 0, idx) + 1
    if comment_ok:
        while True:
            prev_start = text.rfind("\n", 0, line_start - 1) + 1
            prev = text[prev_start:line_start - 1]
            if prev.lstrip().startswith("///"):
                line_start = prev_start
            else:
                break
    brace = text.index("{", idx)
    depth = 0
    i = brace
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    end = i + 1
    # Swallow the trailing newline.
    if end < len(text) and text[end] == "\r":
        end += 2
    elif end < len(text) and text[end] == "\n":
        end += 1
    return text[:line_start] + text[end:]


def cut_span(text: str, start_marker: str, end_marker: str) -> str:
    """Removes [start_marker line, end_marker line) inclusive of start."""
    start = text.index(start_marker)
    line_start = text.rfind("\n", 0, start) + 1
    # Pull in preceding /// comment lines.
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


def cut_declaration(text: str, start_marker: str) -> str:
    """Removes a `;`-terminated declaration plus its /// comment lines."""
    idx = text.index(start_marker)
    line_start = text.rfind("\n", 0, idx) + 1
    while True:
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        prev = text[prev_start:line_start - 1]
        if prev.lstrip().startswith("///") or prev.lstrip().startswith("//"):
            line_start = prev_start
        else:
            break
    end = text.index(";", idx)
    while end + 1 < len(text) and text[end + 1] == "\r":
        end += 1
    if end + 1 < len(text) and text[end + 1] == "\n":
        end += 1
    return text[:line_start] + text[end + 1:]


def transform_cpp(text: str) -> str:
    # Device kernel sections (marker-span cuts; each span holds only
    # hyper-connection / PLE / indexer kernels).
    text = cut_span(
        text,
        "/// grid (tokens, hidden chunks of kThreads): the stream loop stays inside",
        "__global__ void SiluScaleKernel(",
    )
    text = cut_span(
        text, "__global__ void PleGateKernel(", "/// New history row j is row"
    )
    text = cut_span(
        text, "__global__ void PleInjectKernel(", "/// L2-normalizes the convolved"
    )
    # The dense path never reads tail_start, but the division must stay
    # defined for the zero-ratio (no-selection) launches.
    text = text.replace(
        "  const std::uint32_t tail_start = (n_kv / ratio) * ratio;",
        "  const std::uint32_t tail_start ="
        " ratio != 0 ? (n_kv / ratio) * ratio : n_kv;", 1)
    text = cut_span(
        text,
        "__global__ void StoreRowsKernel(",
        "/// grid (heads, queries), block 256 = head dim.",
    )

    # Host wrappers (brace-matched cuts).
    for marker in (
        "std::uint32_t HcInjectParts(std::uint32_t hidden) {",
        "std::uint32_t HcInjectPartsVec4(std::uint32_t hidden) {",
        "void HcMixEpilogue(const float* xn,",
        "void HcMixEpilogueVec4(const float* xn,",
        "void HcMixEpilogueVec4F16(const __half* xn,",
        "void HcCombine(float* res,",
        "bool HcCombineMoeF16(float* res,",
        "void HcCombineF16(float* res,",
        "bool HcDownF16Gemm(const void* w,",
        "bool AttentionF16Gemm(const void* weights,",
        "bool HcMixF16Gemm(const void* up,",
        "void PleGate(const float* key_n,",
        "void PleConv(const float* in,",
        "void PleInject(float* res,",
        "void StoreRows(const float* src,",
        "void PoolIndexerBlocks(const float* raw_keys,",
        "void SelectBlocks(const float* q,",
    ):
        text = cut_function(text, marker)

    # WMMA causal attention: parameterize the query-head count and drop the
    # sparse packed path (qwen35moe never selects blocks).
    text = text.replace(
        "template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kPackHeads,\n"
        "         bool kLateV = false>",
        "template<std::uint32_t kQueryHeads, std::uint32_t kQueryRows,\n"
        "         std::uint32_t kKeys, bool kPackHeads, bool kLateV = false>",
        1,
    )
    text = text.replace(
        "  // The launcher accepts only four-token selection blocks. Keep that\n"
        "  // geometry constant throughout the key sweep.\n"
        "  constexpr std::uint32_t ratio = kWmmaRatio;\n"
        "  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;",
        "  constexpr std::uint32_t ratio = kWmmaRatio;\n"
        "  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;\n"
        "  constexpr std::uint32_t kGqa = kQueryHeads / kWmmaKvHeads;\n"
        "  constexpr std::uint32_t kAttnWidth = kQueryHeads * kWmmaHeadDim;",
        1,
    )
    # Replace the constant uses inside the kernel body only (they are unique
    # spellings within the template).
    text = text.replace("kQueryRows * kWmmaGqa", "kQueryRows * kGqa")
    text = text.replace("(kWmmaGqa / kWmmaHeads)", "(kGqa / kWmmaHeads)")
    text = text.replace("kv_head * kWmmaGqa", "kv_head * kGqa")
    text = text.replace("(rb * 16 + r) / kWmmaGqa", "(rb * 16 + r) / kGqa")
    text = text.replace("(rb * 16 + r) % kWmmaGqa", "(rb * 16 + r) % kGqa")
    text = text.replace(") * kWmmaAttnWidth)", ") * kAttnWidth)")
    text = text.replace(
        "static_assert(!kPackHeads || kQueryRows * kGqa == 48,",
        "static_assert(!kPackHeads || kQueryRows * kGqa == 48,",
    )

    # Launcher: accept 16- and 24-head geometry, dense windows only.
    old_launcher = text[text.index("bool WmmaCausalAttention("):]
    end = old_loader = None
    m = re.search(r"\n}\n", old_launcher)
    assert m, "WmmaCausalAttention end not found"
    old_launcher = old_launcher[: m.end()]
    new_launcher = """bool WmmaCausalAttention(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream,
                         bool last_only) {
  if ((heads != 16 && heads != kWmmaQueryHeads) || kv_heads != kWmmaKvHeads ||
      d != kWmmaHeadDim || n_tokens == 0 ||
      (mask != nullptr && mask_words > kWmmaMaxMaskWords)) {
    return false;
  }
  // Dense windows only: qwen35moe never selects blocks, so the mask arrives
  // null and the selection ratio is meaningless.
  if (mask != nullptr || ratio != 0) {
    return false;
  }
  const std::uint32_t gqa = heads / kWmmaKvHeads;
  const std::uint32_t first_group =
      last_only ? (n_tokens - 1) / kWmmaQueryRows : 0;
  const dim3 grid((n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows - first_group,
                  kWmmaKvHeads * (gqa / kWmmaHeads));
  if (heads == 16) {
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<16, kWmmaQueryRows, kWmmaKeys, false>), grid,
        dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask, mask_words,
        out, start_pos, n_tokens, first_group);
  } else {
    hipLaunchKernelGGL(
        (WmmaCausalAttentionKernel<kWmmaQueryHeads, kWmmaQueryRows, kWmmaKeys,
                                   false>),
        grid, dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask,
        mask_words, out, start_pos, n_tokens, first_group);
  }
  return true;
}
"""
    idx = text.index("bool WmmaCausalAttention(")
    text = text[:idx] + new_launcher + text[idx + len(old_launcher):]

    # Plain residual combine: res += block_out.
    anchor = "void EmbedTokens(const void* table, WeightType type,"
    addition = """/// res[i] += block_out[i]; the plain residual update of a standard
/// pre-norm block.
__global__ void CombineAddKernel(const float* __restrict__ block_out,
                                 float* __restrict__ res, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    res[i] += block_out[i];
  }
}

void CombineAdd(const float* block_out, float* res, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(CombineAddKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, block_out, res, count);
}

"""
    text = text.replace(anchor, addition + anchor, 1)
    return text


def transform_hpp(text: str) -> str:
    for marker in (
        "std::uint32_t HcInjectParts(std::uint32_t hidden);",
        "std::uint32_t HcInjectPartsVec4(std::uint32_t hidden);",
        "void HcMixEpilogue(const float* xn,",
        "void HcMixEpilogueVec4(const float* xn,",
        "void HcMixEpilogueVec4F16(const __half* xn,",
        "void HcCombine(float* res,",
        "bool HcCombineMoeF16(float* res,",
        "void HcCombineF16(float* res,",
        "bool HcDownF16Gemm(const void* w,",
        "bool AttentionF16Gemm(const void* weights,",
        "bool HcMixF16Gemm(const void* up,",
        "void PleGate(const float* key_n,",
        "void PleConv(const float* in,",
        "void PleInject(float* res,",
        "void StoreRows(const float* src,",
        "void PoolIndexerBlocks(const float* raw_keys,",
        "void SelectBlocks(const float* q,",
    ):
        text = cut_declaration(text, marker)
    # The WMMA attention declaration: dense windows only, either head count.
    text = text.replace(
        """/// Fused causal attention on the WMMA cores for wide batches: scores, online
/// softmax, PV and the sigmoid output gate in one launch. `mask` follows
/// Attention's block-selection contract (null for the dense window). Returns
/// false, launching nothing, when the geometry is not the model's 24 x 256
/// heads over two KV heads. `last_only` computes only the final dense query
/// tile, retaining its key sweep and leaving earlier output rows untouched.""",
        """/// Fused causal attention on the WMMA cores for wide batches: scores, online
/// softmax, PV and the sigmoid output gate in one launch. `mask` must be null
/// (dense windows only). Returns false, launching nothing, when the geometry
/// is not 16 or 24 query heads of 256 over two KV heads. `last_only` computes
/// only the final dense query tile, retaining its key sweep and leaving
/// earlier output rows untouched.""",
        1,
    )
    anchor = "void EmbedTokens(const void* table, WeightType type,"
    addition = """/// res[i] += block_out[i], the plain residual update of a standard
/// pre-norm block.
void CombineAdd(const float* block_out, float* res, std::size_t count,
                hipStream_t stream);

"""
    text = text.replace(anchor, addition + anchor, 1)
    return text


def main() -> None:
    cpp = read(f"{SRC}/kernels/rocm/kernels.hip.cpp").replace("\r\n", "\n")
    hpp = read(f"{SRC}/kernels/rocm/kernels.hpp").replace("\r\n", "\n")
    cpp = cpp.replace("qwen38_flash_next", "qwen35moe")
    hpp = hpp.replace("qwen38_flash_next", "qwen35moe")
    cpp = transform_cpp(cpp)
    hpp = transform_hpp(hpp)
    write(f"{DST}/kernels/rocm/kernels.hip.cpp", cpp)
    write(f"{DST}/kernels/rocm/kernels.hpp", hpp)
    # Verification: no removed symbol may remain.
    banned = [
        "PleGate", "PleConv", "PleInject", "HcMixEpilogue", "HcCombine",
        "HcDownF16Gemm", "HcMixF16Gemm", "AttentionF16Gemm", "HcInjectParts",
        "StoreRows", "PoolIndexerBlocks", "SelectBlocks", "SelectScore",
        "SelectMark", "PoolBlocks", "StoreRowsKernel",
    ]
    bad = []
    for symbol in banned:
        for name, body in (("kernels.hip.cpp", cpp), ("kernels.hpp", hpp)):
            if symbol in body:
                bad.append(f"{name}: {symbol}")
    if bad:
        print("LEFTOVER SYMBOLS:")
        for entry in bad:
            print(" ", entry)
        sys.exit(1)
    print("kernels transform ok")


if __name__ == "__main__":
    main()
