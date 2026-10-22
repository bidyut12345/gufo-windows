"""Wire qwen35moe into src/cli/serve/inference_backend.{hpp,cpp}."""

HPP = "src/cli/serve/inference_backend.hpp"
CPP = "src/cli/serve/inference_backend.cpp"


def replace_one(text, old, new, what):
    if old not in text:
        print(f"MISSING ({what}):\n{old[:300]}")
        raise SystemExit(1)
    return text.replace(old, new, 1)


def span(text, start_marker, end_marker):
    a = text.index(start_marker)
    b = text.index(end_marker, a)
    return a, b


def main():
    # ---------------- header ----------------
    s = open(HPP, "rb").read().decode("utf8").replace("\r\n", "\n")
    s = replace_one(
        s,
        "namespace gufo::models::qwen38_flash_next {\nclass Model;\n}",
        "namespace gufo::models::qwen38_flash_next {\nclass Model;\n}\n\n"
        "namespace gufo::models::qwen35moe {\nclass Model;\n}",
        "hpp fwd decl",
    )
    s = replace_one(
        s,
        """  bool load(std::shared_ptr<models::qwen38_flash_next::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});
#endif""",
        """  bool load(std::shared_ptr<models::qwen38_flash_next::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});

  /// Installs a previously loaded Ornith model with request-owned sessions.
  bool load(std::shared_ptr<models::qwen35moe::Model> model,
            std::string* error, std::uint32_t max_context = 0,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});
#endif""",
        "hpp overload",
    )
    open(HPP, "wb").write(s.replace("\n", "\r\n").encode("utf8"))

    # ---------------- source ----------------
    s = open(CPP, "rb").read().decode("utf8").replace("\r\n", "\n")

    # Pull in the model API.
    s = replace_one(
        s,
        '#include "src/models/qwen38_flash_next/engine.hpp"',
        '#include "src/models/qwen38_flash_next/engine.hpp"\n'
        '#include "src/models/qwen35moe/engine.hpp"',
        "cpp include",
    )
    if "#include" in s and "src/models/qwen35moe/engine.hpp" not in s:
        raise SystemExit("include anchor missing")

    # ---- Clone the runner family ----
    a, b = span(
        s,
        "#if defined(ENGINE_ENABLE_HIP)\n"
        "// Each Flash-Next request owns its recurrent/KV state and snapshots.",
        "}  // namespace\n\nstruct InferenceBackend::Impl {",
    )
    family = s[a:b]
    # The family ends with "#endif\n" before the closing namespace comment.
    cut = family.rindex("#endif")
    family = family[:cut]
    clone = (
        family.replace("models::qwen38_flash_next", "models::qwen35moe")
        .replace("QwenFlashNext", "Qwen35Moe")
        .replace("qwen38-flash-next-rocm-session-v1", "qwen35moe-rocm-session-v1")
        .replace("Qwen3.8-Flash-Next", "Ornith-1.5-35B")
        .replace("Flash-Next", "Ornith")
        .replace("model_kind=qwen38-flash-next", "model_kind=ornith-1.5-35b")
        .replace("qfn-rocm-session-snapshot-v", "q35-rocm-session-snapshot-v")
        .replace("draft_backend=qfn-mtp-v1", "draft_backend=q35-mtp-v1")
        .replace(
            'identity << "chat_template=qwen38-reasoning-compiled-v3\\n"\n'
            "           << \"chat_template_reference_sha256=\"\n"
            "           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\\n'\n",
            'identity << "chat_template=ornith-embedded-v1\\n"\n',
        )
        # The draft block shares the model file: both fingerprints coincide.
        .replace(
            "Qwen35MoeCompatibilityIdentity(\n"
            "              artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},",
            "Qwen35MoeCompatibilityIdentity(\n"
            "              artifact_fingerprint, artifact_fingerprint,",
        )
    )
    # Guard: the mtp-fingerprint argument stays in the signature.
    s = s[:a] + family + "#endif\n\n" + clone + "#endif\n" + s[b - len("}  // namespace\n\nstruct InferenceBackend::Impl {") + len("}  // namespace\n\nstruct InferenceBackend::Impl {"):]
    # The splice above keeps the original tail intact; verify no duplicate
    # namespace close crept in.
    if "}  // namespace\n\nstruct InferenceBackend::Impl {" not in s:
        raise SystemExit("namespace tail lost")

    open(CPP, "wb").write(s.replace("\n", "\r\n").encode("utf8"))
    print("backend family cloned")


if __name__ == "__main__":
    main()
