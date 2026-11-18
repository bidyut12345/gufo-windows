"""Add the qwen35moe arch branch and load() overload to inference_backend.cpp."""

CPP = "src/cli/serve/inference_backend.cpp"


def replace_one(text, old, new, what):
    if old not in text:
        print(f"MISSING ({what}):\n{old[:300]}")
        raise SystemExit(1)
    return text.replace(old, new, 1)


def main():
    s = open(CPP, "rb").read().decode("utf8").replace("\r\n", "\n")

    # ---- Arch branch: insert after the flash-next branch ----
    anchor = """    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config));
  }
  std::shared_ptr<models::qwen::vision::Encoder> vision;"""
    branch = """    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config));
  }
  if (reader->GetMetadataString("general.architecture") == "qwen35moe") {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Ornith HTTP models support only MTP speculative decoding "
               "(--speculative mtp)");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        !speculative_config.draft_model_path.empty()) {
      SetError(error,
               "Ornith MTP is built into the model file; do not pass "
               "--mtp-model");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        (speculative_config.max_draft_tokens == 0 ||
         speculative_config.min_draft_tokens != 1)) {
      SetError(error,
               "Ornith MTP requires a positive draft limit and "
               "--min-draft-tokens 1");
      return false;
    }
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                              &load_error)) {
      SetError(error, "Unsupported Ornith chat template: " + load_error);
      return false;
    }
    auto model = models::qwen35moe::Model::Load(
        model_path,
        models::qwen35moe::ModelOptions{
            .max_context = max_context,
            .max_draft_tokens = speculative_config.max_draft_tokens,
            .vision_model_path = vision_model_path,
            .decode_concurrency = static_cast<std::uint32_t>(
                std::clamp<std::size_t>(session_count, 1, 8)),
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create Ornith model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact("Ornith", *reader,
                             &resolved_disk_cache_config.model_artifact_fingerprint,
                             error)) {
      return false;
    }
    // The draft block shares the model file, so both cache fingerprints
    // are the artifact's.
    resolved_disk_cache_config.draft_model_artifact_fingerprint =
        resolved_disk_cache_config.model_artifact_fingerprint;
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config));
  }
  std::shared_ptr<models::qwen::vision::Encoder> vision;"""
    s = replace_one(s, anchor, branch, "arch branch")

    # ---- load() overload: clone the flash-next one ----
    start_marker = """bool InferenceBackend::load(
    std::shared_ptr<models::qwen38_flash_next::Model> model, std::string* error,"""
    a = s.index(start_marker)
    # Find the end: the overload closes with "#endif" followed by a blank line
    # and "std::string InferenceBackend::model_id()".
    b = s.index("std::string InferenceBackend::model_id()", a)
    overload = s[a:b]
    cut = overload.rindex("#endif")
    clone = (
        overload[:cut]
        .replace("models::qwen38_flash_next", "models::qwen35moe")
        .replace("QwenFlashNextTextRunner", "Qwen35MoeTextRunner")
        .replace("Qwen3.8-Flash-Next", "Ornith-1.5-35B")
        .replace("Flash-Next", "Ornith")
    )
    # In-file MTP: the draft-sidecar requirement becomes a no-sidecar rule.
    clone = replace_one(
        clone,
        """  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Ornith MTP requires a model loaded with its draft sidecar");
    return false;
  }""",
        """  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Ornith MTP requires the in-file nextn block, which this "
             "artifact lacks");
    return false;
  }""",
        "overload mtp check",
    )
    s = s[:b] + clone + "#endif\n\n" + s[b:]

    open(CPP, "wb").write(s.replace("\n", "\r\n").encode("utf8"))
    print("backend branch + overload ok")


if __name__ == "__main__":
    main()
