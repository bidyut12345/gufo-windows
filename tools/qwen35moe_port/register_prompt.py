"""Wire qwen35moe into src/cli/prompt/prompt.cpp."""

CPP = "src/cli/prompt/prompt.cpp"


def replace_one(text, old, new, what):
    if old not in text:
        print(f"MISSING ({what}):\n{old[:300]}")
        raise SystemExit(1)
    return text.replace(old, new, 1)


def main():
    s = open(CPP, "rb").read().decode("utf8").replace("\r\n", "\n")

    # Include the new engine API next to the flash-next one.
    anchor = '#include "src/models/qwen38_flash_next/engine.hpp"'
    if anchor in s and "src/models/qwen35moe/engine.hpp" not in s:
        s = replace_one(s, anchor, anchor + '\n#include "src/models/qwen35moe/engine.hpp"',
                        "include")

    # ---- Clone the loader ----
    loader_start = s.index(
        "std::shared_ptr<models::qwen38_flash_next::Model> LoadFlashNextModel(")
    loader_end = s.index("int GenerateFlashNextResponse(", loader_start)
    loader = s[loader_start:loader_end]
    orn_loader = (
        loader.replace("LoadFlashNextModel", "LoadOrnithModel")
        .replace("models::qwen38_flash_next", "models::qwen35moe")
        .replace("Flash-Next", "Ornith")
        .replace(
            """  auto model = models::qwen35moe::Model::Load(
      opt.model_path,
      {.max_context = kDefaultContext,
       .mtp_model_path =
           opt.speculative_backend == "mtp" ? opt.mtp_model_path : "",
       .max_draft_tokens = opt.draft_tokens,
       .vision_model_path = opt.vision_model_path},
      &error);""",
            """  auto model = models::qwen35moe::Model::Load(
      opt.model_path,
      {.max_context = kDefaultContext,
       .max_draft_tokens = opt.draft_tokens,
       .vision_model_path = opt.vision_model_path},
      &error);""",
        )
    )
    if "mtp_model_path" in orn_loader:
        raise SystemExit("ornith loader still references mtp_model_path")
    s = s[:loader_end] + orn_loader + s[loader_end:]

    # ---- Clone the response generator ----
    gen_start = s.index("int GenerateFlashNextResponse(")
    gen_end = s.index("std::unique_ptr<speculative::SpeculativeVerifier> CreateQwenVerifier(")
    gen = s[gen_start:gen_end]
    orn_gen = (
        gen.replace("GenerateFlashNextResponse", "GenerateOrnithResponse")
        .replace("models::qwen38_flash_next", "models::qwen35moe")
        .replace("Flash-Next", "Ornith")
    )
    s = s[:gen_end] + orn_gen + s[gen_end:]

    # ---- RunPrompt dispatch branch ----
    s = replace_one(
        s,
        """#if defined(ENGINE_ENABLE_HIP)
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    auto model = LoadFlashNextModel(opt, *reader, model_load_start);""",
        """#if defined(ENGINE_ENABLE_HIP)
  if (reader->GetMetadataString("general.architecture") == "qwen35moe") {
    auto model = LoadOrnithModel(opt, *reader, model_load_start);
    if (!model)
      return 1;
    auto session =
        model->CreateSession(opt.speculative_backend.empty()
                                 ? gufo::core::SessionMode::kAutoregressive
                                 : gufo::core::SessionMode::kSpeculative,
                             kDefaultContext, &err);
    if (!session) {
      std::cerr << "Ornith session failed: " << err << '\\n';
      return 1;
    }
    try {
      if (!opt.image_paths.empty()) {
        AttachImages(opt, messages.back());
        auto vision = PrepareVision(opt, model->tokenizer(), messages,
                                    model->VisionEncoder());
        session->ConfigureVision(vision);
        return GenerateOrnithResponse(opt, *model, *session, vision->tokens);
      }
      const auto ids = model->Tokenize(rendered_prompt);
      const std::vector<tokenization::TokenId> prompt(ids.begin(), ids.end());
      return GenerateOrnithResponse(opt, *model, *session, prompt);
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\\n';
      return 1;
    }
  }
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    auto model = LoadFlashNextModel(opt, *reader, model_load_start);""",
        "prompt dispatch",
    )

    # ---- RunChat state + dispatch ----
    s = replace_one(
        s,
        """  std::shared_ptr<models::qwen38_flash_next::Model> flash_model;
  std::unique_ptr<models::qwen38_flash_next::Session> flash_session;
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {""",
        """  std::shared_ptr<models::qwen38_flash_next::Model> flash_model;
  std::unique_ptr<models::qwen38_flash_next::Session> flash_session;
  std::shared_ptr<models::qwen35moe::Model> ornith_model;
  std::unique_ptr<models::qwen35moe::Session> ornith_session;
  if (reader->GetMetadataString("general.architecture") == "qwen35moe") {
    ornith_model = LoadOrnithModel(opt, *reader, model_load_start);
    if (!ornith_model)
      return 1;
    ornith_session = ornith_model->CreateSession(
        opt.speculative_backend.empty()
            ? gufo::core::SessionMode::kAutoregressive
            : gufo::core::SessionMode::kSpeculative,
        kDefaultContext, &err);
    if (!ornith_session) {
      std::cerr << "Ornith session failed: " << err << '\\n';
      return 1;
    }
    tokenizer = &ornith_model->tokenizer();
    architecture = "qwen35moe";
    vision_encoder = ornith_model->VisionEncoder();
  }
  if (tokenizer == nullptr &&
      reader->GetMetadataString("general.architecture") == "qwen4exp") {""",
        "chat state",
    )
    s = replace_one(
        s,
        """      if (flash_model) {
        if (GenerateFlashNextResponse(opt, *flash_model, *flash_session,
                                      prompt_tokens, &assistant_reply) != 0)
          return 1;
      } else if (gpu_executor != nullptr) {""",
        """      if (flash_model) {
        if (GenerateFlashNextResponse(opt, *flash_model, *flash_session,
                                      prompt_tokens, &assistant_reply) != 0)
          return 1;
      } else if (ornith_model) {
        if (GenerateOrnithResponse(opt, *ornith_model, *ornith_session,
                                   prompt_tokens, &assistant_reply) != 0)
          return 1;
      } else if (gpu_executor != nullptr) {""",
        "chat generation dispatch",
    )

    open(CPP, "wb").write(s.replace("\n", "\r\n").encode("utf8"))
    print("prompt ok")


if __name__ == "__main__":
    main()
