"""Wire qwen35moe into src/cli/bench/bench.cpp."""

CPP = "src/cli/bench/bench.cpp"


def replace_one(text, old, new, what):
    if old not in text:
        print(f"MISSING ({what}):\n{old[:300]}")
        raise SystemExit(1)
    return text.replace(old, new, 1)


def main():
    s = open(CPP, "rb").read().decode("utf8").replace("\r\n", "\n")

    anchor = '#include "src/models/qwen38_flash_next/engine.hpp"'
    if anchor in s and "src/models/qwen35moe/engine.hpp" not in s:
        s = replace_one(s, anchor, anchor + '\n#include "src/models/qwen35moe/engine.hpp"',
                        "include")

    # ---- Clone the whole benchmark function ----
    start = s.index("int RunQwen38FlashNextBenchmark(")
    end = s.index("#endif\n\n}  // namespace", start)
    fn = s[start:end]
    orn = (
        fn.replace("RunQwen38FlashNextBenchmark", "RunOrnithBenchmark")
        .replace("IsQwen38FlashNext", "IsOrnithModel")
        .replace("namespace qfn = models::qwen38_flash_next;",
                 "namespace q35 = models::qwen35moe;")
        .replace("qfn::", "q35::")
        .replace("Qwen3.8-Flash-Next", "Ornith-1.5-35B")
        .replace("Flash-Next", "Ornith")
        .replace(
            """  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Ornith-1.5-35B supports only --speculative mtp "
                 "or off\\n";
    return 1;
  }
  if (mtp && options.mtp_model_path.empty()) {
    std::cerr << "Error: --speculative mtp requires --mtp-model\\n";
    return 1;
  }""",
            """  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Ornith-1.5-35B supports only --speculative mtp "
                 "or off\\n";
    return 1;
  }
  if (mtp && !options.mtp_model_path.empty()) {
    std::cerr << "Error: Ornith MTP is built into the model file; do not "
                 "pass --mtp-model\\n";
    return 1;
  }""",
        )
        .replace(
            """      q35::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .mtp_model_path = mtp ? options.mtp_model_path : "",
          .max_draft_tokens = std::max<std::uint32_t>(1, options.draft_tokens),
      },""",
            """      q35::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .max_draft_tokens = std::max<std::uint32_t>(1, options.draft_tokens),
      },""",
        )
    )
    if "mtp_model_path =" in orn:
        raise SystemExit("ornith bench still passes mtp_model_path")
    s = s[:end] + orn + s[end:]

    # ---- Dispatch ----
    s = replace_one(
        s,
        """  if (IsQwen38FlashNext(*reader)) {
    return RunQwen38FlashNextBenchmark(opt, reader, model_load_start);
  }""",
        """  if (IsQwen38FlashNext(*reader)) {
    return RunQwen38FlashNextBenchmark(opt, reader, model_load_start);
  }
  if (reader->GetMetadataString("general.architecture") == "qwen35moe") {
    return RunOrnithBenchmark(opt, reader, model_load_start);
  }""",
        "bench dispatch",
    )

    open(CPP, "wb").write(s.replace("\n", "\r\n").encode("utf8"))
    print("bench ok")


if __name__ == "__main__":
    main()
