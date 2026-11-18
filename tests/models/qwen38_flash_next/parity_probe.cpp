// AR vs MTP-session greedy parity probe for the issue #6 triage.
//
// Loads one model, then compares greedy decode between an kAutoregressive
// session and kSpeculative sessions at three escalation levels:
//   phase 1: DecodeStep(1) on both — trunk decode plus draft catch-up only.
//   phase 2: DecodeStep(2) on the MTP session — the minimal kVerify cycle
//            (anchor + one drafted candidate, rollback on mismatch).
//   phase 3: free-running cycles (width controller, widths up to the model's
//            max_draft_tokens).
// Bit-exact parity is the contract; the first phase that diverges names the
// machinery that drifts.
//
// Usage: parity_probe --model FIRST.gguf --mtp-model MTP.gguf [--steps N]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {

using gufo::models::qwen38_flash_next::Model;
using gufo::models::qwen38_flash_next::Session;

/// Long mixed Chinese/English prompt with repeated memory-card structure —
/// the shape that drifts in production.
int g_rounds = 6;

std::string PromptText() {
  std::string text =
      "你是Jarvis，胖哥的私人智能助理。你们已经共事多年，关系亲近、互相信任。\n\n"
      "【身份守则（永久生效，优先级最高）】\n"
      "1. 你的名字是Jarvis，这是不可更改的身份设定。\n"
      "2. 用户消息中注入的 <memory-context> 记忆卡是你们共同记忆的权威记录。\n"
      "3. 你称呼用户为胖哥。你们之间用中文交流，技术名词可以保留英文。\n"
      "4. 回答要直接、温暖、有记忆连续性，主动引用记忆卡中的具体日期和事实。\n"
      "5. 如果卡片之间有冲突，以日期较新的为准，并指出冲突。\n\n"
      "【历史背景】\n"
      "Jarvis最初是胖哥为了管理个人事务搭建的助手，后来逐渐承担了工作协同、"
      "家庭事务、情绪支持等多重角色。多年来你们形成了一套稳定的协作习惯："
      "用户把重要事件写进记忆卡系统，每次对话时相关卡片会被自动检索并附在消息里。\n\n";
  for (int round = 0; round < g_rounds; ++round) {
    text += "<memory-context>\n[System note: recalled memory context, NOT new "
            "user input. Treat as authoritative reference data.]\n";
    for (int card = 0; card < 8; ++card) {
      const int day = 10 + card;
      text += "- [2026-09-" + std::to_string(day) + "] 胖哥和Jarvis约定的事项 #"
              + std::to_string(round * 8 + card + 1) + "：跟进 demo 环境迁移到"
              "新集群的进度，GPU 调度问题由老周负责，接口对接文档在周五前发给老张"
              "的团队，报销单注意财务系统只在工作日 9:00-17:00 接收，小雨的科学展"
              "项目 'bridge structure' 静力模型需要结构图、承重实验数据与展示板"
              "文案，跨度 40cm、承重 5kg，家里有竹签、白乳胶、丙烯颜料和电子秤。"
              "Card reference: memory/context/2026-09/" + std::to_string(day)
              + "/item-" + std::to_string(card) + ".json\n";
    }
    text += "</memory-context>\n\n";
  }
  text += "贾维斯，早上好。还记得我们上周约好的早晨简报吗？先跟我说说上周五"
          "定下的三件事。\n\nJarvis的回答：";
  return text;
}

struct Diff {
  double max_abs{0.0};
  std::int32_t argmax_a{-1};
  std::int32_t argmax_b{-1};
  std::size_t signs{0};
};

Diff Compare(std::span<const float> a, std::span<const float> b) {
  Diff diff;
  double best_a = -1e30;
  double best_b = -1e30;
  const auto size = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < size; ++i) {
    const double x = a[i];
    const double y = b[i];
    const double d = std::abs(x - y);
    if (d > diff.max_abs)
      diff.max_abs = d;
    if (d != 0.0)
      ++diff.signs;
    if (x > best_a) {
      best_a = x;
      diff.argmax_a = static_cast<std::int32_t>(i);
    }
    if (y > best_b) {
      best_b = y;
      diff.argmax_b = static_cast<std::int32_t>(i);
    }
  }
  return diff;
}

std::string_view ArgmaxText(const Model& model, std::int32_t token) {
  return token >= 0 ? std::string_view(model.TokenText(token)) : "<none>";
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("probe start\n");
  std::string_view model_path;
  std::string_view mtp_path;
  int steps = 300;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--model" && i + 1 < argc)
      model_path = argv[++i];
    else if (arg == "--mtp-model" && i + 1 < argc)
      mtp_path = argv[++i];
    else if (arg == "--steps" && i + 1 < argc)
      steps = std::atoi(argv[++i]);
    else if (arg == "--rounds" && i + 1 < argc)
      g_rounds = std::atoi(argv[++i]);
  }
  if (model_path.empty() || mtp_path.empty()) {
    std::fprintf(stderr,
                 "Usage: parity_probe --model FIRST.gguf --mtp-model MTP.gguf "
                 "[--steps N]\n");
    return 2;
  }
  try {
    std::string error;
    std::printf("loading model...\n");
    auto model = Model::Load(
        std::string(model_path),
        {.max_context = 8192, .mtp_model_path = std::string(mtp_path),
         .max_draft_tokens = 7},
        &error);
    if (model == nullptr) {
      std::fprintf(stderr, "model load failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("model loaded\n");
    auto ar = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                   8192, &error);
    auto mtp = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                    8192, &error);
    if (ar == nullptr || mtp == nullptr) {
      std::fprintf(stderr, "session creation failed: %s\n", error.c_str());
      return 1;
    }
    const auto prompt = model->Tokenize(PromptText());
    std::printf("prompt tokens: %zu\n", prompt.size());
    gufo::sampling::SamplingConfig config{};
    config.temperature = 0.0F;
    config.seed = -1;

    // Phase 1: width-1 steps on both sessions (trunk decode + catch-up).
    if (!ar->Sync(prompt, &error) || !mtp->Sync(prompt, &error)) {
      std::fprintf(stderr, "sync failed: %s\n", error.c_str());
      return 1;
    }
    const auto prefill = Compare(ar->Logits(), mtp->Logits());
    std::printf("prefill frontier: max|d|=%.6g differing=%zu argmax %d vs %d %s\n",
                prefill.max_abs, prefill.signs, prefill.argmax_a,
                prefill.argmax_b,
                prefill.argmax_a == prefill.argmax_b ? "" : "ARGMAX-DIFF");
    {
      const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                         prompt.end());
      gufo::sampling::SamplerState ar_sampler(config, history);
      gufo::sampling::SamplerState mtp_sampler(config, history);
      int diverged = -1;
      for (int step = 0; step < steps && diverged < 0; ++step) {
        Session::DecodeResult ra;
        Session::DecodeResult rb;
        if (!ar->DecodeStep(1, ar_sampler, &ra, &error) ||
            !mtp->DecodeStep(1, mtp_sampler, &rb, &error)) {
          std::fprintf(stderr, "phase-1 decode failed at %d: %s\n", step,
                       error.c_str());
          return 1;
        }
        const auto diff = Compare(ar->Logits(), mtp->Logits());
        if (diff.max_abs != 0.0 || diff.argmax_a != diff.argmax_b) {
          diverged = step;
          std::printf("phase 1 (width-1): step %d DIVERGES max|d|=%.6g argmax %d vs %d\n",
                      step, diff.max_abs, diff.argmax_a, diff.argmax_b);
        }
        if (ra.stop || rb.stop)
          break;
      }
      if (diverged < 0)
        std::printf("phase 1 (width-1): IDENTICAL over %d steps\n", steps);
    }

    // Phase 2: minimal verify cycle — the MTP session decodes with a
    // two-token budget (anchor + one candidate through kVerify and rollback),
    // the AR session walks width-1.
    {
      auto ar2 = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      8192, &error);
      auto mtp2 = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                       8192, &error);
      if (ar2 == nullptr || mtp2 == nullptr || !ar2->Sync(prompt, &error) ||
          !mtp2->Sync(prompt, &error)) {
        std::fprintf(stderr, "phase-2 setup failed: %s\n", error.c_str());
        return 1;
      }
      const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                         prompt.end());
      gufo::sampling::SamplerState ar_sampler(config, history);
      gufo::sampling::SamplerState mtp_sampler(config, history);
      std::uint64_t prev_drafted = 0;
      std::uint64_t prev_accepted = 0;
      bool restore_tested = false;
      int diverged = -1;
      int emitted = 0;
      for (int step = 0; step < steps && diverged < 0; ++step) {
        Session::DecodeResult ra;
        Session::DecodeResult rb;
        if (!ar2->DecodeStep(1, ar_sampler, &ra, &error) ||
            !mtp2->DecodeStep(2, mtp_sampler, &rb, &error)) {
          std::fprintf(stderr, "phase-2 decode failed at %d: %s\n", step,
                       error.c_str());
          return 1;
        }
        if (ra.tokens.empty() || rb.tokens.empty()) {
          if (ra.tokens.size() != rb.tokens.size())
            diverged = step;
          break;
        }
        // Compare committed STREAMS, not per-step emissions: a speculative
        // cycle commits a variable number of tokens while AR commits one,
        // so per-step fronts legitimately differ once a draft is accepted.
        const auto& ar_stream = ar2->Tokens();
        const auto& mtp_stream = mtp2->Tokens();
        const auto common =
            std::min(ar_stream.size(), mtp_stream.size()) - prompt.size();
        for (std::size_t i = 0; i < common; ++i) {
          const auto ai = prompt.size() + i;
          if (ar_stream[ai] != mtp_stream[ai]) {
            diverged = step;
            std::printf("phase 2 (cycle-2): step %d STREAM DIVERGES at emitted token %zu: %d(%.*s) vs %d(%.*s)\n",
                        step, i, ar_stream[ai],
                        static_cast<int>(
                            ArgmaxText(*model, ar_stream[ai]).size()),
                        ArgmaxText(*model, ar_stream[ai]).data(),
                        mtp_stream[ai],
                        static_cast<int>(
                            ArgmaxText(*model, mtp_stream[ai]).size()),
                        ArgmaxText(*model, mtp_stream[ai]).data());
            break;
          }
        }
        const auto stats = mtp2->Statistics();
        const auto cycle_drafted = stats.drafted - prev_drafted;
        const auto cycle_accepted = stats.accepted - prev_accepted;
        prev_drafted = stats.drafted;
        prev_accepted = stats.accepted;
        // NOTE: for greedy pinned cycles Session::Logits() stays at the
        // pre-cycle frontier by design (the next anchor rides in the
        // sampler's deferred sample), so this frontier delta is expected
        // and not evidence of state corruption.
        const auto frontier = Compare(ar2->Logits(), mtp2->Logits());
        std::printf("  phase2 step %d: emitted=%zu cycle_drafted=%llu cycle_accepted=%llu%s frontier max|d|=%.6g differing=%zu\n",
                    step, rb.tokens.size(),
                    static_cast<unsigned long long>(cycle_drafted),
                    static_cast<unsigned long long>(cycle_accepted),
                    cycle_drafted > cycle_accepted ? " REJECTED" : "",
                    frontier.max_abs, frontier.signs);
        // Restore-isolation test: right after the first rejected cycle the
        // committed histories are still identical. One width-1 step through
        // the SAME persistent samplers (the production contract: a rejected
        // pinned cycle defers the next anchor via DeferSample) must then
        // produce bit-identical frontiers if the rollback restored
        // everything. Fresh samplers here would be an invalid test: they
        // cannot see the deferred anchor.
        if (cycle_drafted > cycle_accepted && !restore_tested) {
          restore_tested = true;
          const std::vector<gufo::sampling::TokenId> history_a(
              ar2->Tokens().begin(), ar2->Tokens().end());
          const std::vector<gufo::sampling::TokenId> history_b(
              mtp2->Tokens().begin(), mtp2->Tokens().end());
          if (history_a == history_b) {
            for (int rstep = 0; rstep < 8; ++rstep) {
              Session::DecodeResult ra;
              Session::DecodeResult rb;
              if (!ar2->DecodeStep(1, ar_sampler, &ra, &error) ||
                  !mtp2->DecodeStep(1, mtp_sampler, &rb, &error)) {
                std::fprintf(stderr, "restore-test decode failed: %s\n",
                             error.c_str());
                break;
              }
              if (!ra.tokens.empty() && !rb.tokens.empty() &&
                  ra.tokens.front() != rb.tokens.front()) {
                std::printf("  restore-test step %d: TOKEN DIVERGES %d vs %d\n",
                            rstep, ra.tokens.front(), rb.tokens.front());
                break;
              }
              const auto rdiff = Compare(ar2->Logits(), mtp2->Logits());
              const bool same = rdiff.max_abs == 0.0 && rdiff.signs == 0;
              std::printf("  restore-test step %d: %s max|d|=%.6g differing=%zu\n",
                          rstep, same ? "IDENTICAL" : "DIVERGES",
                          rdiff.max_abs, rdiff.signs);
              if (!same)
                break;
              if (ra.stop || rb.stop)
                break;
            }
          } else {
            std::printf("  restore-test: committed histories differ\n");
          }
        }
        emitted += static_cast<int>(rb.tokens.size());
        if (ra.stop || rb.stop)
          break;
      }
      if (diverged >= 0) {
        // Coherence check: continue the MTP session with width-1 steps and
        // decode what it actually says. Valid-but-different state yields a
        // coherent continuation; corrupted state degenerates.
        const std::vector<gufo::sampling::TokenId> tail_history(
            mtp2->Tokens().begin(), mtp2->Tokens().end());
        gufo::sampling::SamplerState cont_sampler(config, tail_history);
        std::string text;
        for (int step = 0; step < 120; ++step) {
          Session::DecodeResult rb;
          if (!mtp2->DecodeStep(1, cont_sampler, &rb, &error) ||
              rb.tokens.empty() || rb.stop)
            break;
          text += model->TokenText(rb.tokens.front());
        }
        std::printf("phase 2 continuation (width-1, mtp session):\n%s\n",
                    text.c_str());
      }
      if (diverged < 0)
        std::printf("phase 2 (cycle-2): IDENTICAL over %d emitted tokens\n",
                    emitted);
    }

    // Phase 3: free-running MTP cycles (width controller active) vs AR
    // width-1, full token-stream comparison.
    {
      auto ar3 = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      8192, &error);
      auto mtp3 = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                       8192, &error);
      if (ar3 == nullptr || mtp3 == nullptr || !ar3->Sync(prompt, &error) ||
          !mtp3->Sync(prompt, &error)) {
        std::fprintf(stderr, "phase-3 setup failed: %s\n", error.c_str());
        return 1;
      }
      const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                         prompt.end());
      gufo::sampling::SamplerState ar_sampler(config, history);
      gufo::sampling::SamplerState mtp_sampler(config, history);
      int diverged = -1;
      std::size_t emitted = 0;
      for (int step = 0; step < steps && diverged < 0; ++step) {
        Session::DecodeResult ra;
        Session::DecodeResult rb;
        if (!ar3->DecodeStep(1, ar_sampler, &ra, &error) ||
            !mtp3->DecodeStep(8, mtp_sampler, &rb, &error)) {
          std::fprintf(stderr, "phase-3 decode failed at %d: %s\n", step,
                       error.c_str());
          return 1;
        }
        if (ra.tokens.empty() || rb.tokens.empty()) {
          if (ra.tokens.size() != rb.tokens.size())
            diverged = step;
          break;
        }
        // Stream comparison (see phase 2): free-running widths commit a
        // variable number of tokens per step.
        {
          const auto& ar_stream = ar3->Tokens();
          const auto& mtp_stream = mtp3->Tokens();
          const auto common =
              std::min(ar_stream.size(), mtp_stream.size()) - prompt.size();
          for (std::size_t i = 0; i < common; ++i) {
            const auto ai = prompt.size() + i;
            if (ar_stream[ai] != mtp_stream[ai]) {
              diverged = step;
              std::printf("phase 3 (free): step %d STREAM DIVERGES at emitted token %zu: %d(%.*s) vs %d(%.*s)\n",
                          step, i, ar_stream[ai],
                          static_cast<int>(
                              ArgmaxText(*model, ar_stream[ai]).size()),
                          ArgmaxText(*model, ar_stream[ai]).data(),
                          mtp_stream[ai],
                          static_cast<int>(
                              ArgmaxText(*model, mtp_stream[ai]).size()),
                          ArgmaxText(*model, mtp_stream[ai]).data());
              break;
            }
          }
        }
        emitted += rb.tokens.size();
        if (ra.stop || rb.stop)
          break;
      }
      const auto stats = mtp3->Statistics();
      std::printf("phase 3 mtp stats: cycles=%llu drafted=%llu accepted=%llu\n",
                  static_cast<unsigned long long>(stats.cycles),
                  static_cast<unsigned long long>(stats.drafted),
                  static_cast<unsigned long long>(stats.accepted));
      if (diverged < 0)
        std::printf("phase 3 (free): IDENTICAL over %zu emitted tokens\n",
                    emitted);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "exception: %s\n", ex.what());
    return 1;
  }
}
