// Engine-level repro for the session_test --sampling-only interleaved-pair
// failure (issue #9): two AR sessions alternate DecodeStep(1) on one shared
// executor. Session A's per-step frontiers must match A's serial run
// bit-exactly; the first differing step names the leak.
//
// Usage: interleave_probe --model FIRST.gguf [--steps N]

#include <algorithm>
#include <array>
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

struct CaseSpec {
  std::string_view name;
  std::string prompt;
  gufo::sampling::SamplingConfig config;
};

std::vector<float> Snapshot(const Session& session) {
  const auto logits = session.Logits();
  return {logits.begin(), logits.end()};
}

struct StepRecord {
  std::vector<float> logits;
  std::vector<gufo::sampling::TokenId> tokens;
};

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("probe start\n");
  std::string_view model_path;
  std::string_view mtp_path;
  int steps = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--model" && i + 1 < argc)
      model_path = argv[++i];
    else if (arg == "--steps" && i + 1 < argc)
      steps = std::atoi(argv[++i]);
    else if (arg == "--mtp-model" && i + 1 < argc)
      mtp_path = argv[++i];
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "Usage: interleave_probe --model FIRST.gguf\n");
    return 2;
  }
  try {
    std::string error;
    std::printf("loading model...\n");
    gufo::models::qwen38_flash_next::ModelOptions options{
        .max_context = 8192};
    if (!mtp_path.empty()) {
      options.mtp_model_path = std::string(mtp_path);
      options.max_draft_tokens = 7;
    }
    auto model = Model::Load(std::string(model_path), options, &error);
    if (model == nullptr) {
      std::fprintf(stderr, "model load failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("model loaded\n");

    // The two cases the session test pairs at offset 2 (both sampled).
    CaseSpec a{"temperature",
               "Sampling temperature: Continue red, blue, blue, red,",
               {.temperature = 1.0F, .seed = 0}};
    CaseSpec b{"temperature-hot",
               "Sampling temperature-hot: Continue red, blue, blue, red,",
               {.temperature = 2.0F, .seed = 808}};

    const auto encode = [&](const std::string& text) {
      return model->Tokenize(text);
    };
    const auto prompt_a = encode(a.prompt);
    const auto prompt_b = encode(b.prompt);
    const auto as_ids = [](const std::vector<std::int32_t>& tokens) {
      return std::vector<gufo::sampling::TokenId>(tokens.begin(), tokens.end());
    };
    const auto history_a = as_ids(prompt_a);
    const auto history_b = as_ids(prompt_b);

    // Phase S: session A alone.
    StepRecord serial_a;
    {
      auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                          6145, &error);
      if (session == nullptr || !session->Sync(prompt_a, &error)) {
        std::fprintf(stderr, "serial setup failed: %s\n", error.c_str());
        return 1;
      }
      gufo::sampling::SamplerState sampler(a.config, history_a);
      for (int step = 0; step < steps; ++step) {
        Session::DecodeResult result;
        if (!session->DecodeStep(1, sampler, &result, &error) ||
            result.tokens.empty()) {
          std::fprintf(stderr, "serial decode failed at %d: %s\n", step,
                       error.c_str());
          return 1;
        }
        const auto frontier = Snapshot(*session);
        serial_a.logits.insert(serial_a.logits.end(), frontier.begin(),
                               frontier.end());
        serial_a.tokens.push_back(result.tokens.front());
      }
      std::printf("serial A tokens:");
      for (auto t : serial_a.tokens) std::printf(" %d", t);
      std::printf("\n");
    }

    // Phase S2: session A alone again (fresh session, same sampler inputs)
    // — the determinism baseline.
    {
      auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                          6145, &error);
      if (session == nullptr || !session->Sync(prompt_a, &error)) {
        std::fprintf(stderr, "baseline setup failed: %s\n", error.c_str());
        return 1;
      }
      gufo::sampling::SamplerState sampler(a.config, history_a);
      double worst = 0.0;
      for (int step = 0; step < steps; ++step) {
        Session::DecodeResult result;
        if (!session->DecodeStep(1, sampler, &result, &error) ||
            result.tokens.empty() || result.tokens.front() != serial_a.tokens[step]) {
          std::fprintf(stderr, "baseline decode differs at %d\n", step);
          return 1;
        }
        const auto again = Snapshot(*session);
        const auto size = again.size();
        for (std::size_t i = 0; i < size; ++i) {
          worst = std::max(worst,
                           std::abs(static_cast<double>(again[i]) -
                                    static_cast<double>(
                                        serial_a.logits[step * size + i])));
        }
      }
      std::printf("baseline rerun: max|d|=%.6g (0 = deterministic)\n", worst);
    }

    // Phase I: A interleaved with B on the shared executor.
    StepRecord interleaved_a;
    {
      auto a2 = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     6145, &error);
      auto b2 = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     6145, &error);
      if (a2 == nullptr || b2 == nullptr || !a2->Sync(prompt_a, &error) ||
          !b2->Sync(prompt_b, &error)) {
        std::fprintf(stderr, "interleaved setup failed: %s\n", error.c_str());
        return 1;
      }
      gufo::sampling::SamplerState sampler_a(a.config, history_a);
      gufo::sampling::SamplerState sampler_b(b.config, history_b);
      std::vector<gufo::sampling::TokenId> tokens_b;
      for (int step = 0; step < steps; ++step) {
        Session::DecodeResult ra;
        Session::DecodeResult rb;
        if (!a2->DecodeStep(1, sampler_a, &ra, &error) || ra.tokens.empty()) {
          std::fprintf(stderr, "interleaved A decode failed at %d: %s\n", step,
                       error.c_str());
          return 1;
        }
        const auto frontier = Snapshot(*a2);
        interleaved_a.logits.insert(interleaved_a.logits.end(), frontier.begin(),
                                    frontier.end());
        interleaved_a.tokens.push_back(ra.tokens.front());
        if (!b2->DecodeStep(1, sampler_b, &rb, &error) || rb.tokens.empty()) {
          std::fprintf(stderr, "interleaved B decode failed at %d: %s\n", step,
                       error.c_str());
          return 1;
        }
        tokens_b.push_back(rb.tokens.front());
      }
      std::printf("interleaved B tokens:");
      for (auto t : tokens_b) std::printf(" %d", t);
      std::printf("\n");
    }

    // Compare A: serial vs interleaved, per step.
    const auto vocab = serial_a.logits.size() / steps;
    int first_bad = -1;
    for (int step = 0; step < steps; ++step) {
      double worst = 0.0;
      std::size_t differing = 0;
      const auto base = static_cast<std::size_t>(step) * vocab;
      for (std::size_t i = 0; i < vocab; ++i) {
        const double d =
            std::abs(static_cast<double>(serial_a.logits[base + i]) -
                     static_cast<double>(interleaved_a.logits[base + i]));
        if (d != 0.0) ++differing;
        worst = std::max(worst, d);
      }
      const bool tokens_match =
          serial_a.tokens[step] == interleaved_a.tokens[step];
      std::printf("step %d: max|d|=%.6g differing=%zu tokens %d vs %d%s\n",
                  step, worst, differing, serial_a.tokens[step],
                  interleaved_a.tokens[step], tokens_match ? "" : " TOKEN-FLIP");
      if (worst != 0.0 && first_bad < 0) first_bad = step;
    }
    std::printf("%s (first differing step: %d)\n",
                first_bad < 0 ? "ENGINE CLEAN" : "ENGINE REPRO", first_bad);

    // Phase B: the SERVING surface. The scheduler's batched plan selects a
    // token per request from the cached frontier, then advances the pair
    // through Session::EvaluateBatch — a ForwardBatch with two rows.
    {
      auto a2 = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     6145, &error);
      auto b2 = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                     6145, &error);
      if (a2 == nullptr || b2 == nullptr || !a2->Sync(prompt_a, &error) ||
          !b2->Sync(prompt_b, &error)) {
        std::fprintf(stderr, "batched-advance setup failed: %s\n",
                     error.c_str());
        return 1;
      }
      gufo::sampling::SamplerState sampler_a(a.config, history_a);
      gufo::sampling::SamplerState sampler_b(b.config, history_b);
      StepRecord batched_a;
      std::vector<gufo::sampling::TokenId> tokens_b;
      for (int step = 0; step < steps; ++step) {
        const auto token_a = sampler_a.Sample(a2->Logits());
        const auto token_b = sampler_b.Sample(b2->Logits());
        sampler_a.Accept(token_a);
        sampler_b.Accept(token_b);
        Session::BatchOutcome oa;
        Session::BatchOutcome ob;
        std::array<Session::AdvanceRequest, 2> requests{
            {{a2.get(), static_cast<std::int32_t>(token_a), &oa},
             {b2.get(), static_cast<std::int32_t>(token_b), &ob}}};
        if (!Session::EvaluateBatch(requests, &error) || !oa.completed ||
            !ob.completed) {
          std::fprintf(stderr, "batched advance failed at %d: %s / %s\n", step,
                       oa.error.c_str(), ob.error.c_str());
          return 1;
        }
        const auto frontier = Snapshot(*a2);
        batched_a.logits.insert(batched_a.logits.end(), frontier.begin(),
                                frontier.end());
        batched_a.tokens.push_back(token_a);
        tokens_b.push_back(token_b);
      }
      std::printf("batched-advance B tokens:");
      for (auto t : tokens_b) std::printf(" %d", static_cast<int>(t));
      std::printf("\n");
      first_bad = -1;
      for (int step = 0; step < steps; ++step) {
        double worst = 0.0;
        std::size_t differing = 0;
        const auto base = static_cast<std::size_t>(step) * vocab;
        for (std::size_t i = 0; i < vocab; ++i) {
          const double d =
              std::abs(static_cast<double>(serial_a.logits[base + i]) -
                       static_cast<double>(batched_a.logits[base + i]));
          if (d != 0.0) ++differing;
          worst = std::max(worst, d);
        }
        const bool tokens_match =
            serial_a.tokens[step] == batched_a.tokens[step];
        std::printf(
            "batch step %d: max|d|=%.6g differing=%zu tokens %d vs %d%s\n",
            step, worst, differing, serial_a.tokens[step],
            batched_a.tokens[step], tokens_match ? "" : " TOKEN-FLIP");
        if (worst != 0.0 && first_bad < 0) first_bad = step;
      }
      std::printf("%s\n", first_bad < 0 ? "BATCHED-ADVANCE CLEAN"
                                        : "BATCHED-ADVANCE REPRO");
    }

    // Phase C: the batched MTP serving surface — two speculative sessions
    // advance through Session::DecodeBatch per cycle (DraftCatchUpBatch,
    // batched draft heads/bodies, ForwardBatch), compared against serial
    // per-cycle DecodeStep runs of the same two sessions.
    {
      auto mk = [&](gufo::core::SessionMode mode) {
        return model->CreateSession(mode, 6145, &error);
      };
      auto sa = mk(gufo::core::SessionMode::kSpeculative);
      auto sb = mk(gufo::core::SessionMode::kSpeculative);
      if (sa == nullptr || sb == nullptr || !sa->Sync(prompt_a, &error) ||
          !sb->Sync(prompt_b, &error)) {
        std::fprintf(stderr, "mtp serial setup failed: %s\n", error.c_str());
        return 1;
      }
      gufo::sampling::SamplerState sa_sampler(a.config, history_a);
      gufo::sampling::SamplerState sb_sampler(b.config, history_b);
      std::vector<gufo::sampling::TokenId> serial_stream;
      std::uint64_t serial_drafted = 0;
      std::uint64_t serial_accepted = 0;
      for (int cycle = 0; cycle < steps && serial_stream.size() < 8; ++cycle) {
        const auto before = sa->Statistics();
        const auto budget =
            std::min<std::size_t>(8 - serial_stream.size(), 8);
        Session::DecodeResult result;
        if (!sa->DecodeStep(budget, sa_sampler, &result, &error) ||
            result.tokens.empty()) {
          std::fprintf(stderr, "mtp serial decode failed at %d: %s\n",
                       cycle, error.c_str());
          return 1;
        }
        const auto after = sa->Statistics();
        serial_drafted += after.drafted - before.drafted;
        serial_accepted += after.accepted - before.accepted;
        serial_stream.insert(serial_stream.end(), result.tokens.begin(),
                             result.tokens.end());
        if (result.stop) break;
      }
      std::printf("mtp serial A: drafted=%llu accepted=%llu tokens:",
                  static_cast<unsigned long long>(serial_drafted),
                  static_cast<unsigned long long>(serial_accepted));
      for (auto t : serial_stream) std::printf(" %d", static_cast<int>(t));
      std::printf("\n");
      // Batched: both sessions per cycle through Session::DecodeBatch.
      auto ba = mk(gufo::core::SessionMode::kSpeculative);
      auto bb = mk(gufo::core::SessionMode::kSpeculative);
      if (ba == nullptr || bb == nullptr || !ba->Sync(prompt_a, &error) ||
          !bb->Sync(prompt_b, &error)) {
        std::fprintf(stderr, "mtp batched setup failed: %s\n", error.c_str());
        return 1;
      }
      gufo::sampling::SamplerState ba_sampler(a.config, history_a);
      gufo::sampling::SamplerState bb_sampler(b.config, history_b);
      std::vector<gufo::sampling::TokenId> batched_stream_a;
      std::vector<gufo::sampling::TokenId> batched_stream_b;
      std::uint64_t batched_drafted = 0;
      std::uint64_t batched_accepted = 0;
      for (int cycle = 0;
           cycle < steps && batched_stream_a.size() < 8; ++cycle) {
        const auto before = ba->Statistics();
        const auto budget =
            std::min<std::size_t>(8 - batched_stream_a.size(), 8);
        Session::DecodeResult ra;
        Session::DecodeResult rb;
        Session::BatchOutcome oa;
        Session::BatchOutcome ob;
        const Session::DecodeRequest lane_a{ba.get(), budget, &ba_sampler,
                                            &ra, true, &oa};
        const Session::DecodeRequest lane_b{bb.get(), budget, &bb_sampler,
                                            &rb, true, &ob};
        const std::array<Session::DecodeRequest, 2> requests{{lane_a, lane_b}};
        if (!Session::DecodeBatch(requests, &error) || !oa.completed ||
            !ob.completed) {
          std::fprintf(stderr,
                       "mtp batched decode failed at %d: %s / %s\n",
                       cycle, oa.error.c_str(), ob.error.c_str());
          return 1;
        }
        const auto after = ba->Statistics();
        batched_drafted += after.drafted - before.drafted;
        batched_accepted += after.accepted - before.accepted;
        batched_stream_a.insert(batched_stream_a.end(), ra.tokens.begin(),
                                ra.tokens.end());
        batched_stream_b.insert(batched_stream_b.end(), rb.tokens.begin(),
                                rb.tokens.end());
        if (ra.stop) break;
      }
      std::printf("mtp batched A: drafted=%llu accepted=%llu tokens:",
                  static_cast<unsigned long long>(batched_drafted),
                  static_cast<unsigned long long>(batched_accepted));
      for (auto t : batched_stream_a) std::printf(" %d", static_cast<int>(t));
      std::printf("\n");
      const bool same = serial_stream == batched_stream_a &&
                        serial_drafted == batched_drafted &&
                        serial_accepted == batched_accepted;
      std::printf("%s\n",
                  same ? "MTP BATCH CLEAN" : "MTP BATCH REPRO");
    }    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "exception: %s\n", ex.what());
    return 1;
  }
}
