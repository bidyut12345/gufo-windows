// Process-local submission/completion failures; never resets or loses the GPU.
// Linked only into the model-local adapter test, not the model or serving
// library.
#if !defined(_WIN32)
#include <dlfcn.h>
#else
#include <windows.h>
#endif
#include <hip/hip_runtime.h>

#include <atomic>

namespace {
std::atomic<bool> fail_next{false};
std::atomic<bool> fail_next_wait{false};

#if defined(_WIN32)
void* ResolveHipProc(const char* name) {
  HMODULE mod = nullptr;
  if (!GetModuleHandleExA(0, "amdhip64_7.dll", &mod) || !mod)
    GetModuleHandleExA(0, "amdhip64.dll", &mod);
  return mod ? reinterpret_cast<void*>(GetProcAddress(mod, name)) : nullptr;
}
#endif
}  // namespace
void FailNextResetSubmission() {
  fail_next = true;
}
void FailNextTransferWait() {
  fail_next_wait = true;
}

extern "C" hipError_t hipMemsetAsync(void* pointer, int value,
                                     std::size_t bytes, hipStream_t stream) {
  if (fail_next.exchange(false))
    return hipErrorInvalidValue;
  using Function = hipError_t (*)(void*, int, std::size_t, hipStream_t);
#if !defined(_WIN32)
  static const auto next =
      reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipMemsetAsync"));
#else
  static const auto next =
      reinterpret_cast<Function>(ResolveHipProc("hipMemsetAsync"));
#endif
  return next ? next(pointer, value, bytes, stream) : hipErrorUnknown;
}

extern "C" hipError_t hipEventSynchronize(hipEvent_t event) {
  using Function = hipError_t (*)(hipEvent_t);
#if !defined(_WIN32)
  static const auto next =
      reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipEventSynchronize"));
#else
  static const auto next =
      reinterpret_cast<Function>(ResolveHipProc("hipEventSynchronize"));
#endif
  const auto result = next ? next(event) : hipErrorUnknown;
  // Drain the real copy before reporting an injected completion error.
  return fail_next_wait.exchange(false) ? hipErrorInvalidValue : result;
}
