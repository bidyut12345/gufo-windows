# Ornith-1.5-35B (qwen35moe) model package. The scalar oracle is built only
# by explicit validation targets, not by production or routine checks.
add_library(gufo_qwen35moe STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_qwen35moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen35moe PUBLIC gufo_core Threads::Threads)

add_library(gufo_qwen35moe_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen35moe_reference PUBLIC gufo_qwen35moe)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(Q35_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The vendored tier is model-agnostic: this package
  # shares Flash-Next's build instead of duplicating its device code.
  set(Q35_HIP_SOURCES
    ${Q35_ROCM_DIR}/kernels.hip.cpp
    ${Q35_ROCM_DIR}/w8a8_wave64.hip
    ${Q35_ROCM_DIR}/blaslt.cpp
    ${Q35_ROCM_DIR}/device_model.cpp
    ${Q35_ROCM_DIR}/executor.cpp
    ${Q35_ROCM_DIR}/batch.cpp
    engine.cpp)
  target_sources(gufo_qwen35moe PRIVATE ${Q35_HIP_SOURCES})
  set_source_files_properties(${Q35_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${Q35_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen35moe PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen35moe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen35moe SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen35moe PUBLIC
    gufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)
endif()
