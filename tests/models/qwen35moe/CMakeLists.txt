# Small operator checks are the default. Full-model qualification is built
# explicitly so routine iterations stay focused.
add_custom_target(qwen35moe_tests)

function(add_qwen35moe_test name library labels)
  set(target qwen35moe_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  # The AMD driver's System32 runtime shadows the build's; stage it beside
  # the executables (the exe directory precedes System32 in the search).
  gufo_stage_hip_runtime(${target})
  add_test(NAME qwen35moe.${name} COMMAND ${target})
  set_tests_properties(qwen35moe.${name} PROPERTIES
    LABELS "qwen35moe;${labels}")
  add_dependencies(qwen35moe_tests ${target})
endfunction()

add_qwen35moe_test(config gufo_qwen35moe "cpu")
add_qwen35moe_test(mtp_sampling gufo_core "cpu")

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS attention_ops gdn_ops)
    add_qwen35moe_test(${name} gufo_qwen35moe "hip;gpu;gfx1151")
  endforeach()
endif()

add_executable(qwen35moe_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen35moe_reference_probe PRIVATE gufo_qwen35moe_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_executable(qwen35moe_gpu_compare EXCLUDE_FROM_ALL gpu_compare.cpp)
target_link_libraries(qwen35moe_gpu_compare PRIVATE gufo_qwen35moe gufo_qwen35moe_reference)
set_source_files_properties(gpu_compare.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_executable(qwen35moe_vec_bench EXCLUDE_FROM_ALL vec_bench.cpp)
target_link_libraries(qwen35moe_vec_bench PRIVATE gufo_qwen35moe)
target_include_directories(qwen35moe_vec_bench SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/kernels/rocm/mmq)
