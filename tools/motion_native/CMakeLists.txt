# motion_native (Gate 10's host oracle and the Kimodo stand-in's embedder), from vendor/ and V-Sekai-fire/ggml.
#   cmake -S tools/motion_native -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release [-DGGML_ROOT=<V-Sekai-fire/ggml>]
# No Vulkan: vk_stub.cpp answers "no device" to the text encoder's Vulkan probe, which then runs on the CPU.
cmake_minimum_required(VERSION 3.25)
project(motion_native C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(MOTION_GUEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(_weft "${MOTION_GUEST_ROOT}/../.." ABSOLUTE)
set(KIMODO_SRC "${MOTION_GUEST_ROOT}/vendor/kimodo-ggml" CACHE PATH "kimodo-ggml sources (vendored at 9e62d0e)")
set(MB_SRC "${MOTION_GUEST_ROOT}/vendor/motion-bricks-ggml" CACHE PATH "motion-bricks-ggml sources (vendored at e61da1f)")
set(GGML_ROOT "${_weft}/2-contract/ggml" CACHE PATH "V-Sekai-fire/ggml checkout")

foreach(opt GGML_BUILD_TESTS GGML_BUILD_EXAMPLES GGML_BACKEND_DL BUILD_SHARED_LIBS GGML_VULKAN GGML_CUDA GGML_OPENMP)
	set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(${GGML_ROOT} ggml EXCLUDE_FROM_ALL)

set(K ${KIMODO_SRC}/src)
set(M ${MB_SRC}/src)
add_executable(motion_native motion_native.cpp vk_stub.cpp
	${K}/ggml_weights.cpp ${K}/gguf.cpp ${K}/denoiser.cpp ${K}/diffusion.cpp ${K}/motion_rep.cpp ${K}/motion_decode.cpp
	${K}/llm_text_encoder.cpp ${K}/llm_tokenizer.cpp
	${M}/agent.cpp ${M}/capi.cpp ${M}/decoder.cpp ${M}/error.cpp ${M}/model.cpp ${M}/motion_rep.cpp
	${M}/neural_runtime.cpp ${M}/planner.cpp ${M}/pose.cpp ${M}/root.cpp ${M}/style.cpp)
target_include_directories(motion_native PRIVATE ${K} ${KIMODO_SRC}/include ${M} ${MB_SRC}/include ${MOTION_GUEST_ROOT}/guest/motion)
target_compile_definitions(motion_native PRIVATE KIMODO_HAVE_GGML=1 MOTIONBRICKS_HAVE_GGML=1 MOTIONBRICKS_BUILD=1)
# The same contraction rule as motion.elf's CPU-side glue: no fused multiply-add the other side cannot form.
target_compile_options(motion_native PRIVATE -ffp-contract=off)
target_link_libraries(motion_native PRIVATE ggml ggml-cpu ggml-base)
