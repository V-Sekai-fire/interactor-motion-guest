# motion.elf: Kimodo-SOMA and MotionBricks-G1 on ggml-rd, with the weights reaching the RD buffer by the pump's UPLOAD.
# Include it after cmake/guest_runtime.cmake with GUEST_RUNTIME_GGML on.
get_filename_component(MOTION_GUEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_k ${MOTION_GUEST_ROOT}/vendor/kimodo-ggml/src)
set(_mb ${MOTION_GUEST_ROOT}/vendor/motion-bricks-ggml)

add_stage_elf(motion
	${MOTION_GUEST_ROOT}/guest/motion/main.cpp
	${MOTION_GUEST_ROOT}/guest/motion/motion_io.cpp
	${MOTION_GUEST_ROOT}/guest/motion/kimodo_weights.cpp
	${MOTION_GUEST_ROOT}/guest/motion/mb_runtime.cpp
	${MOTION_GUEST_ROOT}/guest/motion/retarget.cpp
	${_k}/denoiser.cpp
	${_k}/diffusion.cpp
	${_k}/motion_rep.cpp
	${_mb}/src/agent.cpp
	${_mb}/src/capi.cpp
	${_mb}/src/decoder.cpp
	${_mb}/src/error.cpp
	${_mb}/src/model.cpp
	${_mb}/src/motion_rep.cpp
	${_mb}/src/planner.cpp
	${_mb}/src/pose.cpp
	${_mb}/src/root.cpp
	${_mb}/src/style.cpp
)
target_include_directories(motion PRIVATE ${GUEST_RUNTIME_ROOT}/guest ${MOTION_GUEST_ROOT}/guest/motion
	${_k} ${_mb}/include ${_mb}/src)
target_compile_definitions(motion PRIVATE MOTIONBRICKS_HAVE_GGML=1 MOTIONBRICKS_GUEST_IO=1 MOTIONBRICKS_BUILD=1)
target_compile_options(motion PRIVATE -ffp-contract=off)
target_link_libraries(motion PRIVATE ggml_rd pump ggml ggml-base)
