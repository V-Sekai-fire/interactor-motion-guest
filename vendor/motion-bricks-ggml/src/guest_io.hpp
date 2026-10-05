// interactor-motion-guest addition (not upstream). model.cpp and style.cpp reach
// the filesystem in four places: is_regular_file, is_directory, absolute and
// gguf_init_from_file. A godot-sandbox guest has no filesystem (openat is
// EBADF, AGENTS.md), so under MOTIONBRICKS_GUEST_IO those four go to the
// embedder (guest/motion/mb_runtime.cpp), which reads the bytes through the
// pump. Without it they are exactly upstream's calls.
#pragma once

#include <filesystem>
#include <system_error>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <gguf.h>
#endif

#if defined(MOTIONBRICKS_GUEST_IO)
namespace motionbricks::guest_io {
bool is_regular_file(const std::filesystem::path &path);
bool is_directory(const std::filesystem::path &path, std::error_code &error);
std::filesystem::path absolute(const std::filesystem::path &path, std::error_code &error);
#if defined(MOTIONBRICKS_HAVE_GGML)
gguf_context *gguf_open(const char *path, gguf_init_params params);
#endif
} // namespace motionbricks::guest_io
#define MB_FS motionbricks::guest_io
#define MB_GGUF_OPEN motionbricks::guest_io::gguf_open
#else
#define MB_FS std::filesystem
#define MB_GGUF_OPEN gguf_init_from_file
#endif
