// Bytes for motion.elf's vendored models, through the pump (the guest has no
// filesystem, AGENTS.md). Metadata by READ, weights by UPLOAD straight into
// the RenderingDevice buffer (rfdetr_seg.elf's pattern): weights never enter
// the guest heap. Only valid inside a pump job.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

namespace motion_io {

// The whole file (small ones: style files, the MotionBricks support
// component), in 16 MiB READs.
bool read_file(const std::string &path, std::vector<uint8_t> &out);

// gguf_init_from_callback over pump READs.
gguf_context *gguf_open(const std::string &path, gguf_init_params params);

// Allocate `ctx`'s tensors (from a no_alloc gguf_open) in one buffer of
// `buft` and fill each from the file: UPLOAD into an RD buffer, READ into a
// CPU one (ggml_backend_rd_tensor_upload chooses). Returns the buffer, or
// nullptr with `err` set.
ggml_backend_buffer_t upload_tensors(const std::string &path, gguf_context *g, ggml_context *ctx,
		ggml_backend_buffer_type_t buft, std::string &err, size_t *bytes = nullptr, int *count = nullptr);

// The ggml-rd device's backend (a new instance per call: each model owns and
// frees its own) and buffer type.
ggml_backend_t rd_backend();
ggml_backend_buffer_type_t rd_buft();

} // namespace motion_io
