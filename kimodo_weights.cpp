// kimodo::detail::ggml_motion_weights for the guest: the class of
// vendor/kimodo-ggml/src/ggml_weights.hpp, whose upstream ggml_weights.cpp
// opens the GGUF with fopen/ifstream and picks Vulkan, CUDA or the CPU. Here
// the metadata comes through the pump (READ), every tensor is streamed by the
// host into the ggml-rd RenderingDevice buffer (UPLOAD), and the backend is
// ggml-rd (AGENTS.md rule 10). What upstream checks before loading (the
// architecture, format version, skeleton, dims, F32-only tensors) is checked
// here from the same keys; upstream's hardened header parser (gguf.cpp) is
// not in the guest, ggml's own gguf reader is.
#include "ggml_weights.hpp"

#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "motion_io.h"
#include "skeleton.hpp"

namespace kimodo::detail {

namespace {

bool str_key(gguf_context *g, const char *key, std::string &out) {
	const int64_t k = gguf_find_key(g, key);
	if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_STRING) {
		return false;
	}
	out = gguf_get_val_str(g, k);
	return true;
}

bool uint_key(gguf_context *g, const char *key, uint64_t &out) {
	const int64_t k = gguf_find_key(g, key);
	if (k < 0) {
		return false;
	}
	switch (gguf_get_kv_type(g, k)) {
		case GGUF_TYPE_UINT32: out = gguf_get_val_u32(g, k); return true;
		case GGUF_TYPE_UINT64: out = gguf_get_val_u64(g, k); return true;
		case GGUF_TYPE_INT32: out = uint64_t(gguf_get_val_i32(g, k)); return true;
		case GGUF_TYPE_INT64: out = uint64_t(gguf_get_val_i64(g, k)); return true;
		default: return false;
	}
}

} // namespace

kimodo::expected<std::unique_ptr<ggml_motion_weights>, std::string> ggml_motion_weights::load(std::string_view path) {
	const std::string p(path);
	auto result = std::unique_ptr<ggml_motion_weights>(new ggml_motion_weights);
	gguf_init_params params{ true, &result->context_ };
	result->gguf_ = motion_io::gguf_open(p, params);
	if (!result->gguf_ || !result->context_) {
		return kimodo::unexpected("motion GGUF did not parse: " + p);
	}
	std::string arch;
	uint64_t format = 0, motion_dim = 0, body_dim = 0;
	if (!str_key(result->gguf_, "general.architecture", arch) || arch != "kimodo-motion") {
		return kimodo::unexpected("not a Kimodo motion GGUF: " + p);
	}
	if (!uint_key(result->gguf_, "kimodo.format_version", format) || format != 1) {
		return kimodo::unexpected("unsupported kimodo.format_version");
	}
	if (!str_key(result->gguf_, "kimodo.skeleton", result->skeleton_) || !find_skeleton(result->skeleton_)) {
		return kimodo::unexpected("unknown kimodo.skeleton");
	}
	if (!uint_key(result->gguf_, "kimodo.motion_dim", motion_dim) || !uint_key(result->gguf_, "kimodo.body_dim", body_dim) ||
			motion_dim != find_skeleton(result->skeleton_)->motion_dim() || body_dim != motion_dim - 5) {
		return kimodo::unexpected("kimodo.motion_dim / body_dim do not match the skeleton");
	}
	result->motion_dim_ = size_t(motion_dim);
	result->body_dim_ = size_t(body_dim);
	for (ggml_tensor *t = ggml_get_first_tensor(result->context_); t != nullptr; t = ggml_get_next_tensor(result->context_, t)) {
		if (t->type != GGML_TYPE_F32) {
			return kimodo::unexpected("motion GGUF contains an invalid non-F32 tensor");
		}
	}
	result->backend_ = motion_io::rd_backend();
	if (!result->backend_) {
		return kimodo::unexpected("no ggml-rd device (motion_open attaches one)");
	}
	std::string err;
	result->buffer_ = motion_io::upload_tensors(p, result->gguf_, result->context_, motion_io::rd_buft(), err);
	if (!result->buffer_) {
		return kimodo::unexpected(err);
	}
	return result;
}

ggml_motion_weights::~ggml_motion_weights() {
	if (buffer_) ggml_backend_buffer_free(buffer_);
	if (gguf_) gguf_free(gguf_);
	if (context_) ggml_free(context_);
	if (backend_) ggml_backend_free(backend_);
}

ggml_tensor *ggml_motion_weights::tensor(std::string_view name) const {
	return context_ ? ggml_get_tensor(context_, std::string(name).c_str()) : nullptr;
}

kimodo::expected<std::vector<float>, std::string> ggml_motion_weights::f32_values(std::string_view name) const {
	auto *value = tensor(name);
	if (!value || value->type != GGML_TYPE_F32) return kimodo::unexpected("missing F32 GGML tensor: " + std::string(name));
	std::vector<float> result(static_cast<size_t>(ggml_nelements(value)));
	ggml_backend_tensor_get(value, result.data(), 0, result.size() * sizeof(float));
	return result;
}

} // namespace kimodo::detail
