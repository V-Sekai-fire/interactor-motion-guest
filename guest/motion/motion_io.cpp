#include "motion_io.h"

#include <algorithm>
#include <cstring>

#include "ggml-rd.h"
#include "pump/pump.h"

namespace motion_io {

namespace {

constexpr uint64_t kChunk = uint64_t(16) << 20; // the host's 16 MiB view

size_t pump_reader(void *userdata, void *output, uint64_t offset, size_t len) {
	const std::string &path = *static_cast<const std::string *>(userdata);
	size_t done = 0;
	while (done < len) {
		const uint64_t n = std::min<uint64_t>(len - done, kChunk);
		std::vector<uint8_t> v = pump::read(path, offset + done, n);
		std::memcpy(static_cast<uint8_t *>(output) + done, v.data(), v.size());
		done += v.size();
		if (v.size() != n) {
			break;
		}
	}
	return done;
}

} // namespace

bool read_file(const std::string &path, std::vector<uint8_t> &out) {
	out.clear();
	for (uint64_t off = 0;; off += kChunk) {
		std::vector<uint8_t> v = pump::read(path, off, kChunk);
		out.insert(out.end(), v.begin(), v.end());
		if (v.size() < kChunk) {
			break;
		}
	}
	return !out.empty();
}

gguf_context *gguf_open(const std::string &path, gguf_init_params params) {
	std::string p = path;
	return gguf_init_from_callback(pump_reader, &p, size_t(kChunk), UINT64_MAX, params);
}

ggml_backend_buffer_t upload_tensors(const std::string &path, gguf_context *g, ggml_context *ctx,
		ggml_backend_buffer_type_t buft, std::string &err, size_t *bytes, int *count) {
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
	if (buf == nullptr) {
		err = path + ": buffer allocation failed";
		return nullptr;
	}
	const size_t data_off = gguf_get_data_offset(g);
	size_t total = 0;
	int n = 0;
	for (ggml_tensor *t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
		const int64_t idx = gguf_find_tensor(g, ggml_get_name(t));
		if (idx < 0) {
			err = path + ": tensor " + ggml_get_name(t) + " missing from the index";
			ggml_backend_buffer_free(buf);
			return nullptr;
		}
		const size_t nb = ggml_nbytes(t);
		if (!ggml_backend_rd_tensor_upload(t, 0, path, data_off + gguf_get_tensor_offset(g, idx), nb)) {
			err = path + ": upload of " + ggml_get_name(t) + " failed: " + ggml_backend_rd_last_error();
			ggml_backend_buffer_free(buf);
			return nullptr;
		}
		total += nb;
		++n;
	}
	if (bytes) {
		*bytes = total;
	}
	if (count) {
		*count = n;
	}
	return buf;
}

ggml_backend_t rd_backend() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		return nullptr;
	}
	return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
}

ggml_backend_buffer_type_t rd_buft() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		return nullptr;
	}
	return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, 0));
}

} // namespace motion_io
