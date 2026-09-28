// MotionBricks' neural runtime for the guest. Upstream's neural_runtime.cpp
// (not compiled here) opens each component with fopen and runs it on
// ggml-cpu or ggml-vulkan. Here the four components' metadata come through
// the pump (READ) and their tensors are streamed by the host into the ggml-rd
// RenderingDevice buffer (UPLOAD); the backend is ggml-rd (AGENTS.md rule
// 10). neural_weight's lookup, including upstream's "self_attn." -> "attn."
// rename for names past ggml's 64-byte limit, is upstream's.
//
// Also the four filesystem hooks of vendor/motion-bricks-ggml/src/guest_io.hpp.
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "guest_io.hpp"
#include "motion_io.h"
#include "neural_runtime.hpp"

namespace motionbricks::guest_io {

bool is_regular_file(const std::filesystem::path &) {
	return true; // a missing file fails the READ that follows, with its name
}

bool is_directory(const std::filesystem::path &, std::error_code &error) {
	error.clear();
	return true;
}

std::filesystem::path absolute(const std::filesystem::path &path, std::error_code &error) {
	error.clear();
	return path; // host paths are already absolute
}

gguf_context *gguf_open(const char *path, gguf_init_params params) {
	return motion_io::gguf_open(path, params);
}

} // namespace motionbricks::guest_io

namespace motionbricks::detail {

class neural_runtime {
public:
	struct component {
		std::string name;
		ggml_context *context = nullptr;
		gguf_context *gguf = nullptr;
		ggml_backend_buffer_t buffer = nullptr;
	};
	ggml_backend_t backend = nullptr;
	std::vector<component> components;
	~neural_runtime() {
		for (component &c : components) {
			if (c.buffer) ggml_backend_buffer_free(c.buffer);
			if (c.gguf) gguf_free(c.gguf);
			if (c.context) ggml_free(c.context);
		}
		if (backend) ggml_backend_free(backend);
	}
};

mb_status create_neural_runtime(const std::filesystem::path &bundle, mb_device, std::uint32_t, const std::string &,
		std::shared_ptr<neural_runtime> &output, std::string &reason) {
	output.reset();
	auto runtime = std::make_shared<neural_runtime>();
	runtime->backend = motion_io::rd_backend();
	if (!runtime->backend) {
		reason = "no ggml-rd device (motion_open attaches one)";
		return MB_BACKEND_UNAVAILABLE;
	}
	constexpr std::array files{
		std::pair{ "pose", "pose.gguf" },
		std::pair{ "root", "root.gguf" },
		std::pair{ "vq-decoder", "vq-decoder.gguf" },
		std::pair{ "support", "support.gguf" },
	};
	for (const auto &[name, filename] : files) {
		const std::string path = (bundle / filename).string();
		neural_runtime::component c;
		c.name = name;
		gguf_init_params params{ true, &c.context };
		c.gguf = motion_io::gguf_open(path, params);
		if (!c.gguf || !c.context) {
			reason = "cannot initialize GGUF weights: " + path;
			return MB_INVALID_FORMAT;
		}
		c.buffer = motion_io::upload_tensors(path, c.gguf, c.context, motion_io::rd_buft(), reason);
		if (!c.buffer) {
			return MB_OUT_OF_MEMORY;
		}
		runtime->components.push_back(c);
	}
	output = std::move(runtime);
	return MB_OK;
}

ggml_backend_t neural_backend(const neural_runtime &runtime) noexcept {
	return runtime.backend;
}

ggml_tensor *neural_weight(const neural_runtime &runtime, std::string_view component, std::string_view name) noexcept {
	const auto found = std::find_if(runtime.components.begin(), runtime.components.end(),
			[&](const neural_runtime::component &item) { return item.name == component; });
	if (found == runtime.components.end()) return nullptr;
	std::string owned(name);
	if (auto *direct = ggml_get_tensor(found->context, owned.c_str())) return direct;
	if (owned.size() >= 64U) {
		const std::string needle = "self_attn.";
		if (const auto position = owned.find(needle); position != std::string::npos)
			owned.replace(position, needle.size(), "attn.");
	}
	return ggml_get_tensor(found->context, owned.c_str());
}

bool neural_copy_f32(const neural_runtime &runtime, std::string_view component, std::string_view name,
		std::vector<float> &output, std::string &reason) {
	auto *tensor = neural_weight(runtime, component, name);
	if (tensor == nullptr) {
		reason = "missing neural tensor: " + std::string(component) + ":" + std::string(name);
		return false;
	}
	if (tensor->type != GGML_TYPE_F32) {
		reason = "neural tensor is not F32: " + std::string(component) + ":" + std::string(name);
		return false;
	}
	output.resize(static_cast<std::size_t>(ggml_nelements(tensor)));
	ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
	return true;
}

} // namespace motionbricks::detail
