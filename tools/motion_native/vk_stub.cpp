// kimodo-ggml's llm_text_encoder.cpp calls ggml_backend_vk_get_device_count()
// and ggml_backend_vk_init() unconditionally at link level. This host oracle
// builds ggml without Vulkan, so the probe answers "no device" and the encoder
// takes its CPU path (the same path KIMODO_BACKEND=cpu forces).
#include <ggml-backend.h>
#include <ggml-vulkan.h>

extern "C" {
int ggml_backend_vk_get_device_count(void) {
	return 0;
}
ggml_backend_t ggml_backend_vk_init(size_t) {
	return nullptr;
}
}
