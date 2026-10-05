#pragma once

#include <cstdint>
#include "../include/kimodo/result.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace kimodo::detail {

struct gguf_file {
    std::unordered_map<std::string, std::string> strings;
    std::unordered_map<std::string, std::uint64_t> uints;
    std::uint64_t tensor_count = 0;
    std::unordered_set<std::string> tensor_names;
};

kimodo::expected<gguf_file, std::string> read_gguf_header(std::string_view path);
kimodo::expected<void, std::string> validate_motion_gguf(const gguf_file &file);
kimodo::expected<void, std::string> validate_text_gguf(const gguf_file &file);

} // namespace kimodo::detail
