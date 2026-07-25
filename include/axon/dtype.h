#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace axon {

// Data types supported by the Tensor.
// We start with F32 (the standard in deep learning); the others come in later phases.
enum class DType : std::uint8_t {
    F32,  // float  (32 bits) - default
    F64,  // double (64 bits)
    I32,  // int32_t
};

// Size in bytes of a single element of the dtype.
constexpr std::size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::F32:
            return sizeof(float);
        case DType::F64:
            return sizeof(double);
        case DType::I32:
            return sizeof(std::int32_t);
    }
    return 0;
}

// Human-readable name of the dtype (for repr/logs).
constexpr std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::F32:
            return "f32";
        case DType::F64:
            return "f64";
        case DType::I32:
            return "i32";
    }
    return "?";
}

}  // namespace axon
