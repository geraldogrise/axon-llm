#pragma once

#include <cstdint>
#include <string_view>

namespace axon {

// Where the Tensor's data lives. GPU comes in the final phases (CUDA backend).
enum class Device : std::uint8_t {
    CPU,
    CUDA,
};

constexpr std::string_view device_name(Device d) noexcept {
    switch (d) {
        case Device::CPU:
            return "cpu";
        case Device::CUDA:
            return "cuda";
    }
    return "?";
}

}  // namespace axon
