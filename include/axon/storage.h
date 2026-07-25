#pragma once

#include <cstddef>
#include <memory>

#include "axon/device.h"

namespace axon {

// Raw byte buffer that owns the memory. Several Tensors (views) can
// share the same Storage via shared_ptr: the memory is only freed when
// the last view dies (RAII). Knows neither shape nor dtype - just bytes.
class Storage {
  public:
    // Allocates `nbytes` zeroed bytes on the CPU.
    explicit Storage(std::size_t nbytes, Device device = Device::CPU);

    ~Storage();

    // Not copyable (sole owner of the bytes); movable.
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t nbytes() const noexcept { return nbytes_; }
    [[nodiscard]] Device device() const noexcept { return device_; }

  private:
    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    Device device_ = Device::CPU;
};

}  // namespace axon
