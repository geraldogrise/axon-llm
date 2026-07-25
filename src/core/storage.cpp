#include "axon/storage.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace axon {

Storage::Storage(std::size_t nbytes, Device device) : nbytes_(nbytes), device_(device) {
    if (device != Device::CPU) {
        throw std::invalid_argument("Storage: only Device::CPU is supported in this phase");
    }
    if (nbytes_ == 0) {
        data_ = nullptr;
        return;
    }
    // calloc zeroes the memory - guarantees that zeros() actually starts at zero.
    data_ = std::calloc(nbytes_, 1);
    if (data_ == nullptr) {
        throw std::bad_alloc();
    }
}

Storage::~Storage() {
    std::free(data_);
}

Storage::Storage(Storage&& other) noexcept
    : data_(other.data_), nbytes_(other.nbytes_), device_(other.device_) {
    other.data_ = nullptr;
    other.nbytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        std::free(data_);
        data_ = other.data_;
        nbytes_ = other.nbytes_;
        device_ = other.device_;
        other.data_ = nullptr;
        other.nbytes_ = 0;
    }
    return *this;
}

}  // namespace axon
