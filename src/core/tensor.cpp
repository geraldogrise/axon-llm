#include "axon/tensor.h"

#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace axon {

namespace {
// Product of all dimensions (number of elements).
std::int64_t product(const Shape& shape) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        if (d < 0) {
            throw std::invalid_argument("Tensor: negative dimension in shape");
        }
        n *= d;
    }
    return n;
}
}  // namespace

Strides Tensor::contiguous_strides(const Shape& shape) {
    Strides strides(shape.size());
    std::int64_t acc = 1;
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[static_cast<std::size_t>(i)] = acc;
        acc *= shape[static_cast<std::size_t>(i)];
    }
    return strides;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    if (dtype != DType::F32) {
        throw std::invalid_argument("Tensor: only DType::F32 is supported in this phase");
    }
    Tensor t;
    t.shape_ = shape;
    t.strides_ = contiguous_strides(shape);
    t.dtype_ = dtype;
    t.device_ = Device::CPU;
    t.offset_ = 0;
    const std::size_t nbytes = static_cast<std::size_t>(product(shape)) * dtype_size(dtype);
    t.storage_ = std::make_shared<Storage>(nbytes, Device::CPU);  // calloc -> already zeroed
    return t;
}

Tensor Tensor::full(const Shape& shape, float value, DType dtype) {
    Tensor t = zeros(shape, dtype);
    float* p = t.data_ptr();
    const std::int64_t n = t.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = value;
    }
    return t;
}

Tensor Tensor::ones(const Shape& shape, DType dtype) {
    return full(shape, 1.0f, dtype);
}

Tensor Tensor::from_data(const Shape& shape, const std::vector<float>& data) {
    Tensor t = zeros(shape, DType::F32);
    if (static_cast<std::int64_t>(data.size()) != t.numel()) {
        throw std::invalid_argument("Tensor::from_data: data size != numel(shape)");
    }
    std::memcpy(t.data_ptr(), data.data(), data.size() * sizeof(float));
    return t;
}

Tensor Tensor::randn(const Shape& shape, std::uint64_t seed, float mean, float stddev) {
    Tensor t = zeros(shape, DType::F32);
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    float* p = t.data_ptr();
    const std::int64_t n = t.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = dist(gen);
    }
    return t;
}

Tensor Tensor::rand(const Shape& shape, std::uint64_t seed, float lo, float hi) {
    Tensor t = zeros(shape, DType::F32);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    float* p = t.data_ptr();
    const std::int64_t n = t.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = dist(gen);
    }
    return t;
}

std::int64_t Tensor::numel() const noexcept {
    if (shape_.empty()) {
        return storage_ ? 1 : 0;  // scalar (0-dim) has 1 element
    }
    std::int64_t n = 1;
    for (std::int64_t d : shape_) {
        n *= d;
    }
    return n;
}

bool Tensor::is_contiguous() const noexcept {
    return strides_ == contiguous_strides(shape_);
}

std::int64_t Tensor::flat_index(const Shape& idx) const {
    if (static_cast<std::int64_t>(idx.size()) != ndim()) {
        throw std::invalid_argument("Tensor: number of indices != ndim");
    }
    std::int64_t off = offset_;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= shape_[i]) {
            throw std::out_of_range("Tensor: index out of bounds");
        }
        off += idx[i] * strides_[i];
    }
    return off;
}

float* Tensor::data_ptr() noexcept {
    return static_cast<float*>(storage_->data());
}

const float* Tensor::data_ptr() const noexcept {
    return static_cast<const float*>(storage_->data());
}

float& Tensor::at(const Shape& idx) {
    return data_ptr()[flat_index(idx)];
}

float Tensor::at(const Shape& idx) const {
    return data_ptr()[flat_index(idx)];
}

float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item(): tensor does not have exactly 1 element");
    }
    return data_ptr()[offset_];
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("reshape requires a contiguous tensor (use contiguous() first)");
    }
    if (product(new_shape) != numel()) {
        throw std::invalid_argument("reshape: incompatible numel");
    }
    Tensor t = *this;  // shares the same Storage (view)
    t.shape_ = new_shape;
    t.strides_ = contiguous_strides(new_shape);
    return t;
}

Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim()) {
        throw std::out_of_range("transpose: invalid dimension");
    }
    Tensor t = *this;  // view: just swaps shape/strides, without copying data
    std::swap(t.shape_[static_cast<std::size_t>(dim0)], t.shape_[static_cast<std::size_t>(dim1)]);
    std::swap(t.strides_[static_cast<std::size_t>(dim0)],
              t.strides_[static_cast<std::size_t>(dim1)]);
    return t;
}

Tensor Tensor::permute(const Shape& dims) const {
    if (static_cast<std::int64_t>(dims.size()) != ndim()) {
        throw std::invalid_argument("permute: number of axes != ndim");
    }
    std::vector<bool> seen(dims.size(), false);
    Tensor t = *this;  // view: reorders shape/strides without copying
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const std::int64_t d = dims[i];
        if (d < 0 || d >= ndim() || seen[static_cast<std::size_t>(d)]) {
            throw std::invalid_argument("permute: invalid or repeated axes");
        }
        seen[static_cast<std::size_t>(d)] = true;
        t.shape_[i] = shape_[static_cast<std::size_t>(d)];
        t.strides_[i] = strides_[static_cast<std::size_t>(d)];
    }
    return t;
}

Tensor Tensor::slice(std::int64_t dim, std::int64_t start, std::int64_t len) const {
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("slice: invalid dimension");
    }
    const auto ud = static_cast<std::size_t>(dim);
    if (start < 0 || start + len > shape_[ud]) {
        throw std::out_of_range("slice: range out of bounds");
    }
    Tensor t = *this;  // view: adjusts offset and the dimension size
    t.offset_ = offset_ + start * strides_[ud];
    t.shape_[ud] = len;
    return t;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    Tensor out = zeros(shape_, dtype_);
    // Copy element by element, respecting the current view's strides.
    const std::int64_t n = numel();
    Shape idx(static_cast<std::size_t>(ndim()), 0);
    float* dst = out.data_ptr();
    for (std::int64_t linear = 0; linear < n; ++linear) {
        dst[linear] = at(idx);
        // increment the multidimensional index (row-major)
        for (std::int64_t d = ndim() - 1; d >= 0; --d) {
            auto ud = static_cast<std::size_t>(d);
            if (++idx[ud] < shape_[ud]) {
                break;
            }
            idx[ud] = 0;
        }
    }
    return out;
}

Tensor Tensor::clone() const {
    // Always an independent copy (new Storage), materializing the current view.
    Tensor out = zeros(shape_, dtype_);
    if (is_contiguous()) {
        std::memcpy(out.data_ptr(), data_ptr() + offset_,
                    static_cast<std::size_t>(numel()) * sizeof(float));
        return out;
    }
    Tensor c = contiguous();
    std::memcpy(out.data_ptr(), c.data_ptr(),
                static_cast<std::size_t>(numel()) * sizeof(float));
    return out;
}

std::string Tensor::repr() const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    os << "], dtype=" << dtype_name(dtype_) << ", device=" << device_name(device_) << ")";
    return os.str();
}

}  // namespace axon
