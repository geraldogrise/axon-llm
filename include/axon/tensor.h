#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "axon/device.h"
#include "axon/dtype.h"
#include "axon/storage.h"

namespace axon {

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

// Forward declarations of the autograd subsystem (defined in axon/autograd.h).
struct AutogradMeta;
struct Node;

// N-dimensional Tensor. Stores the data in a contiguous (shareable) Storage and
// describes the "window" over that buffer with shape/strides/offset. This allows
// views (reshape/transpose/slice) WITHOUT copying data - the same technique as NumPy.
//
// Autograd: each Tensor can carry an AutogradMeta (grad + grad_fn) behind
// a shared_ptr. Copies of a Tensor SHARE this meta, so the accumulated
// gradient is seen by all - essential for the backward to work.
class Tensor {
  public:
    Tensor() = default;

    // ----- Factories -----
    static Tensor zeros(const Shape& shape, DType dtype = DType::F32);
    static Tensor ones(const Shape& shape, DType dtype = DType::F32);
    static Tensor full(const Shape& shape, float value, DType dtype = DType::F32);
    static Tensor from_data(const Shape& shape, const std::vector<float>& data);
    // Sample from a normal(mean, stddev). Fixed seed -> reproducible.
    static Tensor randn(const Shape& shape, std::uint64_t seed = 42, float mean = 0.0f,
                        float stddev = 1.0f);
    // Uniform sample in [lo, hi). Fixed seed -> reproducible.
    static Tensor rand(const Shape& shape, std::uint64_t seed = 42, float lo = 0.0f,
                       float hi = 1.0f);

    // ----- Metadata -----
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
    [[nodiscard]] std::int64_t offset() const noexcept { return offset_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] Device device() const noexcept { return device_; }

    [[nodiscard]] std::int64_t ndim() const noexcept {
        return static_cast<std::int64_t>(shape_.size());
    }
    [[nodiscard]] std::int64_t numel() const noexcept;
    [[nodiscard]] bool is_contiguous() const noexcept;

    // ----- Access (F32 only in this phase) -----
    [[nodiscard]] float& at(const Shape& idx);
    [[nodiscard]] float at(const Shape& idx) const;
    [[nodiscard]] float item() const;  // value of a 1-element tensor
    [[nodiscard]] float* data_ptr() noexcept;
    [[nodiscard]] const float* data_ptr() const noexcept;

    // ----- Views (no copy) and copies -----
    [[nodiscard]] Tensor reshape(const Shape& new_shape) const;
    [[nodiscard]] Tensor transpose(std::int64_t dim0, std::int64_t dim1) const;
    // Reorders the axes (view without copy). `dims` is a permutation of 0..ndim-1.
    [[nodiscard]] Tensor permute(const Shape& dims) const;
    // Slices [start, start+len) along `dim` (view without copy).
    [[nodiscard]] Tensor slice(std::int64_t dim, std::int64_t start, std::int64_t len) const;
    [[nodiscard]] Tensor contiguous() const;
    [[nodiscard]] Tensor clone() const;  // ALWAYS an independent copy (new buffer)

    // ----- Autograd -----
    // Marks this tensor as a trainable leaf (parameter). Returns *this for chaining.
    Tensor& requires_grad_(bool req = true);
    [[nodiscard]] bool requires_grad() const noexcept;  // trainable leaf OR has grad_fn
    [[nodiscard]] bool has_grad() const noexcept;
    [[nodiscard]] const Tensor& grad() const;  // accumulated gradient (throws if absent)
    void zero_grad();

    // Internal use by the autograd engine:
    void set_grad_fn(const std::shared_ptr<Node>& fn);
    [[nodiscard]] std::shared_ptr<Node> grad_fn() const;
    void accumulate_grad(const Tensor& g);           // grad += g
    [[nodiscard]] AutogradMeta* autograd_meta() const noexcept;

    // ----- Utility -----
    [[nodiscard]] std::string repr() const;

  private:
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Strides strides_;
    std::int64_t offset_ = 0;
    DType dtype_ = DType::F32;
    Device device_ = Device::CPU;
    std::shared_ptr<AutogradMeta> autograd_;  // null until it participates in autograd

    static Strides contiguous_strides(const Shape& shape);
    [[nodiscard]] std::int64_t flat_index(const Shape& idx) const;
    AutogradMeta& ensure_meta();  // creates the meta on demand
};

}  // namespace axon
