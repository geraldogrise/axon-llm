#include <stdexcept>

#include "axon/ops.h"

#if defined(AXON_HAVE_OPENMP)
#include <omp.h>
#endif

namespace axon::ops {

namespace {
// Above this number of elements, it is worth parallelizing with OpenMP.
constexpr std::int64_t kParallelThreshold = 50000;
}  // namespace

namespace {

// Computes the resulting shape from broadcasting `a` and `b` (NumPy rules).
Shape broadcast_shape(const Shape& a, const Shape& b) {
    const std::int64_t na = static_cast<std::int64_t>(a.size());
    const std::int64_t nb = static_cast<std::int64_t>(b.size());
    const std::int64_t n = std::max(na, nb);
    Shape out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t da = (i < n - na) ? 1 : a[static_cast<std::size_t>(i - (n - na))];
        const std::int64_t db = (i < n - nb) ? 1 : b[static_cast<std::size_t>(i - (n - nb))];
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("broadcast: incompatible shapes");
        }
        out[static_cast<std::size_t>(i)] = std::max(da, db);
    }
    return out;
}

// Contiguous (row-major) strides of a shape.
Strides contiguous_strides(const Shape& shape) {
    Strides s(shape.size());
    std::int64_t acc = 1;
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 1; i >= 0; --i) {
        s[static_cast<std::size_t>(i)] = acc;
        acc *= shape[static_cast<std::size_t>(i)];
    }
    return s;
}

// Linear offset into the contiguous buffer of `src` (right-aligned with `out_coord`),
// applying broadcasting: a dimension of size 1 uses index 0.
std::int64_t broadcast_offset(const Shape& out_coord, const Shape& src_shape,
                              const Strides& src_strides) {
    const std::int64_t n = static_cast<std::int64_t>(out_coord.size());
    const std::int64_t ns = static_cast<std::int64_t>(src_shape.size());
    std::int64_t off = 0;
    for (std::int64_t i = 0; i < ns; ++i) {
        const std::int64_t coord = out_coord[static_cast<std::size_t>(i + (n - ns))];
        const std::int64_t dim = src_shape[static_cast<std::size_t>(i)];
        const std::int64_t idx = (dim == 1) ? 0 : coord;
        off += idx * src_strides[static_cast<std::size_t>(i)];
    }
    return off;
}

Tensor binary_op(const Tensor& a, const Tensor& b, float (*op)(float, float)) {
    const Tensor ca = a.contiguous();
    const Tensor cb = b.contiguous();

    // Fast path: same shape (no broadcasting) -> vectorizable linear loop,
    // parallelized with OpenMP when large.
    if (a.shape() == b.shape()) {
        Tensor out = Tensor::zeros(a.shape());
        const float* pa = ca.data_ptr();
        const float* pb = cb.data_ptr();
        float* po = out.data_ptr();
        const std::int64_t n = out.numel();
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for schedule(static) if (n > kParallelThreshold)
#endif
        for (std::int64_t i = 0; i < n; ++i) {
            po[i] = op(pa[i], pb[i]);
        }
        return out;
    }

    const Shape out_shape = broadcast_shape(a.shape(), b.shape());
    const Strides sa = contiguous_strides(ca.shape());
    const Strides sb = contiguous_strides(cb.shape());

    Tensor out = Tensor::zeros(out_shape);
    const float* pa = ca.data_ptr();
    const float* pb = cb.data_ptr();
    float* po = out.data_ptr();

    const std::int64_t n = out.numel();
    const std::int64_t nd = out.ndim();
    Shape coord(static_cast<std::size_t>(nd), 0);
    for (std::int64_t linear = 0; linear < n; ++linear) {
        const std::int64_t ia = broadcast_offset(coord, ca.shape(), sa);
        const std::int64_t ib = broadcast_offset(coord, cb.shape(), sb);
        po[linear] = op(pa[ia], pb[ib]);
        // increment the multidimensional index (row-major)
        for (std::int64_t d = nd - 1; d >= 0; --d) {
            const auto ud = static_cast<std::size_t>(d);
            if (++coord[ud] < out_shape[ud]) {
                break;
            }
            coord[ud] = 0;
        }
    }
    return out;
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x * y; });
}

Tensor neg(const Tensor& a) {
    return mul_scalar(a, -1.0f);
}

Tensor mul_scalar(const Tensor& a, float s) {
    const Tensor ca = a.contiguous();
    Tensor out = Tensor::zeros(ca.shape());
    const float* p = ca.data_ptr();
    float* po = out.data_ptr();
    const std::int64_t n = out.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        po[i] = p[i] * s;
    }
    return out;
}

}  // namespace axon::ops
