#include <limits>
#include <stdexcept>

#include "axon/ops.h"

namespace axon::ops {

namespace {

Strides contiguous_strides(const Shape& shape) {
    Strides s(shape.size());
    std::int64_t acc = 1;
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 1; i >= 0; --i) {
        s[static_cast<std::size_t>(i)] = acc;
        acc *= shape[static_cast<std::size_t>(i)];
    }
    return s;
}

// Sums over a single axis, keeping the dimension (keepdim) with size 1.
Tensor reduce_sum_axis_keepdim(const Tensor& t, std::int64_t axis) {
    const Tensor ct = t.contiguous();
    Shape out_shape = ct.shape();
    out_shape[static_cast<std::size_t>(axis)] = 1;
    Tensor out = Tensor::zeros(out_shape);

    const Strides in_str = contiguous_strides(ct.shape());
    const Strides out_str = contiguous_strides(out_shape);
    const float* pi = ct.data_ptr();
    float* po = out.data_ptr();

    const std::int64_t n = ct.numel();
    const std::int64_t nd = ct.ndim();
    Shape coord(static_cast<std::size_t>(nd), 0);
    for (std::int64_t linear = 0; linear < n; ++linear) {
        std::int64_t ooff = 0;
        for (std::int64_t d = 0; d < nd; ++d) {
            const auto ud = static_cast<std::size_t>(d);
            const std::int64_t c = (d == axis) ? 0 : coord[ud];
            ooff += c * out_str[ud];
        }
        po[ooff] += pi[linear];
        for (std::int64_t d = nd - 1; d >= 0; --d) {
            const auto ud = static_cast<std::size_t>(d);
            if (++coord[ud] < ct.shape()[ud]) {
                break;
            }
            coord[ud] = 0;
        }
    }
    (void)in_str;
    return out;
}

}  // namespace

Tensor sum_all(const Tensor& a) {
    const Tensor ca = a.contiguous();
    const float* p = ca.data_ptr();
    const std::int64_t n = ca.numel();
    float acc = 0.0f;
#if defined(AXON_HAVE_OPENMP)
#pragma omp parallel for reduction(+ : acc) schedule(static) if (n > 50000)
#endif
    for (std::int64_t i = 0; i < n; ++i) {
        acc += p[i];
    }
    return Tensor::full({1}, acc);
}

Tensor mean_all(const Tensor& a) {
    const std::int64_t n = a.numel();
    Tensor s = sum_all(a);
    return mul_scalar(s, 1.0f / static_cast<float>(n));
}

Tensor ones_like(const Tensor& t) {
    return Tensor::full(t.shape(), 1.0f);
}

namespace {
// Reduces along an axis by applying a combiner (op) over an accumulator.
// init: initial value; final_scale: factor applied at the end (1 for sum/max/min, 1/dim for mean).
template <typename Op>
Tensor reduce_axis(const Tensor& a, std::int64_t axis, bool keepdim, float init, Op op,
                   bool average) {
    const Tensor ca = a.contiguous();
    const std::int64_t nd = ca.ndim();
    if (axis < 0) axis += nd;
    if (axis < 0 || axis >= nd) {
        throw std::invalid_argument("reduce_axis: invalid axis");
    }
    const auto uax = static_cast<std::size_t>(axis);
    const std::int64_t dim = ca.shape()[uax];

    Shape keep_shape = ca.shape();
    keep_shape[uax] = 1;
    Tensor out = Tensor::full(keep_shape, init);
    const Strides in_str = contiguous_strides(ca.shape());
    const Strides out_str = contiguous_strides(keep_shape);
    const float* pi = ca.data_ptr();
    float* po = out.data_ptr();

    const std::int64_t n = ca.numel();
    Shape coord(static_cast<std::size_t>(nd), 0);
    for (std::int64_t linear = 0; linear < n; ++linear) {
        std::int64_t ooff = 0;
        for (std::int64_t d = 0; d < nd; ++d) {
            const auto ud = static_cast<std::size_t>(d);
            const std::int64_t c = (d == axis) ? 0 : coord[ud];
            ooff += c * out_str[ud];
        }
        po[ooff] = op(po[ooff], pi[linear]);
        for (std::int64_t d = nd - 1; d >= 0; --d) {
            const auto ud = static_cast<std::size_t>(d);
            if (++coord[ud] < ca.shape()[ud]) break;
            coord[ud] = 0;
        }
    }
    (void)in_str;
    if (average) {
        for (std::int64_t i = 0; i < out.numel(); ++i) po[i] /= static_cast<float>(dim);
    }
    if (!keepdim) {
        Shape ns;
        for (std::int64_t d = 0; d < nd; ++d) {
            if (d != axis) ns.push_back(ca.shape()[static_cast<std::size_t>(d)]);
        }
        if (ns.empty()) ns.push_back(1);
        out = out.reshape(ns);
    }
    return out;
}
}  // namespace

Tensor sum_axis(const Tensor& a, std::int64_t axis, bool keepdim) {
    return reduce_axis(a, axis, keepdim, 0.0f, [](float acc, float x) { return acc + x; }, false);
}

Tensor mean_axis(const Tensor& a, std::int64_t axis, bool keepdim) {
    return reduce_axis(a, axis, keepdim, 0.0f, [](float acc, float x) { return acc + x; }, true);
}

Tensor max_axis(const Tensor& a, std::int64_t axis, bool keepdim) {
    return reduce_axis(a, axis, keepdim, -std::numeric_limits<float>::infinity(),
                       [](float acc, float x) { return x > acc ? x : acc; }, false);
}

Tensor min_axis(const Tensor& a, std::int64_t axis, bool keepdim) {
    return reduce_axis(a, axis, keepdim, std::numeric_limits<float>::infinity(),
                       [](float acc, float x) { return x < acc ? x : acc; }, false);
}

Tensor sum_to(const Tensor& grad, const Shape& target_shape) {
    Tensor g = grad.contiguous();
    // 1) If grad has more dimensions than the target, sum (removing) the leading axes.
    while (g.ndim() > static_cast<std::int64_t>(target_shape.size())) {
        g = reduce_sum_axis_keepdim(g, 0);
        // remove axis 0 (now size 1) via reshape to the shape without it
        Shape ns(g.shape().begin() + 1, g.shape().end());
        g = g.reshape(ns);
    }
    // 2) Where the target has dim 1 but grad has dim > 1, sum along that axis (keepdim).
    for (std::int64_t d = 0; d < g.ndim(); ++d) {
        const auto ud = static_cast<std::size_t>(d);
        if (target_shape[ud] == 1 && g.shape()[ud] != 1) {
            g = reduce_sum_axis_keepdim(g, d);
        }
    }
    if (g.shape() != target_shape) {
        g = g.reshape(target_shape);
    }
    return g;
}

}  // namespace axon::ops
