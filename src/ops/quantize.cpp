#include <algorithm>
#include <cmath>

#include "axon/ops.h"

namespace axon::ops {

Tensor quantize_int8(const Tensor& t, float& scale_out) {
    const Tensor ct = t.contiguous();
    const float* p = ct.data_ptr();
    const std::int64_t n = ct.numel();
    float amax = 0.0f;
    for (std::int64_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(p[i]));
    const float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    scale_out = scale;

    Tensor q = Tensor::zeros(ct.shape());
    float* pq = q.data_ptr();
    for (std::int64_t i = 0; i < n; ++i) {
        float v = std::round(p[i] / scale);
        v = std::max(-127.0f, std::min(127.0f, v));  // clamp ao intervalo int8
        pq[i] = v;
    }
    return q;
}

Tensor dequantize_int8(const Tensor& q, float scale) {
    return mul_scalar(q, scale);
}

}  // namespace axon::ops
