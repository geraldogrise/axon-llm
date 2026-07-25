// 2D convolution op (forward + backward) for the autograd core.
//
// Input x: (N, Cin, H, W), weight: (Cout, Cin, K, K), bias: (Cout).
// Output: (N, Cout, Hout, Wout) with Hout=(H+2p-K)/s+1, Wout likewise.
// Direct (loop) implementation -- clear and correct; the fast path (im2col+GEMM or
// cuDNN) is a later optimization.

#include "axon/ops.h"

#include <stdexcept>

namespace axon::ops {

namespace {
struct Dims {
    std::int64_t N, Cin, H, W, Cout, K, Hout, Wout;
};
Dims conv_dims(const Tensor& x, const Tensor& w, int stride, int pad) {
    const auto& xs = x.shape();
    const auto& ws = w.shape();
    if (xs.size() != 4 || ws.size() != 4) {
        throw std::invalid_argument("conv2d: x must be (N,Cin,H,W) and weight (Cout,Cin,K,K)");
    }
    Dims d{};
    d.N = xs[0]; d.Cin = xs[1]; d.H = xs[2]; d.W = xs[3];
    d.Cout = ws[0]; d.K = ws[2];
    if (ws[1] != d.Cin || ws[2] != ws[3]) {
        throw std::invalid_argument("conv2d: weight channels/kernel mismatch");
    }
    d.Hout = (d.H + 2 * pad - d.K) / stride + 1;
    d.Wout = (d.W + 2 * pad - d.K) / stride + 1;
    return d;
}
}  // namespace

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int stride, int pad) {
    const Tensor cx = x.contiguous(), cw = weight.contiguous(), cb = bias.contiguous();
    const Dims d = conv_dims(cx, cw, stride, pad);
    const float* px = cx.data_ptr();
    const float* pw = cw.data_ptr();
    const float* pb = cb.data_ptr();
    Tensor out = Tensor::zeros({d.N, d.Cout, d.Hout, d.Wout});
    float* po = out.data_ptr();

    for (std::int64_t n = 0; n < d.N; ++n)
        for (std::int64_t co = 0; co < d.Cout; ++co)
            for (std::int64_t oi = 0; oi < d.Hout; ++oi)
                for (std::int64_t oj = 0; oj < d.Wout; ++oj) {
                    float acc = pb[co];
                    for (std::int64_t ci = 0; ci < d.Cin; ++ci)
                        for (std::int64_t ki = 0; ki < d.K; ++ki)
                            for (std::int64_t kj = 0; kj < d.K; ++kj) {
                                const std::int64_t ii = oi * stride + ki - pad;
                                const std::int64_t jj = oj * stride + kj - pad;
                                if (ii < 0 || ii >= d.H || jj < 0 || jj >= d.W) continue;
                                const float xv =
                                    px[((n * d.Cin + ci) * d.H + ii) * d.W + jj];
                                const float wv =
                                    pw[((co * d.Cin + ci) * d.K + ki) * d.K + kj];
                                acc += xv * wv;
                            }
                    po[((n * d.Cout + co) * d.Hout + oi) * d.Wout + oj] = acc;
                }
    return out;
}

// Returns {dx, dweight, dbias}. g is the gradient of the output (N,Cout,Hout,Wout).
std::vector<Tensor> conv2d_backward(const Tensor& x, const Tensor& weight, const Tensor& g,
                                    int stride, int pad) {
    const Tensor cx = x.contiguous(), cw = weight.contiguous(), cg = g.contiguous();
    const Dims d = conv_dims(cx, cw, stride, pad);
    const float* px = cx.data_ptr();
    const float* pw = cw.data_ptr();
    const float* pg = cg.data_ptr();

    Tensor dx = Tensor::zeros(cx.shape());
    Tensor dw = Tensor::zeros(cw.shape());
    Tensor db = Tensor::zeros({d.Cout});
    float* pdx = dx.data_ptr();
    float* pdw = dw.data_ptr();
    float* pdb = db.data_ptr();

    for (std::int64_t n = 0; n < d.N; ++n)
        for (std::int64_t co = 0; co < d.Cout; ++co)
            for (std::int64_t oi = 0; oi < d.Hout; ++oi)
                for (std::int64_t oj = 0; oj < d.Wout; ++oj) {
                    const float go = pg[((n * d.Cout + co) * d.Hout + oi) * d.Wout + oj];
                    pdb[co] += go;
                    for (std::int64_t ci = 0; ci < d.Cin; ++ci)
                        for (std::int64_t ki = 0; ki < d.K; ++ki)
                            for (std::int64_t kj = 0; kj < d.K; ++kj) {
                                const std::int64_t ii = oi * stride + ki - pad;
                                const std::int64_t jj = oj * stride + kj - pad;
                                if (ii < 0 || ii >= d.H || jj < 0 || jj >= d.W) continue;
                                const std::int64_t xi =
                                    ((n * d.Cin + ci) * d.H + ii) * d.W + jj;
                                const std::int64_t wi =
                                    ((co * d.Cin + ci) * d.K + ki) * d.K + kj;
                                pdw[wi] += go * px[xi];
                                pdx[xi] += go * pw[wi];
                            }
                }
    return {dx, dw, db};
}

}  // namespace axon::ops
