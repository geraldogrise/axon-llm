#include <cmath>
#include <cstring>
#include <stdexcept>

#include "axon/ops.h"

namespace axon::ops {

namespace {
void require_2d(const Tensor& t, const char* who) {
    if (t.ndim() != 2) {
        throw std::invalid_argument(std::string(who) + ": espera tensor 2D");
    }
}
}  // namespace

Tensor slice_cols(const Tensor& a, std::int64_t start, std::int64_t len) {
    require_2d(a, "slice_cols");
    const Tensor ca = a.contiguous();
    const std::int64_t rows = ca.shape()[0];
    const std::int64_t C = ca.shape()[1];
    if (start < 0 || start + len > C) {
        throw std::out_of_range("slice_cols: intervalo invalido");
    }
    Tensor out = Tensor::zeros({rows, len});
    const float* pa = ca.data_ptr();
    float* po = out.data_ptr();
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < len; ++j) {
            po[i * len + j] = pa[i * C + (start + j)];
        }
    }
    return out;
}

Tensor slice_cols_backward(const Tensor& g, std::int64_t start, std::int64_t full_cols) {
    require_2d(g, "slice_cols_backward");
    const Tensor cg = g.contiguous();
    const std::int64_t rows = cg.shape()[0];
    const std::int64_t len = cg.shape()[1];
    Tensor out = Tensor::zeros({rows, full_cols});
    const float* pg = cg.data_ptr();
    float* po = out.data_ptr();
    for (std::int64_t i = 0; i < rows; ++i) {
        for (std::int64_t j = 0; j < len; ++j) {
            po[i * full_cols + (start + j)] = pg[i * len + j];
        }
    }
    return out;
}

Tensor concat_cols(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        throw std::invalid_argument("concat_cols: lista vazia");
    }
    const std::int64_t rows = parts[0].shape()[0];
    std::int64_t total = 0;
    for (const Tensor& p : parts) {
        require_2d(p, "concat_cols");
        if (p.shape()[0] != rows) {
            throw std::invalid_argument("concat_cols: numero de linhas inconsistente");
        }
        total += p.shape()[1];
    }
    Tensor out = Tensor::zeros({rows, total});
    float* po = out.data_ptr();
    std::int64_t col = 0;
    for (const Tensor& p : parts) {
        const Tensor cp = p.contiguous();
        const std::int64_t c = cp.shape()[1];
        const float* pp = cp.data_ptr();
        for (std::int64_t i = 0; i < rows; ++i) {
            for (std::int64_t j = 0; j < c; ++j) {
                po[i * total + (col + j)] = pp[i * c + j];
            }
        }
        col += c;
    }
    return out;
}

Tensor slice_rows(const Tensor& a, std::int64_t start, std::int64_t len) {
    require_2d(a, "slice_rows");
    const Tensor ca = a.contiguous();
    const std::int64_t rows = ca.shape()[0];
    const std::int64_t C = ca.shape()[1];
    if (start < 0 || start + len > rows) {
        throw std::out_of_range("slice_rows: intervalo invalido");
    }
    Tensor out = Tensor::zeros({len, C});
    std::memcpy(out.data_ptr(), ca.data_ptr() + start * C,
                static_cast<std::size_t>(len * C) * sizeof(float));
    return out;
}

Tensor slice_rows_backward(const Tensor& g, std::int64_t start, std::int64_t full_rows) {
    require_2d(g, "slice_rows_backward");
    const Tensor cg = g.contiguous();
    const std::int64_t len = cg.shape()[0];
    const std::int64_t C = cg.shape()[1];
    Tensor out = Tensor::zeros({full_rows, C});
    std::memcpy(out.data_ptr() + start * C, cg.data_ptr(),
                static_cast<std::size_t>(len * C) * sizeof(float));
    return out;
}

Tensor concat_rows(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        throw std::invalid_argument("concat_rows: lista vazia");
    }
    const std::int64_t C = parts[0].shape()[1];
    std::int64_t total = 0;
    for (const Tensor& p : parts) {
        require_2d(p, "concat_rows");
        if (p.shape()[1] != C) {
            throw std::invalid_argument("concat_rows: numero de colunas inconsistente");
        }
        total += p.shape()[0];
    }
    Tensor out = Tensor::zeros({total, C});
    float* po = out.data_ptr();
    std::int64_t row = 0;
    for (const Tensor& p : parts) {
        const Tensor cp = p.contiguous();
        const std::int64_t r = cp.shape()[0];
        std::memcpy(po + row * C, cp.data_ptr(),
                    static_cast<std::size_t>(r * C) * sizeof(float));
        row += r;
    }
    return out;
}

Tensor positional_encoding(std::int64_t L, std::int64_t D) {
    Tensor pe = Tensor::zeros({L, D});
    float* p = pe.data_ptr();
    for (std::int64_t pos = 0; pos < L; ++pos) {
        for (std::int64_t i = 0; i < D; ++i) {
            const auto pair = static_cast<float>(2 * (i / 2));
            const float denom = std::pow(10000.0f, pair / static_cast<float>(D));
            const float angle = static_cast<float>(pos) / denom;
            p[pos * D + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return pe;
}

}  // namespace axon::ops
