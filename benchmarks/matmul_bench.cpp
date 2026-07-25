// Benchmark of the matmul variants. Measures GFLOP/s and the speedup relative to
// the naive version. "No optimization by guesswork": here is the number.

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "axon/ops.h"
#include "axon/tensor.h"

using axon::Tensor;
using Clock = std::chrono::high_resolution_clock;

namespace {

// Measures the average time (ms) of `fn` over `reps` runs (with 1 warmup).
double time_ms(const std::function<void()>& fn, int reps) {
    fn();  // warmup (cache, threads)
    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r) {
        fn();
    }
    const auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

void bench_size(std::int64_t m, std::int64_t k, std::int64_t n, int reps) {
    Tensor a = Tensor::randn({m, k}, 1);
    Tensor b = Tensor::randn({k, n}, 2);
    const double gflop = 2.0 * static_cast<double>(m) * k * n / 1e9;

    struct Variant {
        std::string name;
        std::function<Tensor()> run;
    };
    std::vector<Variant> variants = {
        {"naive  (ijk)", [&] { return axon::ops::detail::matmul_naive(a, b); }},
        {"ikj    (cache)", [&] { return axon::ops::detail::matmul_ikj(a, b); }},
        {"blocked(tiling)", [&] { return axon::ops::detail::matmul_blocked(a, b); }},
        {"omp    (threads)", [&] { return axon::ops::detail::matmul_omp(a, b); }},
        {"avx2   (simd+omp)", [&] { return axon::ops::detail::matmul_avx2(a, b); }},
    };

    std::printf("\n=== matmul  (%lld x %lld) x (%lld x %lld)  =  %.2f GFLOP ===\n",
                (long long)m, (long long)k, (long long)k, (long long)n, gflop);
    std::printf("%-18s %10s %12s %10s\n", "variant", "ms", "GFLOP/s", "speedup");

    double base_ms = 0.0;
    for (auto& v : variants) {
        const double ms = time_ms([&] { volatile auto t = v.run(); (void)t; }, reps);
        if (v.name.rfind("naive", 0) == 0) {
            base_ms = ms;
        }
        const double gflops = gflop / (ms / 1e3);
        const double speedup = base_ms > 0 ? base_ms / ms : 1.0;
        std::printf("%-18s %10.3f %12.2f %9.1fx\n", v.name.c_str(), ms, gflops, speedup);
    }
}

}  // namespace

int main() {
    std::printf("pyaxon - matmul benchmark (speedup vs. naive)\n");
    bench_size(128, 128, 128, 50);
    bench_size(256, 256, 256, 20);
    bench_size(512, 512, 512, 5);
    return 0;
}
