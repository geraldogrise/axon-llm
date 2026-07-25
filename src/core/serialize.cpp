#include "axon/serialize.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace axon {

namespace {
constexpr char kMagic[4] = {'A', 'X', 'O', 'N'};
constexpr std::int32_t kVersion = 1;

template <typename T>
void write_pod(std::ofstream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T read_pod(std::ifstream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}
}  // namespace

void save(const std::vector<Tensor>& tensors, const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("save: could not open " + path);
    }
    os.write(kMagic, 4);
    write_pod(os, kVersion);
    write_pod(os, static_cast<std::int64_t>(tensors.size()));

    for (const Tensor& t : tensors) {
        const Tensor c = t.contiguous();
        write_pod(os, c.ndim());
        for (std::int64_t d : c.shape()) {
            write_pod(os, d);
        }
        const std::int64_t n = c.numel();
        os.write(reinterpret_cast<const char*>(c.data_ptr()),
                 static_cast<std::streamsize>(n) * static_cast<std::streamsize>(sizeof(float)));
    }
}

void load(std::vector<Tensor>& tensors, const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("load: could not open " + path);
    }
    char magic[4];
    is.read(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0) {
        throw std::runtime_error("load: invalid file (incorrect magic)");
    }
    const std::int32_t version = read_pod<std::int32_t>(is);
    if (version != kVersion) {
        throw std::runtime_error("load: incompatible checkpoint version");
    }
    const std::int64_t count = read_pod<std::int64_t>(is);
    if (count != static_cast<std::int64_t>(tensors.size())) {
        throw std::runtime_error("load: number of tensors does not match the model");
    }

    for (Tensor& t : tensors) {
        const std::int64_t ndim = read_pod<std::int64_t>(is);
        Shape shape(static_cast<std::size_t>(ndim));
        std::int64_t numel = 1;
        for (std::int64_t i = 0; i < ndim; ++i) {
            shape[static_cast<std::size_t>(i)] = read_pod<std::int64_t>(is);
            numel *= shape[static_cast<std::size_t>(i)];
        }
        if (shape != t.shape()) {
            throw std::runtime_error("load: shape of a tensor does not match the model");
        }
        // Copy into the existing tensor (parameters are contiguous).
        is.read(reinterpret_cast<char*>(t.data_ptr()),
                static_cast<std::streamsize>(numel) *
                    static_cast<std::streamsize>(sizeof(float)));
    }
}

}  // namespace axon
