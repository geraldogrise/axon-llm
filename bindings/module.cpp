// pyaxon Python bindings (pybind11). Exposes the C++ core "libaxon" as the
// _axon module, which the pyaxon Python package wraps in a friendly API.

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "axon/autograd.h"
#include "axon/backend.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/optim.h"
#include "axon/ops.h"
#include "axon/serialize.h"
#include "axon/tensor.h"
#include "axon/tokenizer.h"

namespace py = pybind11;
using namespace axon;

namespace {

// Discovers the shape of a nested Python list (e.g. [[0,1],[1,0]] -> [2,2]).
void infer_shape(const py::handle& o, Shape& shape) {
    if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
        auto seq = py::reinterpret_borrow<py::sequence>(o);
        shape.push_back(static_cast<std::int64_t>(seq.size()));
        if (seq.size() > 0) {
            infer_shape(seq[0], shape);
        }
    }
}

// Flattens the values of a nested list in row-major order.
void flatten(const py::handle& o, std::vector<float>& out) {
    if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
        for (const py::handle& item : py::reinterpret_borrow<py::sequence>(o)) {
            flatten(item, out);
        }
    } else {
        out.push_back(o.cast<float>());
    }
}

Tensor make_tensor(const py::object& data, bool requires_grad) {
    Shape shape;
    infer_shape(data, shape);
    std::vector<float> flat;
    flatten(data, flat);
    Tensor t = Tensor::from_data(shape, flat);
    if (requires_grad) {
        t.requires_grad_(true);
    }
    return t;
}

// Converts a Tensor into nested Python lists (for reading/printing).
// Creates a Tensor from a NumPy array (float32, C-contiguous; converts if needed).
Tensor from_numpy(const py::array& arr) {
    auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!a) {
        throw std::invalid_argument("from_numpy: invalid array");
    }
    const py::buffer_info info = a.request();
    Shape shape(info.shape.begin(), info.shape.end());
    std::vector<float> data(static_cast<std::size_t>(info.size));
    std::memcpy(data.data(), info.ptr, static_cast<std::size_t>(info.size) * sizeof(float));
    return Tensor::from_data(shape, data);
}

// Converts a Tensor into a NumPy array (copy, respecting shape and offset/strides).
py::array_t<float> to_numpy(const Tensor& t) {
    const Tensor c = t.clone();
    std::vector<py::ssize_t> shape(c.shape().begin(), c.shape().end());
    py::array_t<float> out(shape);
    std::memcpy(out.request().ptr, c.data_ptr(),
                static_cast<std::size_t>(c.numel()) * sizeof(float));
    return out;
}

py::object to_list(const Tensor& t) {
    const Tensor c = t.clone();  // clone respects offset/strides (e.g. slices)
    const float* p = c.data_ptr();
    const Shape& shape = c.shape();

    std::function<py::object(std::int64_t, std::int64_t&)> build =
        [&](std::int64_t dim, std::int64_t& pos) -> py::object {
        if (dim == t.ndim()) {
            return py::float_(p[pos++]);
        }
        py::list lst;
        for (std::int64_t i = 0; i < shape[static_cast<std::size_t>(dim)]; ++i) {
            lst.append(build(dim + 1, pos));
        }
        return std::move(lst);
    };
    std::int64_t pos = 0;
    if (t.ndim() == 0) {
        return py::float_(p[0]);
    }
    return build(0, pos);
}

}  // namespace

PYBIND11_MODULE(_axon, m) {
    m.doc() = "pyaxon - C++ core (Tensor, autograd, nn, optim)";

    // ----- Tensor -----
    py::class_<Tensor>(m, "Tensor")
        .def("shape", [](const Tensor& t) { return t.shape(); })
        .def("numel", &Tensor::numel)
        .def("ndim", &Tensor::ndim)
        .def("item", &Tensor::item)
        .def("tolist", &to_list)
        .def("numpy", &to_numpy)
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("permute", &Tensor::permute, py::arg("dims"))
        .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"), py::arg("len"))
        .def("requires_grad_", &Tensor::requires_grad_, py::arg("req") = true,
             py::return_value_policy::reference)
        .def_property_readonly("requires_grad", &Tensor::requires_grad)
        .def("grad", &Tensor::grad)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", [](Tensor& self) { backward(self); })
        .def("__add__", [](const Tensor& a, const Tensor& b) { return fn::add(a, b); })
        .def("__sub__", [](const Tensor& a, const Tensor& b) { return fn::sub(a, b); })
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return fn::mul(a, b); })
        .def("__matmul__", [](const Tensor& a, const Tensor& b) { return fn::matmul(a, b); })
        .def("__repr__", &Tensor::repr);

    // ----- Factories / module functions -----
    m.def("tensor", &make_tensor, py::arg("data"), py::arg("requires_grad") = false,
          "Creates a Tensor from a (nested) Python list.");
    m.def("from_numpy", &from_numpy, py::arg("array"),
          "Creates a Tensor from a NumPy array.");
    m.def("zeros", [](const Shape& s) { return Tensor::zeros(s); });
    m.def("ones", [](const Shape& s) { return Tensor::ones(s); });
    m.def("randn", [](const Shape& s, std::uint64_t seed) { return Tensor::randn(s, seed); },
          py::arg("shape"), py::arg("seed") = 42);
    m.def("rand",
          [](const Shape& s, std::uint64_t seed, float lo, float hi) {
              return Tensor::rand(s, seed, lo, hi);
          },
          py::arg("shape"), py::arg("seed") = 42, py::arg("lo") = 0.0f, py::arg("hi") = 1.0f);

    m.def("matmul", &fn::matmul);
    m.def("add", &fn::add);
    m.def("sub", &fn::sub);
    m.def("mul", &fn::mul);
    m.def("relu", &fn::relu);
    m.def("sigmoid", &fn::sigmoid);
    m.def("tanh", &fn::tanh);
    m.def("gelu", &fn::gelu);
    m.def("softmax", &fn::softmax);
    m.def("log_softmax", &fn::log_softmax);
    m.def("sum", &fn::sum);
    m.def("mean", &fn::mean);
    // Axis reductions (utility, no autograd).
    m.def("sum_axis", &ops::sum_axis, py::arg("a"), py::arg("axis"), py::arg("keepdim") = false);
    m.def("mean_axis", &ops::mean_axis, py::arg("a"), py::arg("axis"), py::arg("keepdim") = false);
    m.def("max_axis", &ops::max_axis, py::arg("a"), py::arg("axis"), py::arg("keepdim") = false);
    m.def("min_axis", &ops::min_axis, py::arg("a"), py::arg("axis"), py::arg("keepdim") = false);
    m.def("mse_loss", &fn::mse_loss);
    m.def("cross_entropy", &fn::cross_entropy);
    m.def("bce_loss", &fn::bce_loss, py::arg("pred"), py::arg("target"),
          "Binary cross-entropy with logits (autograd).");
    m.def("huber_loss", &fn::huber_loss, py::arg("pred"), py::arg("target"),
          py::arg("delta") = 1.0f, "Huber loss / smooth L1 (autograd).");
    m.def("conv2d", &fn::conv2d, py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("pad") = 0, "2D convolution (autograd).");
    // Fused kernels (inference).
    m.def("bias_relu", &ops::bias_relu, py::arg("x"), py::arg("bias"));
    m.def("bias_gelu", &ops::bias_gelu, py::arg("x"), py::arg("bias"));
    // int8 quantization (returns (dequantized, scale)).
    m.def("quantize_int8", [](const Tensor& t) {
        float scale = 1.0f;
        Tensor q = ops::quantize_int8(t, scale);
        return py::make_tuple(ops::dequantize_int8(q, scale), scale);
    }, py::arg("tensor"), "Quantizes/dequantizes to int8; returns (tensor_deq, scale).");
    m.def("attention", &fn::scaled_dot_product_attention, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("causal") = false);
    m.def("backward", [](Tensor& t) { backward(t); });

    // ----- Autograd / inference -----
    m.def("set_grad_enabled", &set_grad_enabled);
    m.def("is_grad_enabled", &is_grad_enabled);

    // ----- Backend -----
    m.def("cuda_available", &cuda_available,
          "True if compiled with CUDA support (-DAXON_ENABLE_CUDA=ON).");
    m.def("cuda_device_name", &cuda_device_name,
          "Name of the active CUDA device (e.g. 'Tesla T4'), or '' if CPU-only.");
    m.def("set_cuda_enabled", &set_cuda_enabled, py::arg("on"),
          "Turn the transparent GPU matmul dispatch on/off (for A/B benchmarking).");
    m.def("is_cuda_enabled", &is_cuda_enabled,
          "Whether large matmuls currently dispatch to the GPU.");
    m.def("set_cuda_matmul_threshold", &set_cuda_matmul_threshold, py::arg("work"),
          "Min work (m*k*n) before a matmul is sent to the GPU (round-trip crossover).");
    m.def("cuda_matmul_threshold", &cuda_matmul_threshold,
          "Current GPU matmul work threshold.");

    // ----- Checkpoints -----
    m.def("save", &save, py::arg("tensors"), py::arg("path"));
    m.def("load",
          [](std::vector<Tensor> tensors, const std::string& path) {
              // In-place copy: the tensors share storage with the model's.
              load(tensors, path);
          },
          py::arg("tensors"), py::arg("path"));

    // ----- nn submodule -----
    py::module_ nnm = m.def_submodule("nn", "Neural network layers");
    py::class_<nn::Module, std::shared_ptr<nn::Module>>(nnm, "Module")
        .def("forward", &nn::Module::forward)
        .def("parameters", &nn::Module::parameters)
        .def("__call__", &nn::Module::forward);
    py::enum_<nn::Init>(nnm, "Init")
        .value("He", nn::Init::He)
        .value("Xavier", nn::Init::Xavier)
        .value("Uniform", nn::Init::Uniform);
    py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nnm, "Linear")
        .def(py::init<std::int64_t, std::int64_t, std::uint64_t, nn::Init>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("seed") = 42,
             py::arg("init") = nn::Init::He);
    py::class_<nn::ReLU, nn::Module, std::shared_ptr<nn::ReLU>>(nnm, "ReLU").def(py::init<>());
    py::class_<nn::Dropout, nn::Module, std::shared_ptr<nn::Dropout>>(nnm, "Dropout")
        .def(py::init<float, std::uint64_t>(), py::arg("p") = 0.5f, py::arg("seed") = 1234);
    py::class_<nn::RNN, nn::Module, std::shared_ptr<nn::RNN>>(nnm, "RNN")
        .def(py::init<std::int64_t, std::int64_t, std::uint64_t>(), py::arg("in_features"),
             py::arg("hidden"), py::arg("seed") = 42);
    py::class_<nn::LSTM, nn::Module, std::shared_ptr<nn::LSTM>>(nnm, "LSTM")
        .def(py::init<std::int64_t, std::int64_t, std::uint64_t>(), py::arg("in_features"),
             py::arg("hidden"), py::arg("seed") = 42);
    py::class_<nn::Sigmoid, nn::Module, std::shared_ptr<nn::Sigmoid>>(nnm, "Sigmoid")
        .def(py::init<>());
    py::class_<nn::Tanh, nn::Module, std::shared_ptr<nn::Tanh>>(nnm, "Tanh").def(py::init<>());
    py::class_<nn::GELU, nn::Module, std::shared_ptr<nn::GELU>>(nnm, "GELU").def(py::init<>());
    py::class_<nn::Sequential, nn::Module, std::shared_ptr<nn::Sequential>>(nnm, "Sequential")
        .def(py::init<>())
        .def(py::init<std::vector<std::shared_ptr<nn::Module>>>(), py::arg("layers"))
        .def("add", &nn::Sequential::add);
    py::class_<nn::LayerNorm, nn::Module, std::shared_ptr<nn::LayerNorm>>(nnm, "LayerNorm")
        .def(py::init<std::int64_t, float>(), py::arg("dim"), py::arg("eps") = 1e-5f);
    py::class_<nn::Embedding, nn::Module, std::shared_ptr<nn::Embedding>>(nnm, "Embedding")
        .def(py::init<std::int64_t, std::int64_t, std::uint64_t>(), py::arg("num_embeddings"),
             py::arg("dim"), py::arg("seed") = 42);
    py::class_<nn::SelfAttention, nn::Module, std::shared_ptr<nn::SelfAttention>>(nnm,
                                                                                 "SelfAttention")
        .def(py::init<std::int64_t, bool, std::uint64_t>(), py::arg("dim"),
             py::arg("causal") = false, py::arg("seed") = 42);
    py::class_<nn::MultiHeadAttention, nn::Module, std::shared_ptr<nn::MultiHeadAttention>>(
        nnm, "MultiHeadAttention")
        .def(py::init<std::int64_t, std::int64_t, bool, std::uint64_t>(), py::arg("dim"),
             py::arg("num_heads"), py::arg("causal") = false, py::arg("seed") = 42);
    py::class_<nn::TransformerBlock, nn::Module, std::shared_ptr<nn::TransformerBlock>>(
        nnm, "TransformerBlock")
        .def(py::init<std::int64_t, std::int64_t, bool, std::uint64_t, std::int64_t>(),
             py::arg("dim"), py::arg("num_heads"), py::arg("causal") = true, py::arg("seed") = 42,
             py::arg("ff_mult") = 4);
    py::class_<nn::AxonLM, nn::Module, std::shared_ptr<nn::AxonLM>>(nnm, "AxonLM")
        .def(py::init<std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::uint64_t>(),
             py::arg("vocab_size"), py::arg("dim"), py::arg("num_heads"), py::arg("num_layers"),
             py::arg("seed") = 42)
        .def("forward_batch", &nn::AxonLM::forward_batch, py::arg("idx"), py::arg("batch"),
             py::arg("seq_len"))
        .def("reset_cache", &nn::AxonLM::reset_cache)
        .def("forward_step", &nn::AxonLM::forward_step, py::arg("token_id"), py::arg("position"))
        .def_property_readonly("vocab_size", &nn::AxonLM::vocab_size);

    // ----- Tokenizer BPE -----
    py::class_<BPETokenizer>(m, "BPETokenizer")
        .def(py::init<>())
        .def("train", &BPETokenizer::train, py::arg("text"), py::arg("vocab_size"))
        .def("encode", &BPETokenizer::encode, py::arg("text"))
        .def("decode", &BPETokenizer::decode, py::arg("ids"))
        .def("save", &BPETokenizer::save, py::arg("path"))
        .def("load", &BPETokenizer::load, py::arg("path"))
        .def_property_readonly("vocab_size", &BPETokenizer::vocab_size)
        .def_property_readonly("unk_id", &BPETokenizer::unk_id)
        .def_property_readonly("bos_id", &BPETokenizer::bos_id)
        .def_property_readonly("eos_id", &BPETokenizer::eos_id)
        .def_property_readonly("pad_id", &BPETokenizer::pad_id);

    py::class_<WordPieceTokenizer>(m, "WordPieceTokenizer")
        .def(py::init<>())
        .def("train", &WordPieceTokenizer::train, py::arg("text"), py::arg("vocab_size"))
        .def("encode", &WordPieceTokenizer::encode, py::arg("text"))
        .def("decode", &WordPieceTokenizer::decode, py::arg("ids"))
        .def("save", &WordPieceTokenizer::save, py::arg("path"))
        .def("load", &WordPieceTokenizer::load, py::arg("path"))
        .def_property_readonly("vocab_size", &WordPieceTokenizer::vocab_size)
        .def_property_readonly("unk_id", &WordPieceTokenizer::unk_id);

    // ----- optim submodule -----
    py::module_ optm = m.def_submodule("optim", "Optimizers");
    py::class_<optim::Optimizer, std::shared_ptr<optim::Optimizer>>(optm, "Optimizer")
        .def("step", &optim::Optimizer::step)
        .def("zero_grad", &optim::Optimizer::zero_grad)
        .def("set_lr", &optim::Optimizer::set_lr, py::arg("lr"))
        .def_property_readonly("lr", &optim::Optimizer::lr)
        .def("state_dict", &optim::Optimizer::state_dict)
        .def("load_state_dict", &optim::Optimizer::load_state_dict, py::arg("state"));
    py::class_<optim::SGD, optim::Optimizer, std::shared_ptr<optim::SGD>>(optm, "SGD")
        .def(py::init<std::vector<Tensor>, float, float, float>(), py::arg("params"),
             py::arg("lr"), py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f);
    py::class_<optim::Adam, optim::Optimizer, std::shared_ptr<optim::Adam>>(optm, "Adam")
        .def(py::init<std::vector<Tensor>, float, float, float, float, float>(), py::arg("params"),
             py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f,
             py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f);
    py::class_<optim::RMSProp, optim::Optimizer, std::shared_ptr<optim::RMSProp>>(optm, "RMSProp")
        .def(py::init<std::vector<Tensor>, float, float, float, float>(), py::arg("params"),
             py::arg("lr") = 1e-3f, py::arg("alpha") = 0.99f, py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.0f);
}
