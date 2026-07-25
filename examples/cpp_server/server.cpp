// Minimal HTTP server in C++ that trains an AxonLM and serves it (plan.md asks
// for an "HTTP server in C++"). Uses raw sockets (Winsock on Windows, BSD sockets
// on POSIX). OPTIONAL target: compile with -DAXON_BUILD_CPP_SERVER=ON.
//
//   GET /generate?prompt=hi    -> generates text (greedy) and returns plain text
//   GET /                      -> simple instructions

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/nn.h"
#include "axon/optim.h"
#include "axon/tokenizer.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#define INVALID_SOCKET (-1)
#define closesocket close
#endif

using namespace axon;

namespace {
constexpr int kPort = 8080;
constexpr int kContext = 24;

const char* kText =
    "the little cockroach says she has seven silk skirts. it is a lie she has only one. "
    "ha ha ha ho ho ho she has only one. the little cockroach says she has a buckled shoe. "
    "it is a lie what she has is a bare foot. ha ha ha ho ho ho what she has is a bare foot. ";

BPETokenizer g_tok;
std::shared_ptr<nn::AxonLM> g_model;

Tensor ids_to_tensor(const std::vector<int>& ids) {
    std::vector<float> f(ids.begin(), ids.end());
    return Tensor::from_data({static_cast<std::int64_t>(ids.size())}, f);
}

// Greedy generation (argmax) — no extra dependencies, ideal for the C++ demo.
std::string generate(const std::string& prompt, int n_tokens) {
    std::vector<int> ids = g_tok.encode(prompt);
    if (ids.empty()) ids.push_back(0);
    NoGradGuard ng;
    for (int step = 0; step < n_tokens; ++step) {
        std::vector<int> ctx(ids.end() - std::min<std::size_t>(ids.size(), kContext), ids.end());
        Tensor logits = g_model->forward(ids_to_tensor(ctx));
        const std::int64_t L = logits.shape()[0];
        const std::int64_t V = logits.shape()[1];
        const float* row = logits.data_ptr() + (L - 1) * V;
        int best = 0;
        for (std::int64_t j = 1; j < V; ++j) {
            if (row[j] > row[best]) best = static_cast<int>(j);
        }
        if (best == g_tok.eos_id()) break;
        ids.push_back(best);
    }
    return g_tok.decode(ids);
}

void train() {
    g_tok.train(kText, 80);
    std::vector<int> ids = g_tok.encode(kText);
    g_model = std::make_shared<nn::AxonLM>(g_tok.vocab_size(), 64, 4, 3, 1);
    optim::Adam opt(g_model->parameters(), 0.002f);

    std::printf("[train] vocab=%d tokens=%zu\n", g_tok.vocab_size(), ids.size());
    for (int epoch = 1; epoch <= 8; ++epoch) {
        // simple batch: sliding windows grouped via forward_batch.
        const int L = kContext;
        std::vector<float> xb, yb;
        int b = 0;
        float total = 0.0f;
        int steps = 0;
        auto flush = [&]() {
            if (b == 0) return;
            Tensor idx = Tensor::from_data({static_cast<std::int64_t>(xb.size())}, xb);
            Tensor tgt = Tensor::from_data({static_cast<std::int64_t>(yb.size())}, yb);
            Tensor logits = g_model->forward_batch(idx, b, L);
            Tensor loss = fn::cross_entropy(logits, tgt);
            opt.zero_grad();
            backward(loss);
            opt.step();
            total += loss.item();
            ++steps;
            xb.clear();
            yb.clear();
            b = 0;
        };
        for (std::size_t i = 0; i + L < ids.size(); ++i) {
            for (int j = 0; j < L; ++j) {
                xb.push_back(static_cast<float>(ids[i + j]));
                yb.push_back(static_cast<float>(ids[i + j + 1]));
            }
            if (++b == 8) flush();
        }
        flush();
        std::printf("[train] epoch %d/8 loss=%.4f\n", epoch, total / std::max(steps, 1));
    }
}

std::string url_decode(const std::string& s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') {
            out.push_back(' ');
        } else if (s[i] == '%' && i + 2 < s.size()) {
            out.push_back(static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16)));
            i += 2;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

void handle(socket_t client) {
    char buf[4096] = {0};
    const int n = recv(client, buf, sizeof(buf) - 1, 0);
    std::string body, status = "200 OK";
    if (n > 0) {
        std::string req(buf);
        const std::size_t sp = req.find(' ');
        const std::size_t sp2 = req.find(' ', sp + 1);
        const std::string path = req.substr(sp + 1, sp2 - sp - 1);
        if (path.rfind("/generate", 0) == 0) {
            std::string prompt = "the cockroach";
            const std::size_t q = path.find("prompt=");
            if (q != std::string::npos) {
                std::string p = path.substr(q + 7);
                const std::size_t amp = p.find('&');
                if (amp != std::string::npos) p = p.substr(0, amp);
                prompt = url_decode(p);
            }
            body = generate(prompt, 40);
        } else {
            body = "AxonLM (C++). Use GET /generate?prompt=your+text";
        }
    }
    std::string resp = "HTTP/1.1 " + status + "\r\nContent-Type: text/plain; charset=utf-8\r\n";
    resp += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    resp += body;
    send(client, resp.c_str(), static_cast<int>(resp.size()), 0);
    closesocket(client);
}

}  // namespace

int main() {
#if defined(_WIN32)
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    train();

    socket_t srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(kPort);
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::printf("error binding to port %d\n", kPort);
        return 1;
    }
    listen(srv, 8);
    std::printf("[serve] AxonLM in C++ listening at http://localhost:%d\n", kPort);
    std::fflush(stdout);
    for (;;) {
        socket_t client = accept(srv, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        handle(client);
    }
#if defined(_WIN32)
    WSACleanup();
#endif
    return 0;
}
