#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "axon/autograd.h"
#include "axon/functional.h"
#include "axon/serialize.h"
#include "axon/tensor.h"
#include "axon/tokenizer.h"

using namespace axon;

namespace {
// Path in a writable directory (TEMP/TMP), avoiding the drive root.
std::string tmp_path(const char* name) {
    const char* base = std::getenv("TEMP");
    if (base == nullptr) base = std::getenv("TMP");
    if (base == nullptr) base = ".";
    return std::string(base) + "/pyaxon_test" + name;
}
}  // namespace

TEST(Serialize, SaveAndLoadPreservesValues) {
    std::vector<Tensor> src = {Tensor::from_data({2, 2}, {1, 2, 3, 4}),
                               Tensor::from_data({3}, {5, 6, 7})};
    const std::string path = tmp_path("_ckpt.bin");
    save(src, path);

    std::vector<Tensor> dst = {Tensor::zeros({2, 2}), Tensor::zeros({3})};
    load(dst, path);
    EXPECT_FLOAT_EQ(dst[0].at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(dst[1].at({2}), 7.0f);
    std::remove(path.c_str());
}

TEST(Serialize, IncompatibleShapeThrows) {
    std::vector<Tensor> src = {Tensor::zeros({2, 2})};
    const std::string path = tmp_path("_ckpt2.bin");
    save(src, path);
    std::vector<Tensor> dst = {Tensor::zeros({3, 3})};
    EXPECT_THROW(load(dst, path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Autograd, NoGradDoesNotBuildGraph) {
    Tensor x = Tensor::from_data({3}, {1, 2, 3});
    x.requires_grad_(true);
    {
        NoGradGuard g;
        Tensor y = fn::sum(fn::mul(x, x));
        EXPECT_FALSE(y.requires_grad());  // no grad_fn inside no_grad
    }
    EXPECT_TRUE(is_grad_enabled());  // restored when leaving the scope
}

TEST(Tokenizer, TrainEncodeDecodeRoundTrip) {
    const std::string text = "the cat chased the mouse across the room";
    BPETokenizer tok;
    tok.train(text, /*vocab_size=*/60);
    EXPECT_GT(tok.vocab_size(), 10);

    std::vector<int> ids = tok.encode(text);
    EXPECT_FALSE(ids.empty());
    EXPECT_EQ(tok.decode(ids), text);  // exact round-trip (single spaces)
}

TEST(Tokenizer, MergesReduceTokenCount) {
    const std::string text = "abab abab abab";  // pair "ab" very frequent
    BPETokenizer tok;
    tok.train(text, /*vocab_size=*/40);
    // With merges, "abab" becomes few tokens (much fewer than 4 chars + </w>).
    std::vector<int> ids = tok.encode("abab");
    EXPECT_LE(static_cast<int>(ids.size()), 3);
    EXPECT_EQ(tok.decode(ids), "abab");
}

TEST(Tokenizer, WordPieceEncodeDecode) {
    const std::string text = "playing player played playground playful";
    WordPieceTokenizer wp;
    wp.train(text, /*vocab_size=*/60);
    EXPECT_GT(wp.vocab_size(), 10);

    std::vector<int> ids = wp.encode("playing player");
    EXPECT_FALSE(ids.empty());
    // round-trip of words covered by the vocabulary
    EXPECT_EQ(wp.decode(ids), "playing player");

    // save/load preserves the tokenization
    const std::string path = tmp_path("_wp.txt");
    wp.save(path);
    WordPieceTokenizer wp2;
    wp2.load(path);
    EXPECT_EQ(wp2.decode(wp2.encode("played")), "played");
    std::remove(path.c_str());
}

TEST(Tokenizer, SaveAndLoad) {
    BPETokenizer a;
    a.train("banana banana banda", 40);
    const std::string path = tmp_path("_tok.txt");
    a.save(path);

    BPETokenizer b;
    b.load(path);
    EXPECT_EQ(b.vocab_size(), a.vocab_size());
    EXPECT_EQ(b.decode(b.encode("banana")), "banana");
    std::remove(path.c_str());
}
