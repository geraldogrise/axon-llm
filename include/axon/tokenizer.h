#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace axon {

// BPE (Byte-Pair Encoding) tokenizer - the same kind used by GPTs.
// Starts with character symbols and iteratively merges the most frequent
// adjacent pair into a new symbol, until reaching the target vocabulary size.
//
// Works per word (separated by whitespace), with an end-of-word marker
// "</w>", so that merges don't cross word boundaries.
class BPETokenizer {
  public:
    // Learns the merges from `text`, up to `vocab_size` tokens (approx.).
    void train(const std::string& text, int vocab_size);

    // Text -> list of token ids.
    [[nodiscard]] std::vector<int> encode(const std::string& text) const;
    // List of ids -> text.
    [[nodiscard]] std::string decode(const std::vector<int>& ids) const;

    [[nodiscard]] int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

    // Ids of the special tokens (reserved at the start of the vocabulary).
    [[nodiscard]] int unk_id() const { return unk_id_; }
    [[nodiscard]] int bos_id() const { return bos_id_; }  // beginning of sequence
    [[nodiscard]] int eos_id() const { return eos_id_; }  // end of sequence
    [[nodiscard]] int pad_id() const { return pad_id_; }  // padding

    void save(const std::string& path) const;
    void load(const std::string& path);

  private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    // Merges in the order they were learned (application priority).
    std::vector<std::pair<std::string, std::string>> merges_;
    int unk_id_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int pad_id_ = 3;

    int add_token(const std::string& tok);
    // Applies the learned merges to a word (list of symbols).
    void apply_merges(std::vector<std::string>& symbols) const;
};

// WordPiece tokenizer (used by BERT). Learns a subword vocabulary and tokenizes
// each word by greedy longest-prefix matching; continuation pieces carry the
// "##" prefix. A word with no coverage becomes <unk>.
class WordPieceTokenizer {
  public:
    void train(const std::string& text, int vocab_size);
    [[nodiscard]] std::vector<int> encode(const std::string& text) const;
    [[nodiscard]] std::string decode(const std::vector<int>& ids) const;

    [[nodiscard]] int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    [[nodiscard]] int unk_id() const { return unk_id_; }

    void save(const std::string& path) const;
    void load(const std::string& path);

  private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    int unk_id_ = 0;

    int add_token(const std::string& tok);
};

}  // namespace axon
