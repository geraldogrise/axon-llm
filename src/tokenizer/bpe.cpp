#include "axon/tokenizer.h"

#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>

namespace axon {

namespace {
constexpr const char* kEow = "</w>";  // end-of-word marker
constexpr const char* kUnk = "<unk>";
constexpr const char* kBos = "<bos>";  // beginning of sequence
constexpr const char* kEos = "<eos>";  // end of sequence
constexpr const char* kPad = "<pad>";  // padding

bool is_special(const std::string& tok) {
    return tok == kUnk || tok == kBos || tok == kEos || tok == kPad;
}

bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

// Split the text into words (separated by any whitespace).
std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (char c : text) {
        if (is_space(c)) {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        words.push_back(cur);
    }
    return words;
}

// Word -> symbols: each byte becomes a symbol, plus "</w>" at the end.
std::vector<std::string> word_to_symbols(const std::string& word) {
    std::vector<std::string> syms;
    syms.reserve(word.size() + 1);
    for (char c : word) {
        syms.emplace_back(1, c);
    }
    syms.emplace_back(kEow);
    return syms;
}

// Merge all adjacent occurrences of the pair (a,b) into `ab` in the word.
void merge_in_word(std::vector<std::string>& syms, const std::string& a, const std::string& b,
                   const std::string& ab) {
    std::vector<std::string> out;
    out.reserve(syms.size());
    std::size_t i = 0;
    while (i < syms.size()) {
        if (i + 1 < syms.size() && syms[i] == a && syms[i + 1] == b) {
            out.push_back(ab);
            i += 2;
        } else {
            out.push_back(syms[i]);
            ++i;
        }
    }
    syms.swap(out);
}
}  // namespace

int BPETokenizer::add_token(const std::string& tok) {
    auto it = token_to_id_.find(tok);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    const int id = static_cast<int>(id_to_token_.size());
    token_to_id_[tok] = id;
    id_to_token_.push_back(tok);
    return id;
}

void BPETokenizer::train(const std::string& text, int target_vocab) {
    token_to_id_.clear();
    id_to_token_.clear();
    merges_.clear();
    // Special tokens reserved at the start of the vocabulary (ids 0,1,2,3).
    unk_id_ = add_token(kUnk);
    bos_id_ = add_token(kBos);
    eos_id_ = add_token(kEos);
    pad_id_ = add_token(kPad);

    // Frequency of each word and its symbol representation.
    std::map<std::string, int> word_freq;
    for (const std::string& w : split_words(text)) {
        word_freq[w]++;
    }
    std::vector<std::vector<std::string>> words;
    std::vector<int> freqs;
    for (const auto& [w, f] : word_freq) {
        std::vector<std::string> syms = word_to_symbols(w);
        for (const std::string& s : syms) {
            add_token(s);  // register initial symbols in the vocabulary
        }
        words.push_back(std::move(syms));
        freqs.push_back(f);
    }

    // Merge the most frequent adjacent pair until reaching the target vocabulary.
    while (vocab_size() < target_vocab) {
        std::map<std::pair<std::string, std::string>, int> pair_freq;
        for (std::size_t w = 0; w < words.size(); ++w) {
            const auto& syms = words[w];
            for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
                pair_freq[{syms[i], syms[i + 1]}] += freqs[w];
            }
        }
        if (pair_freq.empty()) {
            break;  // nothing more to merge
        }
        // Best pair (highest frequency; deterministic tie-break by map order).
        std::pair<std::string, std::string> best;
        int best_count = -1;
        for (const auto& [pr, cnt] : pair_freq) {
            if (cnt > best_count) {
                best_count = cnt;
                best = pr;
            }
        }
        const std::string ab = best.first + best.second;
        merges_.push_back(best);
        add_token(ab);
        for (auto& syms : words) {
            merge_in_word(syms, best.first, best.second, ab);
        }
    }
}

void BPETokenizer::apply_merges(std::vector<std::string>& symbols) const {
    for (const auto& [a, b] : merges_) {
        merge_in_word(symbols, a, b, a + b);
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const std::string& w : split_words(text)) {
        std::vector<std::string> syms = word_to_symbols(w);
        apply_merges(syms);
        for (const std::string& s : syms) {
            auto it = token_to_id_.find(s);
            ids.push_back(it != token_to_id_.end() ? it->second : unk_id_);
        }
    }
    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) {
            throw std::out_of_range("decode: invalid token id");
        }
        const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];
        if (is_special(tok)) {
            continue;  // don't render <unk>/<bos>/<eos>
        }
        out += tok;
    }
    // "</w>" -> space.
    std::string result;
    for (std::size_t i = 0; i < out.size();) {
        if (out.compare(i, 4, kEow) == 0) {
            result.push_back(' ');
            i += 4;
        } else {
            result.push_back(out[i]);
            ++i;
        }
    }
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

void BPETokenizer::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("tokenizer save: could not open " + path);
    }
    os << id_to_token_.size() << "\n";
    for (const std::string& tok : id_to_token_) {
        os << tok << "\n";
    }
    os << merges_.size() << "\n";
    for (const auto& [a, b] : merges_) {
        os << a << " " << b << "\n";
    }
}

void BPETokenizer::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("tokenizer load: could not open " + path);
    }
    token_to_id_.clear();
    id_to_token_.clear();
    merges_.clear();

    std::string line;
    std::getline(is, line);
    const int n = std::stoi(line);
    for (int i = 0; i < n; ++i) {
        std::getline(is, line);
        token_to_id_[line] = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(line);
    }
    std::getline(is, line);
    const int m = std::stoi(line);
    for (int i = 0; i < m; ++i) {
        std::getline(is, line);
        const std::size_t sp = line.find(' ');
        merges_.emplace_back(line.substr(0, sp), line.substr(sp + 1));
    }
    auto find_id = [this](const char* name, int fallback) {
        auto it = token_to_id_.find(name);
        return it != token_to_id_.end() ? it->second : fallback;
    };
    unk_id_ = find_id(kUnk, 0);
    bos_id_ = find_id(kBos, 1);
    eos_id_ = find_id(kEos, 2);
    pad_id_ = find_id(kPad, 3);
}

}  // namespace axon
