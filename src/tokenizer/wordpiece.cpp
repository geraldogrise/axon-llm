#include "axon/tokenizer.h"

#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>

namespace axon {

namespace {
constexpr const char* kUnk = "[UNK]";
constexpr const char* kCont = "##";  // continuation piece prefix

bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (char c : text) {
        if (is_space(c)) {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

// Merge the pair (a,b) -> ab in all adjacent occurrences.
void merge_in_word(std::vector<std::string>& s, const std::string& a, const std::string& b,
                   const std::string& ab) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) { out.push_back(ab); i += 2; }
        else { out.push_back(s[i]); ++i; }
    }
    s.swap(out);
}
}  // namespace

int WordPieceTokenizer::add_token(const std::string& tok) {
    auto it = token_to_id_.find(tok);
    if (it != token_to_id_.end()) return it->second;
    const int id = static_cast<int>(id_to_token_.size());
    token_to_id_[tok] = id;
    id_to_token_.push_back(tok);
    return id;
}

void WordPieceTokenizer::train(const std::string& text, int target_vocab) {
    token_to_id_.clear();
    id_to_token_.clear();
    unk_id_ = add_token(kUnk);

    // Word frequencies and their decomposition into characters.
    std::map<std::string, int> word_freq;
    for (const std::string& w : split_words(text)) word_freq[w]++;

    std::vector<std::vector<std::string>> words;
    std::vector<int> freqs;
    for (const auto& [w, f] : word_freq) {
        std::vector<std::string> syms;
        for (char c : w) syms.emplace_back(1, c);
        // register each character as a piece (initial and continuation)
        for (std::size_t i = 0; i < syms.size(); ++i) {
            add_token(syms[i]);
            add_token(kCont + syms[i]);
        }
        words.push_back(std::move(syms));
        freqs.push_back(f);
    }

    // Learn subwords by merging the most frequent adjacent pair (BPE style),
    // adding each resulting piece in both initial and "##continuation" forms.
    while (vocab_size() < target_vocab) {
        std::map<std::pair<std::string, std::string>, int> pair_freq;
        for (std::size_t w = 0; w < words.size(); ++w) {
            const auto& s = words[w];
            for (std::size_t i = 0; i + 1 < s.size(); ++i) pair_freq[{s[i], s[i + 1]}] += freqs[w];
        }
        if (pair_freq.empty()) break;
        std::pair<std::string, std::string> best;
        int best_count = -1;
        for (const auto& [pr, cnt] : pair_freq) {
            if (cnt > best_count) { best_count = cnt; best = pr; }
        }
        const std::string ab = best.first + best.second;
        add_token(ab);
        add_token(kCont + ab);
        for (auto& s : words) merge_in_word(s, best.first, best.second, ab);
    }
}

std::vector<int> WordPieceTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const std::string& w : split_words(text)) {
        std::vector<int> word_ids;
        std::size_t i = 0;
        bool first = true;
        bool ok = true;
        while (i < w.size()) {
            std::size_t end = w.size();
            int found = -1;
            std::size_t found_end = i;
            for (; end > i; --end) {
                std::string piece = w.substr(i, end - i);
                if (!first) piece = kCont + piece;
                auto it = token_to_id_.find(piece);
                if (it != token_to_id_.end()) { found = it->second; found_end = end; break; }
            }
            if (found < 0) { ok = false; break; }
            word_ids.push_back(found);
            i = found_end;
            first = false;
        }
        if (ok) {
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        } else {
            ids.push_back(unk_id_);  // WordPiece: word with no coverage -> [UNK]
        }
    }
    return ids;
}

std::string WordPieceTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) {
            throw std::out_of_range("wordpiece decode: invalid id");
        }
        const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];
        if (tok == kUnk) continue;
        if (tok.rfind(kCont, 0) == 0) {
            out += tok.substr(2);  // continuation: glue without space
        } else {
            if (!out.empty()) out += ' ';
            out += tok;
        }
    }
    return out;
}

void WordPieceTokenizer::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("wordpiece save: could not open " + path);
    os << id_to_token_.size() << "\n";
    for (const std::string& t : id_to_token_) os << t << "\n";
}

void WordPieceTokenizer::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("wordpiece load: could not open " + path);
    token_to_id_.clear();
    id_to_token_.clear();
    std::string line;
    std::getline(is, line);
    const int n = std::stoi(line);
    for (int i = 0; i < n; ++i) {
        std::getline(is, line);
        token_to_id_[line] = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(line);
    }
    auto it = token_to_id_.find(kUnk);
    unk_id_ = (it != token_to_id_.end()) ? it->second : 0;
}

}  // namespace axon
