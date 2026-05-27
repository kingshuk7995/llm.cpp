#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <stdexcept>

namespace gpt2lab {

struct Gpt2Config {
  int vocab_size = 50257;
  int hidden_size = 768;
  int num_layers = 12;
  int num_heads = 12;
  int seq_len = 1024;
  float dropout_prob = 0.1f;

  static Gpt2Config from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open config file: " + path);
    }
    nlohmann::json j;
    f >> j;

    Gpt2Config config;
    if (j.contains("vocab_size")) config.vocab_size = j["vocab_size"];
    if (j.contains("hidden_size")) config.hidden_size = j["hidden_size"];
    if (j.contains("num_layers")) config.num_layers = j["num_layers"];
    if (j.contains("num_heads")) config.num_heads = j["num_heads"];
    if (j.contains("seq_len")) config.seq_len = j["seq_len"];
    if (j.contains("dropout_prob")) config.dropout_prob = j["dropout_prob"];
    
    return config;
  }
};

} // namespace gpt2lab
