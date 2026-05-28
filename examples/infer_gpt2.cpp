#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/runtime/logger.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace gpt2lab;

// Simple argmax to get the most likely next token
int argmax(const float* logits, size_t vocab_size) {
    int best_idx = 0;
    float best_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int main(int argc, char** argv) {
    std::string model_path = "gpt2_weights.bin";
    if (argc > 1) {
        model_path = argv[1];
    }

    try {
        Gpt2Config config; // Defaults to 124M GPT-2
        
        gpt2lab::log_info("Initializing GPT-2 Model (124M)...");
        Gpt2Model model(config);

        gpt2lab::log_info("Loading weights from " + model_path + "...");
        model.load_weights(model_path);
        gpt2lab::log_info("Weights loaded successfully!");

        // Token sequence: "Hello, my name is"
        // Tokens: [15496, 11, 616, 1438, 318]
        std::vector<float> input_tokens = {15496.0f, 11.0f, 616.0f, 1438.0f, 318.0f};
        
        Tensor input({1, input_tokens.size()}, DType::Float32);
        for (size_t i = 0; i < input_tokens.size(); ++i) {
            input.data()[i] = input_tokens[i];
        }

        gpt2lab::log_info("Running forward pass...");
        Tensor logits = model.forward(input);

        // The logits tensor shape is [1, seq_len, vocab_size]
        size_t seq_len = input_tokens.size();
        size_t vocab_size = config.vocab_size;

        const float* logits_data = logits.data();
        
        // Predict the next token for the last position
        const float* next_token_logits = logits_data + (seq_len - 1) * vocab_size;
        int next_token = argmax(next_token_logits, vocab_size);

        std::cout << "\nInput sequence length: " << seq_len << "\n";
        std::cout << "Predicted next token ID: " << next_token << "\n";
        
        // " GPT" -> 304, " John" -> 1757, etc.

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
