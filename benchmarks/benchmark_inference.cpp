#include <gpt2lab/model/gpt2.hpp>
#include <gpt2lab/core/tensor.hpp>
#include <gpt2lab/runtime/logger.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

using namespace gpt2lab;

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
        Gpt2Config config;
        gpt2lab::log_info("Initializing GPT-2 Model (124M) for Benchmarking...");
        Gpt2Model model(config);

        gpt2lab::log_info("Loading weights from " + model_path + "...");
        model.load_weights(model_path);
        gpt2lab::log_info("Weights loaded successfully!");

        // Start with a small prompt
        std::vector<float> context = {15496.0f, 11.0f, 616.0f, 1438.0f, 318.0f}; // "Hello, my name is"
        int num_tokens_to_generate = 20;

        std::cout << "\nStarting Autoregressive Generation Benchmark...\n";
        std::cout << "Prompt length: " << context.size() << " tokens\n";
        std::cout << "Generating " << num_tokens_to_generate << " tokens without KV Cache.\n\n";

        // Time to First Token (TTFT)
        auto ttft_start = std::chrono::high_resolution_clock::now();
        
        Tensor input({1, context.size()}, DType::Float32);
        for (size_t i = 0; i < context.size(); ++i) {
            input.data()[i] = context[i];
        }
        Tensor logits = model.forward(input);
        
        const float* logits_data = logits.data();
        const float* next_token_logits = logits_data + (context.size() - 1) * config.vocab_size;
        int next_token = argmax(next_token_logits, config.vocab_size);
        context.push_back(static_cast<float>(next_token));
        
        auto ttft_end = std::chrono::high_resolution_clock::now();
        double ttft_ms = std::chrono::duration<double, std::milli>(ttft_end - ttft_start).count();
        
        std::cout << "Time to First Token (TTFT): " << std::fixed << std::setprecision(2) << ttft_ms << " ms\n";

        // Generation Loop (Tokens per second)
        auto gen_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tokens_to_generate - 1; ++i) {
            Tensor step_input({1, context.size()}, DType::Float32);
            for (size_t j = 0; j < context.size(); ++j) {
                step_input.data()[j] = context[j];
            }
            
            Tensor step_logits = model.forward(step_input);
            const float* step_logits_data = step_logits.data();
            const float* cur_token_logits = step_logits_data + (context.size() - 1) * config.vocab_size;
            
            int cur_token = argmax(cur_token_logits, config.vocab_size);
            context.push_back(static_cast<float>(cur_token));
            
            std::cout << "." << std::flush; // Progress indicator
        }
        std::cout << "\n\n";
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_sec = std::chrono::duration<double>(gen_end - gen_start).count();
        double tokens_per_sec = (num_tokens_to_generate - 1) / gen_sec;
        double ms_per_token = (gen_sec * 1000.0) / (num_tokens_to_generate - 1);
        
        std::cout << "--- Benchmark Results ---\n";
        std::cout << "Generation Time: " << std::fixed << std::setprecision(2) << gen_sec << " s\n";
        std::cout << "Throughput:      " << std::fixed << std::setprecision(2) << tokens_per_sec << " tok/s\n";
        std::cout << "Latency:         " << std::fixed << std::setprecision(2) << ms_per_token << " ms/tok\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
