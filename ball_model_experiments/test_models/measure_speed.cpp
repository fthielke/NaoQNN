// SPDX-License-Identifier: MIT
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <CompiledNN/Model.h>
#include <CompiledNN/CompiledNN.h>

constexpr size_t DATASET_SIZE = 5252;
constexpr size_t NUM_WARMUP_RUNS = 100;
constexpr size_t NUM_TEST_RUNS = 100;
constexpr size_t NUM_MEASURE_RUNS = 100;

extern "C" {
    extern const unsigned char test_data[DATASET_SIZE*32*32];
    void run_gerrit_original_quantized_gerrit_original_quantized_gerrit_original_quantized(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
    void run_gerrit_validpad_quantized_gerrit_quantized_gerrit_quantized(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
    void run_gerrit_validpad_splitconvs_quantized_gerrit_quantized_gerrit_quantized(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
    void run_4_gerrit_quantized_gerrit_quantized(const unsigned char* input, unsigned char* output_class, unsigned char* output_detection, unsigned int batch_size);
}

alignas(16) unsigned char output_class[DATASET_SIZE];
alignas(64) unsigned char output_buffer[DATASET_SIZE * 32 * 32 * 8];

int main() {
    // Float models
    for(auto model_name : {"gerrit_original_gerrit_original_gerrit_original.h5", "gerrit_validpad_gerrit_original_gerrit_original.h5", "gerrit_validpad_splitconvs_gerrit_original_gerrit_original.h5", "4_float_gerrit_original_gerrit_original.h5"}) {
        NeuralNetwork::Model model(model_name);
        model.setInputUInt8(0);
        NeuralNetwork::CompiledNN nn;
        nn.compile(model);

        // Warmup
        for(size_t i = 0; i < NUM_WARMUP_RUNS; i++) {
            nn.apply();
        }

        double min_time_us = 10000000.;
        for(size_t i = 0; i < NUM_MEASURE_RUNS; i++) {
            auto start = std::chrono::steady_clock::now();
            for(size_t j = 0; j < NUM_TEST_RUNS; j++) {
                std::copy_n(test_data, 32*32, nn.input(0).data());
                nn.apply();
            }
            auto end = std::chrono::steady_clock::now();

            const std::chrono::duration<double> diff = end - start;
            const double diff_us = diff.count() * 1000000. / static_cast<double>(NUM_TEST_RUNS);
            if(diff_us < min_time_us)
                min_time_us = diff_us;
        }

        std::cout << "Model " << model_name << ":\n" << min_time_us << " us per patch\n\n";
    }

    // Quantized models
    for(auto model : {std::make_pair<>("gerrit_original_quantized_gerrit_original_quantized_gerrit_original_quantized", run_gerrit_original_quantized_gerrit_original_quantized_gerrit_original_quantized), std::make_pair<>("gerrit_validpad_quantized_gerrit_quantized_gerrit_quantized", run_gerrit_validpad_quantized_gerrit_quantized_gerrit_quantized), std::make_pair<>("gerrit_validpad_splitconvs_quantized_gerrit_quantized_gerrit_quantized", run_gerrit_validpad_splitconvs_quantized_gerrit_quantized_gerrit_quantized), std::make_pair<>("4_gerrit_quantized_gerrit_quantized", run_4_gerrit_quantized_gerrit_quantized)}) {
        const std::string model_name(model.first);
        auto inference_func = model.second;

        std::cout << "Model " << model_name << ":\n";
        for(const unsigned int batch_size : {1, 2, 4, 8, 16, 32, 64}) {
            // Warmup
            for(size_t i = 0; i < NUM_WARMUP_RUNS; i++) {
                inference_func(test_data, output_class, output_buffer, batch_size);
            }

            double min_time_us = 10000000.;
            for(size_t i = 0; i < NUM_MEASURE_RUNS; i++) {
                auto start = std::chrono::steady_clock::now();
                for(size_t j = 0; j < NUM_TEST_RUNS; j++) {
                    inference_func(test_data, output_class, output_buffer, batch_size);
                }
                auto end = std::chrono::steady_clock::now();

                const std::chrono::duration<double> diff = end - start;
                const double diff_us = diff.count() * 1000000. / static_cast<double>(batch_size * NUM_TEST_RUNS);
                if(diff_us < min_time_us)
                    min_time_us = diff_us;
            }

            std::cout << "Batch size " << std::setw(2) << batch_size << ": " << min_time_us << " us per patch\n";
        }
        std::cout << "\n";
    }

    return 0;
}
