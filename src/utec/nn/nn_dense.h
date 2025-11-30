//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>

namespace utec::neural_network {

    /**
     * Dense (fully connected) layer implementation
     * Forward: Y = X @ W + b
     * Backward: Computes gradients for W, b, and X
     */
    template<typename T>
    class Dense final : public ILayer<T> {
    private:
        using Tensor2 = Tensor<T, 2>;

        // Parameters
        Tensor2 W;        // Weights: (in_features, out_features)
        Tensor2 b;        // Bias: (1, out_features)

        // Gradients
        Tensor2 dW;       // Gradient of W
        Tensor2 db;       // Gradient of b

        // Cache for backward pass
        Tensor2 X_cache;  // Cached input from forward pass

        // Xavier/Glorot initialization
        void initialize_weights(size_t in_features, size_t out_features) {
            std::random_device rd;
            std::mt19937 gen(rd());
            T stddev = std::sqrt(T(2.0) / (in_features + out_features));
            std::normal_distribution<T> dist(T(0), stddev);

            // Initialize weights with Xavier initialization
            for (size_t i = 0; i < in_features; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    W(i, j) = dist(gen);
                }
            }

            // Initialize biases to zero
            for (size_t j = 0; j < out_features; ++j) {
                b(0, j) = T{0};
            }
        }

    public:
        /**
         * Constructor with automatic Xavier initialization
         * Creates a Dense layer with properly initialized weights and biases
         */
        Dense(size_t in_features, size_t out_features)
            : W(in_features, out_features)
            , b(1, out_features)
            , dW(in_features, out_features)
            , db(1, out_features)
            , X_cache(1, 1)
        {
            // Initialize weights using Xavier initialization
            initialize_weights(in_features, out_features);
        }

        /**
         * Constructor with custom initializers
         * Initializers are functions that take a Tensor2& and initialize its values
         */
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_features, size_t out_features,
              InitWFun init_w_fun, InitBFun init_b_fun)
            : W(in_features, out_features)
            , b(1, out_features)
            , dW(in_features, out_features)
            , db(1, out_features)
            , X_cache(1, 1)
        {
            // Initialize weights and biases using provided functions
            init_w_fun(W);
            init_b_fun(b);
        }

        /**
         * Forward pass: Y = X @ W + b
         * X: (batch_size, in_features)
         * W: (in_features, out_features)
         * b: (1, out_features)
         * Returns: Y of shape (batch_size, out_features)
         */
        Tensor2 forward(const Tensor2& X) override {
            // Cache input for backward pass
            X_cache = X;

            // Compute Y = X @ W
            Tensor2 Y = matrix_product(X, W);

            // Add bias: Y += b (broadcasted across batch dimension)
            const size_t batch_size = Y.shape()[0];
            const size_t out_features = Y.shape()[1];

            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    Y(i, j) += b(0, j);
                }
            }

            return Y;
        }

        /**
         * Backward pass: Computes gradients
         * dZ: gradient from next layer, shape (batch_size, out_features)
         * Computes:
         *   - dW = X^T @ dZ
         *   - db = sum(dZ, axis=0)
         *   - dX = dZ @ W^T (returned)
         * Returns: dX of shape (batch_size, in_features)
         */
        Tensor2 backward(const Tensor2& dZ) override {
            // Compute gradient w.r.t. weights: dW = X^T @ dZ
            dW = matrix_product(transpose_2d(X_cache), dZ);

            // Compute gradient w.r.t. bias: db = sum(dZ, axis=0)
            const size_t batch_size = dZ.shape()[0];
            const size_t out_features = dZ.shape()[1];

            db = Tensor2(1, out_features);
            for (size_t j = 0; j < out_features; ++j) {
                T sum = T{0};
                for (size_t i = 0; i < batch_size; ++i) {
                    sum += dZ(i, j);
                }
                db(0, j) = sum;
            }

            // Compute gradient w.r.t. input: dX = dZ @ W^T
            return matrix_product(dZ, transpose_2d(W));
        }

        /**
         * Update parameters using optimizer
         */
        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(W, dW);
            optimizer.update(b, db);
        }

        /**
         * Save weights and biases to files
         * @param weights_file Path to save weights
         * @param biases_file Path to save biases
         */
        void save_weights(const std::string& weights_file, const std::string& biases_file) const {
            // Save weights
            std::ofstream wfile(weights_file);
            if (!wfile.is_open()) {
                throw std::runtime_error("Cannot open file for writing: " + weights_file);
            }

            const size_t in_features = W.shape()[0];
            const size_t out_features = W.shape()[1];

            wfile << in_features << " " << out_features << "\n";
            for (size_t i = 0; i < in_features; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    wfile << W(i, j);
                    if (j < out_features - 1) wfile << " ";
                }
                wfile << "\n";
            }
            wfile.close();

            // Save biases
            std::ofstream bfile(biases_file);
            if (!bfile.is_open()) {
                throw std::runtime_error("Cannot open file for writing: " + biases_file);
            }

            bfile << out_features << "\n";
            for (size_t j = 0; j < out_features; ++j) {
                bfile << b(0, j);
                if (j < out_features - 1) bfile << " ";
            }
            bfile << "\n";
            bfile.close();
        }

        /**
         * Load weights and biases from files
         * @param weights_file Path to load weights from
         * @param biases_file Path to load biases from
         */
        void load_weights(const std::string& weights_file, const std::string& biases_file) {
            // Load weights
            std::ifstream wfile(weights_file);
            if (!wfile.is_open()) {
                throw std::runtime_error("Cannot open file for reading: " + weights_file);
            }

            size_t in_features, out_features;
            wfile >> in_features >> out_features;

            if (W.shape()[0] != in_features || W.shape()[1] != out_features) {
                throw std::runtime_error("Weight dimensions mismatch");
            }

            for (size_t i = 0; i < in_features; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    wfile >> W(i, j);
                }
            }
            wfile.close();

            // Load biases
            std::ifstream bfile(biases_file);
            if (!bfile.is_open()) {
                throw std::runtime_error("Cannot open file for reading: " + biases_file);
            }

            size_t bias_size;
            bfile >> bias_size;

            if (b.shape()[1] != bias_size) {
                throw std::runtime_error("Bias dimensions mismatch");
            }

            for (size_t j = 0; j < bias_size; ++j) {
                bfile >> b(0, j);
            }
            bfile.close();
        }

        /**
         * Get weights tensor (for inspection/saving)
         */
        const Tensor2& get_weights() const {
            return W;
        }

        /**
         * Get biases tensor (for inspection/saving)
         */
        const Tensor2& get_biases() const {
            return b;
        }
    };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
