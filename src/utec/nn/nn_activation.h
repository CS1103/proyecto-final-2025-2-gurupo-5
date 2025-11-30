//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec::neural_network {

    /**
     * ReLU (Rectified Linear Unit) Activation Function
     * Forward: f(x) = max(0, x)
     * Backward: f'(x) = 1 if x > 0, else 0
     */
    template<typename T>
    class ReLU final : public ILayer<T> {
    private:
        using Tensor2 = Tensor<T, 2>;
        Tensor2 z_cache;  // Cache input for backward pass

    public:
        /**
         * Forward pass: Apply ReLU element-wise
         * f(x) = max(0, x)
         */
        Tensor2 forward(const Tensor2& z) override {
            // Cache input for backward pass
            z_cache = z;

            // Create result tensor with same shape
            Tensor2 result(z.shape());

            // Apply ReLU element-wise: max(0, x)
            auto& z_data = z.get_data();
            auto& result_data = result.get_data();

            for (size_t i = 0; i < z_data.size(); ++i) {
                result_data[i] = std::max(T{0}, z_data[i]);
            }

            return result;
        }

        /**
         * Backward pass: Compute gradient
         * Gradient: g * (z > 0 ? 1 : 0)
         * g: gradient from next layer
         */
        Tensor2 backward(const Tensor2& g) override {
            // Create gradient tensor with same shape
            Tensor2 grad(g.shape());

            auto& z_data = z_cache.get_data();
            auto& g_data = g.get_data();
            auto& grad_data = grad.get_data();

            // Apply derivative: g * (z > 0 ? 1 : 0)
            for (size_t i = 0; i < g_data.size(); ++i) {
                grad_data[i] = (z_data[i] > T{0}) ? g_data[i] : T{0};
            }

            return grad;
        }
    };

    /**
     * Sigmoid Activation Function
     * Forward: f(x) = 1 / (1 + exp(-x))
     * Backward: f'(x) = f(x) * (1 - f(x))
     */
    template<typename T>
    class Sigmoid final : public ILayer<T> {
    private:
        using Tensor2 = Tensor<T, 2>;
        Tensor2 sig_cache;  // Cache sigmoid output for backward pass

    public:
        /**
         * Forward pass: Apply sigmoid element-wise
         * f(x) = 1 / (1 + exp(-x))
         */
        Tensor2 forward(const Tensor2& z) override {
            // Create result tensor with same shape
            sig_cache = Tensor2(z.shape());

            auto& z_data = z.get_data();
            auto& sig_data = sig_cache.get_data();

            // Apply sigmoid element-wise
            for (size_t i = 0; i < z_data.size(); ++i) {
                sig_data[i] = T{1} / (T{1} + std::exp(-z_data[i]));
            }

            return sig_cache;
        }

        /**
         * Backward pass: Compute gradient
         * Gradient: g * sigmoid(z) * (1 - sigmoid(z))
         * g: gradient from next layer
         */
        Tensor2 backward(const Tensor2& g) override {
            // Create gradient tensor (copy of g)
            Tensor2 grad = g;

            auto& sig_data = sig_cache.get_data();
            auto& grad_data = grad.get_data();

            // Apply derivative: g * s * (1 - s)
            for (size_t i = 0; i < grad_data.size(); ++i) {
                const T s = sig_data[i];
                grad_data[i] *= s * (T{1} - s);
            }

            return grad;
        }
    };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
