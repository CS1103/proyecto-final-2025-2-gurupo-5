//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace utec::neural_network {

    /**
     * Mean Squared Error (MSE) Loss Function
     * Loss: L = (1/n) * sum((y_pred - y_true)^2)
     * Gradient: dL/dy_pred = (2/n) * (y_pred - y_true)
     */
    template<typename T>
    class MSELoss final : public ILoss<T, 2> {
    private:
        using Tensor2 = Tensor<T, 2>;
        Tensor2 y_pred;  // Predicted values
        Tensor2 y_true;  // True/target values

    public:
        /**
         * Constructor
         * @param y_prediction Predicted tensor
         * @param y_true True/target tensor
         */
        MSELoss(const Tensor2& y_prediction, const Tensor2& y_true)
            : y_pred(y_prediction), y_true(y_true)
        {
            // Shapes should match
            if (y_pred.shape() != y_true.shape()) {
                throw std::invalid_argument("Prediction and target shapes must match");
            }
        }

        /**
         * Compute MSE loss
         * L = (1/n) * sum((y_pred - y_true)^2)
         */
        T loss() const override {
            T sum = T{0};

            const auto& pred_data = y_pred.get_data();
            const auto& true_data = y_true.get_data();
            const size_t n = pred_data.size();

            // Sum of squared differences
            for (size_t i = 0; i < n; ++i) {
                const T diff = pred_data[i] - true_data[i];
                sum += diff * diff;
            }

            // Return mean
            return sum / static_cast<T>(n);
        }

        /**
         * Compute gradient of MSE loss
         * dL/dy_pred = (2/n) * (y_pred - y_true)
         */
        Tensor2 loss_gradient() const override {
            // Compute difference: y_pred - y_true
            Tensor2 grad = y_pred - y_true;

            // Scale by 2/n
            const T scale = T{2} / static_cast<T>(y_pred.size());

            return grad * scale;
        }
    };

    /**
     * Binary Cross Entropy (BCE) Loss Function
     * Loss: L = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
     * Gradient: dL/dy_pred = (1/n) * (y_pred - y_true) / (y_pred * (1 - y_pred))
     */
    template<typename T>
    class BCELoss final : public ILoss<T, 2> {
    private:
        using Tensor2 = Tensor<T, 2>;
        Tensor2 y_pred;  // Predicted values (probabilities)
        Tensor2 y_true;  // True/target values (0 or 1)

    public:
        /**
         * Constructor
         * @param y_prediction Predicted tensor (should be probabilities in [0, 1])
         * @param y_true True/target tensor (should be 0 or 1)
         */
        BCELoss(const Tensor2& y_prediction, const Tensor2& y_true)
            : y_pred(y_prediction), y_true(y_true)
        {
            // Shapes should match
            if (y_pred.shape() != y_true.shape()) {
                throw std::invalid_argument("Prediction and target shapes must match");
            }
        }

        /**
         * Compute BCE loss
         * L = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
         */
        T loss() const override {
            T sum = T{0};

            const auto& pred_data = y_pred.get_data();
            const auto& true_data = y_true.get_data();
            const size_t n = pred_data.size();

            // Small epsilon to avoid log(0)
            const T epsilon = std::numeric_limits<T>::epsilon();

            // Sum of cross entropy terms
            for (size_t i = 0; i < n; ++i) {
                // Clamp predictions to [epsilon, 1-epsilon] to avoid log(0)
                const T p = std::clamp(pred_data[i], epsilon, T{1} - epsilon);
                const T y = true_data[i];

                // BCE = -[y * log(p) + (1-y) * log(1-p)]
                sum += -y * std::log(p) - (T{1} - y) * std::log(T{1} - p);
            }

            // Return mean
            return sum / static_cast<T>(n);
        }

        /**
         * Compute gradient of BCE loss
         * dL/dy_pred = (1/n) * (y_pred - y_true) / (y_pred * (1 - y_pred))
         */
        Tensor2 loss_gradient() const override {
            Tensor2 grad(y_pred.shape());

            const auto& pred_data = y_pred.get_data();
            const auto& true_data = y_true.get_data();
            auto& grad_data = grad.get_data();
            const size_t n = pred_data.size();

            // Small epsilon to avoid division by zero
            const T epsilon = std::numeric_limits<T>::epsilon();

            for (size_t i = 0; i < n; ++i) {
                // Clamp predictions to [epsilon, 1-epsilon]
                const T p = std::clamp(pred_data[i], epsilon, T{1} - epsilon);
                const T y = true_data[i];

                // Gradient: (p - y) / (p * (1 - p) * n)
                grad_data[i] = (p - y) / (p * (T{1} - p) * static_cast<T>(n));
            }

            return grad;
        }
    };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
