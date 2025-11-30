//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H


#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec::neural_network {

    template <typename T>
    class SGD : public IOptimizer<T> {
        T learning_rate_;
    public:
        explicit SGD(T lr) : learning_rate_(lr) { }
        void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
            for (size_t i = 0; i < param.size(); ++i) { param.get_data()[i] -= learning_rate_ * grad.get_data()[i]; }
        }
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
        T learning_rate_;
        T beta1_, beta2_, epsilon_;
        Tensor<T, 2> m, v;
        size_t t = 0;
    public:
        Adam(T lr = 0.001, T b1 = 0.9, T b2 = 0.999, T eps = 1e-8) : learning_rate_(lr), beta1_(b1), beta2_(b2), epsilon_(eps) { }
        void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
            if (m.shape() != grad.shape()) { m = Tensor<T, 2>(grad.shape()); m.fill(0); }
            if (v.shape() != grad.shape()) { v = Tensor<T, 2>(grad.shape()); v.fill(0); }
            ++t;
            for (size_t i = 0; i < grad.size(); ++i) {
                m.get_data()[i] = beta1_ * m.get_data()[i] + (1 - beta1_) * grad.get_data()[i];
                v.get_data()[i] = beta2_ * v.get_data()[i] + (1 - beta2_) * grad.get_data()[i] * grad.get_data()[i];
                T m_hat = m.get_data()[i] / (1 - std::pow(beta1_, t));
                T v_hat = v.get_data()[i] / (1 - std::pow(beta2_, t));
                param.get_data()[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
        void step() override { }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
