//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H


#include <vector>
#include <memory>
#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_dense.h"
#include "nn_loss.h"
#include "tensor.h"
using namespace std;

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        using Tensor2 = Tensor<T, 2>;
        vector<std::unique_ptr<ILayer<T>>> layers_;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) { layers_.emplace_back(std::move(layer)); }

        template<template <typename...> class LossType, template <typename...> class OptimizerType = SGD>
        void train(const Tensor2& X, const Tensor2& Y, size_t epochs, size_t batch_size, T lr) {
            OptimizerType<T> optimizer(lr);
            std::cout << "Entrenando con " << X.shape()[0] << " muestras..." << std::endl;

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                std::cout << "Epoca " << (epoch + 1) << "/" << epochs << "..." << std::endl;

                Tensor2 output = X;
                for (auto& layer : layers_) { output = layer->forward(output); }
                LossType<T> loss_fn(output, Y);
                T loss_value = loss_fn.loss();

                std::cout << "  Loss: " << loss_value << std::endl;

                Tensor2 grad = loss_fn.loss_gradient();
                for (size_t i = layers_.size(); i-- > 0; ) {
                    grad = layers_[i]->backward(grad);
                    if (auto* dense = dynamic_cast<Dense<T>*>(layers_[i].get())) {
                        dense->update_params(optimizer);
                    }
                }
            }
            std::cout << "Entrenamiento completado." << std::endl;
        }
        Tensor2 predict(const Tensor2& X) {
            Tensor2 out = X;
            for (auto& layer : layers_) { out = layer->forward(out); }
            return out;
        }
    };

}








#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
