//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H


#include <vector>
#include <memory>
#include <string>
#include <filesystem>
#include <limits>
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
        void train(const Tensor2& X, const Tensor2& Y, size_t epochs, size_t batch_size, T lr,
                   const std::string& model_path = "", size_t patience = 2) {
            OptimizerType<T> optimizer(lr);

            const size_t num_samples = X.shape()[0];
            const size_t num_features = X.shape()[1];
            const size_t num_outputs = Y.shape()[1];
            const size_t num_batches = (num_samples + batch_size - 1) / batch_size;

            std::cout << "Entrenando con " << num_samples << " muestras en "
                      << num_batches << " batches de tamanio " << batch_size << std::endl;

            // Early stopping variables
            T best_loss = std::numeric_limits<T>::max();
            size_t epochs_without_improvement = 0;
            size_t best_epoch = 0;

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                std::cout << "Epoca " << (epoch + 1) << "/" << epochs << std::flush;

                T epoch_loss = 0.0;

                // Iterar sobre mini-batches
                for (size_t b = 0; b < num_batches; ++b) {
                    // Calcular tamaño del batch actual
                    size_t current_batch_start = b * batch_size;
                    size_t current_batch_size = std::min(batch_size, num_samples - current_batch_start);

                    // Crear tensores para el mini-batch
                    Tensor2 X_batch(current_batch_size, num_features);
                    Tensor2 Y_batch(current_batch_size, num_outputs);

                    // Copiar datos del batch
                    for (size_t i = 0; i < current_batch_size; ++i) {
                        size_t src_idx = current_batch_start + i;
                        for (size_t j = 0; j < num_features; ++j) {
                            X_batch(i, j) = X(src_idx, j);
                        }
                        for (size_t j = 0; j < num_outputs; ++j) {
                            Y_batch(i, j) = Y(src_idx, j);
                        }
                    }

                    // Forward pass
                    Tensor2 output = X_batch;
                    for (auto& layer : layers_) {
                        output = layer->forward(output);
                    }

                    // Calcular loss
                    LossType<T> loss_fn(output, Y_batch);
                    T batch_loss = loss_fn.loss();
                    epoch_loss += batch_loss;

                    // Backward pass
                    Tensor2 grad = loss_fn.loss_gradient();
                    for (size_t i = layers_.size(); i-- > 0; ) {
                        grad = layers_[i]->backward(grad);
                        if (auto* dense = dynamic_cast<Dense<T>*>(layers_[i].get())) {
                            dense->update_params(optimizer);
                        }
                    }

                    // Mostrar progreso cada 10 batches
                    if ((b + 1) % 10 == 0 || (b + 1) == num_batches) {
                        std::cout << "\r  Epoca " << (epoch + 1) << "/" << epochs
                                  << " - Batch " << (b + 1) << "/" << num_batches << std::flush;
                    }
                }

                // Promedio de loss por época
                epoch_loss /= num_batches;
                std::cout << " - Loss: " << epoch_loss;

                // Early stopping logic
                if (epoch_loss < best_loss) {
                    best_loss = epoch_loss;
                    best_epoch = epoch + 1;
                    epochs_without_improvement = 0;

                    // Guardar mejor modelo si se proporciono path
                    if (!model_path.empty()) {
                        save_model(model_path);
                        std::cout << " [BEST - Guardado]";
                    } else {
                        std::cout << " [BEST]";
                    }
                } else {
                    epochs_without_improvement++;
                    std::cout << " (sin mejora: " << epochs_without_improvement << "/" << patience << ")";
                }

                std::cout << std::endl;

                // Detener si no hay mejora por 'patience' epocas
                if (epochs_without_improvement >= patience) {
                    std::cout << "\n*** EARLY STOPPING ***" << std::endl;
                    std::cout << "No hubo mejora en " << patience << " epocas consecutivas." << std::endl;
                    std::cout << "Mejor loss: " << best_loss << " (Epoca " << best_epoch << ")" << std::endl;
                    break;
                }
            }
            std::cout << "\nEntrenamiento completado." << std::endl;
            std::cout << "Mejor modelo: Epoca " << best_epoch << " con Loss: " << best_loss << std::endl;
        }
        Tensor2 predict(const Tensor2& X) {
            Tensor2 out = X;
            for (auto& layer : layers_) { out = layer->forward(out); }
            return out;
        }

        /**
         * Save model weights to directory
         * @param model_dir Directory to save weights (will be created if doesn't exist)
         */
        void save_model(const std::string& model_dir) {
            // Create directory if it doesn't exist
            std::filesystem::create_directories(model_dir);

            std::cout << "Guardando modelo en: " << model_dir << std::endl;

            size_t dense_layer_idx = 0;
            for (size_t i = 0; i < layers_.size(); ++i) {
                if (auto* dense = dynamic_cast<Dense<T>*>(layers_[i].get())) {
                    std::string weights_file = model_dir + "/layer" + std::to_string(dense_layer_idx) + "_weights.txt";
                    std::string biases_file = model_dir + "/layer" + std::to_string(dense_layer_idx) + "_biases.txt";

                    dense->save_weights(weights_file, biases_file);
                    std::cout << "  - Capa Dense " << dense_layer_idx << " guardada" << std::endl;
                    dense_layer_idx++;
                }
            }
            std::cout << "Modelo guardado exitosamente!" << std::endl;
        }

        /**
         * Load model weights from directory
         * @param model_dir Directory containing saved weights
         * @return true if loaded successfully, false if files don't exist
         */
        bool load_model(const std::string& model_dir) {
            // Check if directory exists
            if (!std::filesystem::exists(model_dir)) {
                return false;
            }

            std::cout << "Cargando modelo desde: " << model_dir << std::endl;

            size_t dense_layer_idx = 0;
            for (size_t i = 0; i < layers_.size(); ++i) {
                if (auto* dense = dynamic_cast<Dense<T>*>(layers_[i].get())) {
                    std::string weights_file = model_dir + "/layer" + std::to_string(dense_layer_idx) + "_weights.txt";
                    std::string biases_file = model_dir + "/layer" + std::to_string(dense_layer_idx) + "_biases.txt";

                    // Check if files exist
                    if (!std::filesystem::exists(weights_file) || !std::filesystem::exists(biases_file)) {
                        std::cout << "  - Archivos de pesos no encontrados" << std::endl;
                        return false;
                    }

                    dense->load_weights(weights_file, biases_file);
                    std::cout << "  - Capa Dense " << dense_layer_idx << " cargada" << std::endl;
                    dense_layer_idx++;
                }
            }
            std::cout << "Modelo cargado exitosamente!" << std::endl;
            return true;
        }
    };

}








#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
