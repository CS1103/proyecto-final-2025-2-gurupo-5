#include "../src/utec/nn/neural_network.h"
#include "../src/utec/nn/nn_dense.h"
#include "../src/utec/nn/nn_activation.h"
#include "../src/utec/nn/nn_loss.h"
#include "../src/utec/nn/nn_optimizer.h"
#include "../include/utec/data/medical_mnist_loader.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace std;
using namespace utec::neural_network;
using namespace utec::data;

/**
 * EJEMPLO 2: Entrenamiento Medical MNIST
 *
 * Dataset: 58,954 imagenes medicas de 64x64 pixeles en escala de grises
 * Clases: 6 (AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT)
 * Arquitectura: 4096 -> 256 -> 128 -> 6
 * Loss: SoftmaxCrossEntropy (clasificacion multiclase)
 * Optimizer: Adam
 */

// Funcion para aplicar Softmax a logits
Tensor<float, 2> apply_softmax(const Tensor<float, 2>& logits) {
    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    Tensor<float, 2> probabilities(batch_size, num_classes);

    for (size_t i = 0; i < batch_size; ++i) {
        // Encontrar max para estabilidad numerica
        float max_logit = logits(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            max_logit = max(max_logit, logits(i, j));
        }

        // Calcular exp(x - max) y suma
        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; ++j) {
            probabilities(i, j) = exp(logits(i, j) - max_logit);
            sum_exp += probabilities(i, j);
        }

        // Normalizar para obtener probabilidades
        for (size_t j = 0; j < num_classes; ++j) {
            probabilities(i, j) /= sum_exp;
        }
    }

    return probabilities;
}

// Funcion para calcular accuracy
float calculate_accuracy(const Tensor<float, 2>& predictions, const Tensor<float, 2>& labels) {
    size_t num_samples = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];
    size_t correct = 0;

    for (size_t i = 0; i < num_samples; ++i) {
        // Encontrar clase predicha (indice con valor maximo)
        size_t pred_class = 0;
        float max_pred = predictions(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            if (predictions(i, j) > max_pred) {
                max_pred = predictions(i, j);
                pred_class = j;
            }
        }

        // Encontrar clase verdadera (indice con valor 1.0 en one-hot)
        size_t true_class = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (labels(i, j) > 0.5f) {  // Es 1.0
                true_class = j;
                break;
            }
        }

        if (pred_class == true_class) {
            correct++;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(num_samples);
}

int main() {
    cout << "=== ENTRENAMIENTO DE RED NEURONAL - MEDICAL MNIST ===\n\n";

    // 1. CARGAR LOS DATOS
    cout << "PASO 1: Cargando datos del dataset Medical MNIST...\n";

    MedicalMNISTLoader loader(
        "../data/processed/train.csv",
        "../data/processed/test.csv"
    );

    Tensor<float, 2> X_train, Y_train;
    Tensor<float, 2> X_test, Y_test;

    auto start_load = chrono::high_resolution_clock::now();

    // Cargar datos de entrenamiento
    // PRUEBA RAPIDA: Solo 1000 imagenes (descomentar para pruebas rapidas)
    // size_t train_samples = loader.load_train(X_train, Y_train, true, 1000);
    // size_t test_samples = loader.load_test(X_test, Y_test, true, 500);

    // ENTRENAMIENTO COMPLETO: Todas las imagenes
    size_t train_samples = loader.load_train(X_train, Y_train, true, 0);
    size_t test_samples = loader.load_test(X_test, Y_test, true, 0);

    auto end_load = chrono::high_resolution_clock::now();
    auto load_time = chrono::duration_cast<chrono::seconds>(end_load - start_load).count();

    cout << "\nDatos cargados en " << load_time << " segundos.\n";
    cout << "Train samples: " << train_samples << "\n";
    cout << "Test samples: " << test_samples << "\n\n";

    // Mostrar estadisticas
    loader.print_stats(X_train, Y_train);

    // 2. CREAR LA RED NEURONAL
    cout << "\nPASO 2: Creando arquitectura de la red neuronal...\n";
    cout << "Arquitectura: 4096 -> 256 -> 128 -> 6\n";
    cout << "Activacion: ReLU (capas ocultas), sin activacion en salida\n\n";

    NeuralNetwork<float> network;

    // Capa 1: 4096 -> 256
    network.add_layer(make_unique<Dense<float>>(4096, 256));
    network.add_layer(make_unique<ReLU<float>>());

    // Capa 2: 256 -> 128
    network.add_layer(make_unique<Dense<float>>(256, 128));
    network.add_layer(make_unique<ReLU<float>>());

    // Capa de salida: 128 -> 6 (sin activacion, Softmax esta en el loss)
    network.add_layer(make_unique<Dense<float>>(128, 6));

    cout << "Red neuronal creada exitosamente!\n\n";

    // Intentar cargar pesos existentes
    bool model_loaded = network.load_model("../models/medical_mnist_model");

    if (model_loaded) {
        cout << "\n*** MODELO PRE-ENTRENADO ENCONTRADO ***\n";
        cout << "Opciones:\n";
        cout << "  1. Usar modelo guardado (saltar entrenamiento)\n";
        cout << "  2. Re-entrenar desde cero (borra modelo anterior)\n\n";
        cout << "Para re-entrenar, ejecuta: rm -rf ../models/medical_mnist_model\n";
        cout << "Usando modelo guardado...\n\n";
    }

    // Variables para tracking del tiempo de entrenamiento
    long train_time = 0;

    // Solo entrenar si NO se cargo un modelo
    if (!model_loaded) {
        // 3. CONFIGURAR HIPERPARAMETROS
        // ============================================
        // VALORES PARA PRUEBA RAPIDA (10-15 minutos con 1000 imagenes):
        // size_t epochs = 10;
        // size_t batch_size = 256;

        // VALORES PARA ENTRENAMIENTO COMPLETO (1-2 horas con 47,163 imagenes):
        size_t epochs = 10;  // Reducido - con early stopping
        size_t batch_size = 64;
        // ============================================

        float learning_rate = 0.002f;  // Ajustado segun MNIST GroupGPT (0.001 × 100/64 ≈ 0.0016)

        cout << "PASO 3: Hiperparametros de entrenamiento:\n";
        cout << "  - Epocas: " << epochs << "\n";
        cout << "  - Batch size: " << batch_size << "\n";
        cout << "  - Learning rate: " << learning_rate << "\n";
        cout << "  - Optimizer: Adam\n";
        cout << "  - Loss function: SoftmaxCrossEntropy\n\n";

        // 4. ENTRENAMIENTO
        cout << "PASO 4: Iniciando entrenamiento...\n";
        cout << "Esto puede tomar varios minutos...\n\n";

        auto start_train = chrono::high_resolution_clock::now();

        // Entrenar con Adam optimizer, SoftmaxCrossEntropyLoss y Early Stopping
        // Patience = 2 (detiene si el loss sube por 2 epocas consecutivas)
        network.train<SoftmaxCrossEntropyLoss, Adam>(
            X_train, Y_train,
            epochs, batch_size, learning_rate,
            "../models/medical_mnist_model",  // Auto-guarda mejor modelo
            2  // patience
        );

        auto end_train = chrono::high_resolution_clock::now();
        train_time = chrono::duration_cast<chrono::minutes>(end_train - start_train).count();

        cout << "\nEntrenamiento completado en " << train_time << " minutos!\n\n";
    }
    // 5. EVALUACIÓN EN DATOS DE ENTRENAMIENTO
    cout << "PASO 5: Evaluando el modelo...\n\n";

    cout << "=== EVALUACION EN TRAINING SET ===\n";
    auto train_logits = network.predict(X_train);
    auto train_predictions = apply_softmax(train_logits);
    float train_accuracy = calculate_accuracy(train_logits, Y_train);
    cout << "Training Accuracy: " << fixed << setprecision(2)
         << (train_accuracy * 100) << "%\n\n";

    // 6. EVALUACIÓN EN DATOS DE PRUEBA
    cout << "=== EVALUACION EN TEST SET ===\n";
    auto test_logits = network.predict(X_test);
    auto test_predictions = apply_softmax(test_logits);
    float test_accuracy = calculate_accuracy(test_logits, Y_test);
    cout << "Test Accuracy: " << fixed << setprecision(2)
         << (test_accuracy * 100) << "%\n\n";

    // 7. MOSTRAR EJEMPLOS DE PREDICCIONES
    cout << "=== EJEMPLOS DE PREDICCIONES ===\n";
    cout << "Mostrando primeras 10 predicciones del test set:\n\n";

    for (size_t i = 0; i < min(size_t(10), test_samples); ++i) {
        // Encontrar clase predicha (usar probabilidades, no logits)
        size_t pred_class = 0;
        float max_prob = test_predictions(i, 0);
        for (size_t j = 1; j < 6; ++j) {
            if (test_predictions(i, j) > max_prob) {
                max_prob = test_predictions(i, j);
                pred_class = j;
            }
        }

        // Encontrar clase verdadera
        size_t true_class = 0;
        for (size_t j = 0; j < 6; ++j) {
            if (Y_test(i, j) > 0.5f) {
                true_class = j;
                break;
            }
        }

        bool correct = (pred_class == true_class);

        cout << "Imagen " << (i + 1) << ": "
             << "Prediccion = " << loader.get_class_name(pred_class)
             << " (" << fixed << setprecision(1) << (max_prob * 100) << "%), "
             << "Real = " << loader.get_class_name(true_class)
             << " " << (correct ? "OK" : "FAIL") << "\n";
    }

    cout << "\n=== RESUMEN FINAL ===\n";
    cout << "Dataset: Medical MNIST (58,954 imagenes)\n";
    cout << "Arquitectura: 4096 -> 256 -> 128 -> 6\n";
    if (train_time > 0) {
        cout << "Tiempo de entrenamiento: " << train_time << " minutos\n";
    }
    cout << "Training Accuracy: " << fixed << setprecision(2)
         << (train_accuracy * 100) << "%\n";
    cout << "Test Accuracy: " << fixed << setprecision(2)
         << (test_accuracy * 100) << "%\n";

    return 0;
}
