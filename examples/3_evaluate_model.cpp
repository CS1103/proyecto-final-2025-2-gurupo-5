#include "../src/utec/nn/neural_network.h"
#include "../src/utec/nn/nn_dense.h"
#include "../src/utec/nn/nn_activation.h"
#include "../include/utec/data/medical_mnist_loader.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>

using namespace std;
using namespace utec::neural_network;
using namespace utec::data;

/**
 * EJEMPLO 3: Evaluacion de Modelo Pre-entrenado
 *
 * Este script SOLO evalua un modelo ya entrenado.
 * No realiza entrenamiento, solo carga pesos y evalua.
 * 
 * Genera reporte en "evaluation_results.txt".
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
        // Encontrar clase predicha
        size_t pred_class = 0;
        float max_pred = predictions(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            if (predictions(i, j) > max_pred) {
                max_pred = predictions(i, j);
                pred_class = j;
            }
        }

        // Encontrar clase verdadera
        size_t true_class = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (labels(i, j) > 0.5f) {
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

// Funcion para calcular y reportar metricas detalladas
void report_metrics(const Tensor<float, 2>& logits,
                    const Tensor<float, 2>& labels,
                    const MedicalMNISTLoader& loader,
                    std::ostream& out) {
    size_t num_samples = logits.shape()[0];
    size_t num_classes = 6;

    // Matriz de confusion (6x6)
    vector<vector<int>> confusion(num_classes, vector<int>(num_classes, 0));

    for (size_t i = 0; i < num_samples; ++i) {
        // Encontrar clase predicha
        size_t pred_class = 0;
        float max_pred = logits(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            if (logits(i, j) > max_pred) {
                max_pred = logits(i, j);
                pred_class = j;
            }
        }

        // Encontrar clase verdadera
        size_t true_class = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (labels(i, j) > 0.5f) {
                true_class = j;
                break;
            }
        }

        confusion[true_class][pred_class]++;
    }

    // Imprimir matriz de confusion
    out << "\n=== MATRIZ DE CONFUSION ===\n\n";
    out << "Formato: Filas = Clase Real, Columnas = Clase Predicha\n\n";

    // Encabezado
    out << setw(12) << " ";
    for (size_t i = 0; i < num_classes; ++i) {
        out << setw(10) << loader.get_class_name(i).substr(0, 8);
    }
    out << "\n";

    // Filas
    for (size_t i = 0; i < num_classes; ++i) {
        out << setw(12) << loader.get_class_name(i).substr(0, 10);
        for (size_t j = 0; j < num_classes; ++j) {
            out << setw(10) << confusion[i][j];
        }
        out << "\n";
    }

    // Calcular precision y recall por clase
    out << "\n=== METRICAS POR CLASE ===\n\n";
    out << setw(12) << "Clase"
         << setw(12) << "Precision"
         << setw(12) << "Recall"
         << setw(12) << "F1-Score" << "\n";
    out << string(48, '-') << "\n";

    for (size_t i = 0; i < num_classes; ++i) {
        // True Positives
        int tp = confusion[i][i];

        // False Positives (suma de columna i, excluyendo diagonal)
        int fp = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (j != i) fp += confusion[j][i];
        }

        // False Negatives (suma de fila i, excluyendo diagonal)
        int fn = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            if (j != i) fn += confusion[i][j];
        }

        float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        float recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;

        out << setw(12) << loader.get_class_name(i).substr(0, 10)
             << setw(11) << fixed << setprecision(2) << (precision * 100) << "%"
             << setw(11) << (recall * 100) << "%"
             << setw(11) << (f1 * 100) << "%\n";
    }
}

int main() {
    cout << "=== EVALUACION DE MODELO PRE-ENTRENADO - MEDICAL MNIST ===\n\n";

    // 1. CARGAR DATOS DE PRUEBA
    cout << "PASO 1: Cargando datos de prueba...\n";

    MedicalMNISTLoader loader(
        "../data/processed/train.csv",
        "../data/processed/test.csv"
    );

    Tensor<float, 2> X_test, Y_test;

    auto start_load = chrono::high_resolution_clock::now();

    // Cargar TODO el dataset de prueba
    size_t test_samples = loader.load_test(X_test, Y_test, true, 0);

    auto end_load = chrono::high_resolution_clock::now();
    auto load_time = chrono::duration_cast<chrono::seconds>(end_load - start_load).count();

    cout << "\nDatos cargados en " << load_time << " segundos.\n";
    cout << "Test samples: " << test_samples << "\n\n";

    // 2. CREAR ARQUITECTURA DE LA RED
    cout << "PASO 2: Creando arquitectura de la red neuronal...\n";
    cout << "Arquitectura: 4096 -> 256 -> 128 -> 6\n\n";

    NeuralNetwork<float> network;

    // Capa 1: 4096 -> 256
    network.add_layer(make_unique<Dense<float>>(4096, 256));
    network.add_layer(make_unique<ReLU<float>>());

    // Capa 2: 256 -> 128
    network.add_layer(make_unique<Dense<float>>(256, 128));
    network.add_layer(make_unique<ReLU<float>>());

    // Capa de salida: 128 -> 6
    network.add_layer(make_unique<Dense<float>>(128, 6));

    cout << "Red neuronal creada.\n\n";

    // 3. CARGAR MODELO PRE-ENTRENADO
    cout << "PASO 3: Cargando pesos del modelo entrenado...\n";

    bool model_loaded = network.load_model("../models/medical_mnist_model");

    if (!model_loaded) {
        cerr << "ERROR: No se pudo cargar el modelo.\n";
        cerr << "Asegurate de haber entrenado el modelo primero ejecutando:\n";
        cerr << "  ./build/train_medical_mnist\n";
        return 1;
    }

    cout << "Modelo cargado exitosamente!\n\n";

    // 4. EVALUACION EN DATOS DE PRUEBA
    cout << "PASO 4: Evaluando modelo en test set completo...\n";
    cout << "Esto puede tomar 1-2 minutos...\n\n";

    auto start_eval = chrono::high_resolution_clock::now();

    auto test_logits = network.predict(X_test);
    auto test_predictions = apply_softmax(test_logits);

    auto end_eval = chrono::high_resolution_clock::now();
    
    // Calculo de tiempos con mayor precision (microsegundos)
    auto duration_us = chrono::duration_cast<chrono::microseconds>(end_eval - start_eval).count();
    double duration_sec = duration_us / 1000000.0;
    
    // Calculo de metricas de rendimiento
    double latency_ms = (double)duration_us / test_samples / 1000.0;
    double throughput = test_samples / duration_sec;

    float test_accuracy = calculate_accuracy(test_logits, Y_test);

    // Preparar archivo de salida
    ofstream outFile("../evaluation_results.txt");
    if (!outFile.is_open()) {
        cerr << "ADVERTENCIA: No se pudo crear el archivo 'evaluation_results.txt'.\n";
    }

    // Funcion helper para imprimir a consola y archivo
    auto print_dual = [&](ostream& console, ostream& file, const string& msg) {
        console << msg;
        if (file.good()) file << msg;
    };

    stringstream ss;
    ss << "Evaluacion completada en " << fixed << setprecision(3) << duration_sec << " segundos.\n\n";
    ss << "=== RESULTADOS FINALES ===\n";
    ss << "Test Accuracy: " << fixed << setprecision(2) << (test_accuracy * 100) << "%\n";
    ss << "Muestras evaluadas: " << test_samples << "\n";
    ss << "Correctas: " << (int)(test_accuracy * test_samples) << "\n";
    ss << "Incorrectas: " << (int)((1 - test_accuracy) * test_samples) << "\n\n";
    
    ss << "=== RENDIMIENTO (COMPUTACIONAL) ===\n";
    ss << "Tiempo Total Inferencia: " << duration_sec << " s\n";
    ss << "Latencia Promedio: " << latency_ms << " ms/imagen\n";
    ss << "Throughput: " << (int)throughput << " imagenes/segundo\n";
    
    print_dual(cout, outFile, ss.str());

    // 5. MATRIZ DE CONFUSION Y METRICAS DETALLADAS
    // Reportar a consola
    report_metrics(test_logits, Y_test, loader, cout);
    // Reportar a archivo
    if (outFile.good()) {
        report_metrics(test_logits, Y_test, loader, outFile);
        cout << "\n[INFO] Resultados detallados guardados en 'evaluation_results.txt'\n";
    }

    // 6. EJEMPLOS DE PREDICCIONES (Solo consola)
    cout << "\n=== EJEMPLOS DE PREDICCIONES ===\n";
    cout << "Mostrando 20 predicciones aleatorias:\n\n";

    for (size_t i = 0; i < min(size_t(20), test_samples); ++i) {
        size_t idx = (i * 593) % test_samples; // Pseudo-aleatorio

        // Encontrar clase predicha
        size_t pred_class = 0;
        float max_prob = test_predictions(idx, 0);
        for (size_t j = 1; j < 6; ++j) {
            if (test_predictions(idx, j) > max_prob) {
                max_prob = test_predictions(idx, j);
                pred_class = j;
            }
        }

        // Encontrar clase verdadera
        size_t true_class = 0;
        for (size_t j = 0; j < 6; ++j) {
            if (Y_test(idx, j) > 0.5f) {
                true_class = j;
                break;
            }
        }

        bool correct = (pred_class == true_class);

        cout << "Muestra " << setw(5) << idx << ": "
             << "Pred = " << setw(10) << loader.get_class_name(pred_class)
             << " (" << fixed << setprecision(1) << setw(5) << (max_prob * 100) << "%), "
             << "Real = " << setw(10) << loader.get_class_name(true_class)
             << " " << (correct ? "[OK]" : "[FAIL]") << "\n";
    }

    cout << "\n=== EVALUACION COMPLETADA ===\n";
    
    if (outFile.is_open()) outFile.close();

    return 0;
}
