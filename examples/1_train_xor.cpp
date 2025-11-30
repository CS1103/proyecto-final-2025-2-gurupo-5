#include "../src/utec/nn/neural_network.h"
#include "../src/utec/nn/nn_dense.h"
#include "../src/utec/nn/nn_activation.h"
#include "../src/utec/nn/nn_loss.h"
#include "../src/utec/nn/nn_optimizer.h"
#include "../src/utec/nn/tensor.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace utec::neural_network;

/**
 * EJEMPLO 1: Entrenamiento XOR
 *
 * Este es un ejemplo BÁSICO para verificar que la red neuronal funciona.
 * XOR es un problema clásico que NO es linealmente separable,
 * por lo que requiere una red neuronal con capas ocultas.
 *
 * Arquitectura: 2 → 4 → 4 → 1
 */

int main() {
    cout << "=== ENTRENAMIENTO DE RED NEURONAL - XOR ===\n\n";

    // 1. PREPARAR LOS DATOS
    // Datos de entrada XOR: [0,0], [0,1], [1,0], [1,1]
    Tensor<float, 2> X(4, 2);
    X(0,0) = 0; X(0,1) = 0;  // [0,0] -> 0
    X(1,0) = 0; X(1,1) = 1;  // [0,1] -> 1
    X(2,0) = 1; X(2,1) = 0;  // [1,0] -> 1
    X(3,0) = 1; X(3,1) = 1;  // [1,1] -> 0

    // Salidas esperadas
    Tensor<float, 2> Y(4, 1);
    Y(0,0) = 0;  // 0 XOR 0 = 0
    Y(1,0) = 1;  // 0 XOR 1 = 1
    Y(2,0) = 1;  // 1 XOR 0 = 1
    Y(3,0) = 0;  // 1 XOR 1 = 0

    cout << "Datos de entrenamiento creados:\n";
    for (int i = 0; i < 4; ++i) {
        cout << "Input: [" << X(i,0) << ", " << X(i,1)
             << "] -> Output: " << Y(i,0) << "\n";
    }
    cout << "\n";

    // 2. CREAR LA RED NEURONAL
    NeuralNetwork<float> network;

    // Arquitectura: 2 -> 4 -> 4 -> 1 (con activaciones ReLU)
    // Los pesos se inicializan automáticamente con Xavier initialization
    network.add_layer(make_unique<Dense<float>>(2, 4));
    network.add_layer(make_unique<ReLU<float>>());
    network.add_layer(make_unique<Dense<float>>(4, 4));
    network.add_layer(make_unique<ReLU<float>>());
    network.add_layer(make_unique<Dense<float>>(4, 1));
    network.add_layer(make_unique<Sigmoid<float>>());  // Sigmoid para salida [0,1]

    cout << "Red neuronal creada: 2->4->4->1\n\n";

    // 3. ENTRENAMIENTO
    cout << "=== ENTRENAMIENTO ===\n";
    size_t epochs = 1000;
    size_t batch_size = 4;  // Usar todos los datos en cada batch
    float learning_rate = 0.1f;

    cout << "Hiperparametros:\n";
    cout << "  - Epocas: " << epochs << "\n";
    cout << "  - Batch size: " << batch_size << "\n";
    cout << "  - Learning rate: " << learning_rate << "\n";
    cout << "  - Optimizer: SGD\n";
    cout << "  - Loss: MSE\n\n";

    cout << "Entrenando...\n";

    // Entrenar con SGD optimizer y MSELoss
    network.train<MSELoss, SGD>(X, Y, epochs, batch_size, learning_rate);

    cout << "Entrenamiento completado!\n\n";

    // 4. PREDICCIONES
    cout << "=== PREDICCIONES ===\n";
    auto predictions = network.predict(X);

    cout << "Input -> Esperado vs Prediccion\n";
    for (int i = 0; i < 4; ++i) {
        float pred = predictions(i, 0);
        float expected = Y(i, 0);
        cout << "[" << X(i,0) << "," << X(i,1) << "] -> "
             << expected << " vs " << fixed << setprecision(4) << pred;

        // Clasificacion binaria (threshold = 0.5)
        bool correct = (pred > 0.5f) == (expected > 0.5f);
        cout << " (" << (correct ? "OK" : "FAIL") << ")\n";
    }

    // 5. CALCULAR ACCURACY
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float pred = predictions(i, 0);
        float expected = Y(i, 0);
        if ((pred > 0.5f) == (expected > 0.5f)) {
            correct++;
        }
    }

    cout << "\nAccuracy: " << (correct * 100 / 4) << "%\n";

    return 0;
}
