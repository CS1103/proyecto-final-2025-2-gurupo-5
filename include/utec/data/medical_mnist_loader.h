#ifndef MEDICAL_MNIST_LOADER_H
#define MEDICAL_MNIST_LOADER_H

#include "../../../src/utec/nn/tensor.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

using namespace std;

namespace utec::data {

// Usar el alias de Tensor desde neural_network
using utec::neural_network::Tensor;

/**
 * Clase para cargar datos Medical MNIST desde archivos CSV
 *
 * Formato CSV esperado: label,p0,p1,p2,...,p4095
 * - label: Clase (0-5)
 * - p0...p4095: Pixeles en escala de grises (0-255)
 *
 * Normalizaciones aplicadas:
 * - Pixeles: [0-255] -> [0.0-1.0]
 * - Labels: one-hot encoding (6 clases)
 */
class MedicalMNISTLoader {
private:
    string train_path;
    string test_path;

    // Nombres de las clases
    const vector<string> CLASS_NAMES = {
        "AbdomenCT",   // 0
        "BreastMRI",   // 1
        "ChestCT",     // 2
        "CXR",         // 3
        "Hand",        // 4
        "HeadCT"       // 5
    };

public:
    /**
     * Constructor
     * @param train_csv_path Ruta al archivo train.csv
     * @param test_csv_path Ruta al archivo test.csv
     */
    MedicalMNISTLoader(const string& train_csv_path, const string& test_csv_path)
        : train_path(train_csv_path), test_path(test_csv_path) {}

    /**
     * Carga datos desde un archivo CSV
     * @param filepath Ruta al archivo CSV
     * @param images Tensor de salida para imágenes (N, 4096)
     * @param labels Tensor de salida para labels (N, 6) - one-hot encoded
     * @param normalize Si es true, normaliza píxeles a [0.0-1.0]
     * @return Número de imágenes cargadas
     */
    size_t load_csv(const string& filepath,
                    Tensor<float, 2>& images,
                    Tensor<float, 2>& labels,
                    bool normalize = true,
                    size_t max_samples = 0) {  // 0 = cargar todas

        ifstream file(filepath);
        if (!file.is_open()) {
            throw runtime_error("No se pudo abrir el archivo: " + filepath);
        }

        // Primera pasada: contar líneas
        size_t num_samples = 0;
        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                num_samples++;
            }
        }

        // Limitar si se especifica max_samples
        if (max_samples > 0 && num_samples > max_samples) {
            num_samples = max_samples;
        }

        cout << "Encontradas " << num_samples << " muestras en " << filepath << endl;

        // Reiniciar archivo
        file.clear();
        file.seekg(0);

        // Crear tensores con el tamano correcto
        images = Tensor<float, 2>(num_samples, 4096);  // 64*64 = 4096 pixeles
        labels = Tensor<float, 2>(num_samples, 6);     // 6 clases

        // Inicializar labels a cero (para one-hot encoding)
        labels.fill(0.0f);

        // Segunda pasada: cargar datos
        size_t row = 0;
        while (getline(file, line)) {
            if (line.empty()) continue;

            stringstream ss(line);
            string value;

            // Leer label (primera columna)
            getline(ss, value, ',');
            int label = stoi(value);

            // Validar label
            if (label < 0 || label >= 6) {
                throw runtime_error("Label inválido: " + to_string(label));
            }

            // One-hot encoding: labels(row, label) = 1.0
            labels(row, label) = 1.0f;

            // Leer píxeles (4096 columnas)
            for (size_t col = 0; col < 4096; ++col) {
                if (!getline(ss, value, ',')) {
                    throw runtime_error("Formato CSV inválido en fila " + to_string(row));
                }

                float pixel_value = stof(value);

                // Normalizar si es necesario
                if (normalize) {
                    pixel_value /= 255.0f;  // [0-255] -> [0.0-1.0]
                }

                images(row, col) = pixel_value;
            }

            row++;

            // Mostrar progreso cada 5000 imagenes
            if (row % 5000 == 0) {
                cout << "  Cargadas " << row << "/" << num_samples << " imagenes..." << endl;
            }

            // Si alcanzamos el limite, salir
            if (row >= num_samples) {
                break;
            }
        }

        file.close();
        cout << "Carga completada: " << row << " imagenes procesadas." << endl;

        return row;
    }

    /**
     * Carga el dataset de entrenamiento
     * @param images Tensor de salida para imágenes (N, 4096)
     * @param labels Tensor de salida para labels (N, 6)
     * @param normalize Si es true, normaliza píxeles a [0.0-1.0]
     * @return Número de imágenes cargadas
     */
    size_t load_train(Tensor<float, 2>& images,
                      Tensor<float, 2>& labels,
                      bool normalize = true,
                      size_t max_samples = 0) {
        cout << "\n=== CARGANDO DATASET DE ENTRENAMIENTO ===" << endl;
        return load_csv(train_path, images, labels, normalize, max_samples);
    }

    /**
     * Carga el dataset de prueba
     * @param images Tensor de salida para imágenes (N, 4096)
     * @param labels Tensor de salida para labels (N, 6)
     * @param normalize Si es true, normaliza píxeles a [0.0-1.0]
     * @return Número de imágenes cargadas
     */
    size_t load_test(Tensor<float, 2>& images,
                     Tensor<float, 2>& labels,
                     bool normalize = true,
                     size_t max_samples = 0) {
        cout << "\n=== CARGANDO DATASET DE PRUEBA ===" << endl;
        return load_csv(test_path, images, labels, normalize, max_samples);
    }

    /**
     * Obtiene el nombre de una clase dado su índice
     * @param class_idx Índice de la clase (0-5)
     * @return Nombre de la clase
     */
    string get_class_name(int class_idx) const {
        if (class_idx < 0 || class_idx >= static_cast<int>(CLASS_NAMES.size())) {
            return "Unknown";
        }
        return CLASS_NAMES[class_idx];
    }

    /**
     * Muestra estadisticas del dataset
     * @param images Tensor de imagenes
     * @param labels Tensor de labels (one-hot)
     */
    void print_stats(const Tensor<float, 2>& images,
                     const Tensor<float, 2>& labels) const {
        size_t num_samples = images.shape()[0];

        cout << "\n=== ESTADISTICAS DEL DATASET ===" << endl;
        cout << "Total de muestras: " << num_samples << endl;
        cout << "Dimension de imagen: " << images.shape()[1] << " pixeles (64x64)" << endl;
        cout << "Numero de clases: " << labels.shape()[1] << endl;

        // Contar muestras por clase
        vector<int> class_counts(6, 0);
        for (size_t i = 0; i < num_samples; ++i) {
            // Encontrar la clase (indice con valor 1.0 en one-hot)
            for (size_t j = 0; j < 6; ++j) {
                if (labels(i, j) > 0.5f) {  // Es 1.0
                    class_counts[j]++;
                    break;
                }
            }
        }

        cout << "\nDistribucion por clase:" << endl;
        for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
            cout << "  " << i << " (" << CLASS_NAMES[i] << "): "
                 << class_counts[i] << " muestras" << endl;
        }
    }
};

} // namespace utec::data

#endif // MEDICAL_MNIST_LOADER_H
