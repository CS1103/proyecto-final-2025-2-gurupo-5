[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)

# Proyecto Final 2025-1: AI Neural Network

## **CS2013 Programación III** · Informe Final

#### --------------------------------------------------------------
### [Ver Video Explicativo del Modelo](https://drive.google.com/file/d/15bL10KMTnmroiDwyxgu8lXwRkyecNhgs/view?usp=sharing)
#### --------------------------------------------------------------

### **Descripción**

Implementación de un framework completo de redes neuronales en C++ para clasificación de imágenes médicas (Medical MNIST).

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)

---

### Datos generales

- **Tema**: Redes Neuronales en AI
- **Grupo**: `gurupo-5`
- **Integrantes**:

    - Caballero Canchanya, Hector Junior – 202420043 (Desarrollo de la arquitectura)
    - Shibuya Rengifo, Kenzo Akira – 209900003 (Métricas y resultados)
    - Pariansullca Ventrua Javier – 202420057 (Responsable de investigación teórica)


---

### Requisitos e instalación

1. **Compilador**: GCC 11+ o MSVC (C++20)
2. **Dependencias**:

    - CMake 3.30+
    - C++20 standard library (filesystem, chrono)
    - stb_image.h (incluido en `scripts/`)

3. **Instalación**:

   ```bash
   git clone git@github.com:CS1103/proyecto-final-2025-2-gurupo-5.git
   cd proyecto-final-2025-2-gurupo-5
   mkdir cmake-build-debug && cd cmake-build-debug
   cmake ..
   cmake --build .
   ```

---

### 1. Investigación teórica

- **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.  
  [Ir a Investigación teórica](docs/documentation.md#1-investigación-teórica)

1. [Historia y evolución de las NNs](docs/documentation.md#historia-y-evolución-de-las-nns)

2. [Principales arquitecturas: MLP, CNN, RNN](docs/documentation.md#principales-arquitecturas-mlp-cnn-rnn)

3. [Algoritmos de entrenamiento: backpropagation, optimizadores](docs/documentation.md#algoritmos-de-entrenamiento-backpropagation-optimizadores)

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

**Patrones de Diseño**

- **Strategy Pattern:** Flexibilidad en los Algoritmos de Optimización
- **Factory Pattern:** Creación Flexible de Capas
- **Template Method Pattern:** El Esqueleto del Entrenamiento
- **Composite Pattern:** Gestión Unificada de la Arquitectura  
  [Ir a Patrones de Diseño](docs/documentation.md#patrones-de-diseño)

**Estructura de carpetas**:

  ```
  proyecto-final-2025-2-gurupo-5
  ├── cmake-build-debug/        # Archivos de build generados por CMake
  ├── data/
  │   ├── medical_mnist/         # Imágenes JPEG originales (no incluidas)
  │   └── processed/             # Datos CSV procesados (no incluidos)
  │       ├── train.csv          # 47,163 muestras de entrenamiento
  │       └── test.csv           # 11,791 muestras de prueba
  ├── examples/
  │   ├── 1_train_xor.cpp        # Validación básica (problema XOR)
  │   ├── 2_train_medical_mnist.cpp  # Entrenamiento completo
  │   ├── 3_evaluate_model.cpp   # Evaluación con métricas
  │   └── 4_predict_single.cpp   # Predicción de imagen individual
  ├── include/utec/data/
  │   └── medical_mnist_loader.h # Carga y preprocesamiento de datos
  ├── models/                    # Pesos entrenados (generado tras train)
  │   └── medical_mnist_model/
  ├── scripts/
  │   ├── convert_images_to_csv.cpp  # Conversor de JPEG a CSV
  │   └── stb_image.h            # Librería de carga de imágenes
  ├── src/utec/nn/
  │   ├── neural_network.h       # Clase principal de la red
  │   ├── nn_activation.h        # Funciones de activación (ReLU, Sigmoid)
  │   ├── nn_dense.h             # Capa densa (fully connected)
  │   ├── nn_interfaces.h        # Interfaces abstractas
  │   ├── nn_loss.h              # Funciones de pérdida (MSE, BCE, Softmax-CE)
  │   ├── nn_optimizer.h         # Optimizadores (SGD, Adam)
  │   └── tensor.h               # Clase tensor N-dimensional
  ├── tests/
  │   ├── test_tensor.cpp        # Tests de operaciones de tensores
  │   ├── test_neural_network.cpp    # Tests de componentes NN
  │   └── test_data_loader.cpp   # Tests de carga de datos
  ├── .gitignore
  ├── CMakeLists.txt
  ├── evaluation_results.txt     # Resultados de evaluación (generado)
  └── README.md
  ```

#### 2.2 Manual de uso y casos de prueba

**Paso 1: Obtener el Dataset**

Las imágenes originales (58,954 JPEGs, ~2.5 GB) y archivos CSV procesados (train.csv: 621 MB, test.csv: 156 MB) **no fueron incluidos en el repositorio debido a su peso**.

**Opción A: Descargar CSVs procesados (recomendado)**
```bash
# Descargar desde Google Drive del equipo:
# https://drive.google.com/drive/folders/1K1vMnTFw7ZOtMcHWdPYIAI5JnA2KG9l6?usp=sharing
# Colocar archivos en: data/processed/train.csv y data/processed/test.csv
```

**Opción B: Convertir imágenes manualmente desde Kaggle**
```bash
# 1. Descargar dataset Medical MNIST en formato JPEG desde:
#    https://www.kaggle.com/datasets/rishantenis/medical-mnist/data
# 2. Extraer y colocar en data/medical_mnist/
#    (estructura: AbdomenCT/, BreastMRI/, ChestCT/, CXR/, Hand/, HeadCT/)
# 3. Ejecutar conversor:
./cmake-build-debug/convert_images
# Esto generará train.csv y test.csv en data/processed/
```

**Paso 2: Entrenar el Modelo**
```bash
# Desde la raíz del proyecto
./cmake-build-debug/train_medical_mnist

# Genera: models/medical_mnist_model/ con pesos entrenados
# Duración: ~60-180 minutos (depende del hardware)
```

**Paso 3: Evaluar el Modelo**
```bash
./cmake-build-debug/evaluate_model

# Genera: evaluation_results.txt con métricas detalladas
```

**Paso 4 (Opcional): Ejecutar Tests Unitarios**
```bash
# Compilar con tests habilitados
cd cmake-build-debug
cmake .. -DBUILD_TESTS=ON
cmake --build .

# Ejecutar tests individuales
./test_tensor              # Tests de operaciones de tensores
./test_neural_network      # Tests de red neuronal (XOR convergence)
./test_data_loader         # Tests de carga de datos y preprocesamiento
```

**Ejemplos y Tests Implementados:**
- `1_train_xor.cpp`: Validación básica (problema XOR, 100% accuracy)
- `2_train_medical_mnist.cpp`: Entrenamiento completo con early stopping
- `3_evaluate_model.cpp`: Evaluación con matriz de confusión y métricas por clase
- `4_predict_single.cpp`: Predicción de imagen individual
- `test_tensor.cpp`: Suite de tests para operaciones de tensores (7 tests)
- `test_neural_network.cpp`: Tests de componentes NN (7 tests, incluye XOR)
- `test_data_loader.cpp`: Tests de carga y preprocesamiento (4 tests)

---

### 3. Ejecución

**Flujo de Trabajo Completo:**

1. **Preparar datos de entrenamiento**:
    - Descargar CSVs desde [Google Drive](https://drive.google.com/drive/folders/1K1vMnTFw7ZOtMcHWdPYIAI5JnA2KG9l6?usp=sharing)
    - O convertir imágenes desde [Kaggle Medical MNIST](https://www.kaggle.com/datasets/rishantenis/medical-mnist/data)
    - Verificar estructura: `data/processed/train.csv` y `data/processed/test.csv`

2. **Entrenar el modelo**:
   ```bash
   ./cmake-build-debug/train_medical_mnist
   ```
    - Arquitectura: 4096 → 256 → 128 → 6
    - Early stopping con patience=2
    - Guarda automáticamente el mejor modelo en `models/medical_mnist_model/`

3. **Evaluar resultados**:
   ```bash
   ./cmake-build-debug/evaluate_model
   ```
    - Carga modelo pre-entrenado
    - Genera `evaluation_results.txt` con:
        - Accuracy global
        - Matriz de confusión 6×6
        - Precision, Recall, F1-Score por clase
        - Métricas de rendimiento (latencia, throughput)

---

### 4. Análisis del rendimiento

#### 4.1 Resultados Generales

- **Dataset**: Medical MNIST (58,954 imágenes).
- **Arquitectura**: MLP (4096 -> 256 -> 128 -> 6).
- **Tiempo de entrenamiento**: 184 minutos.
- **Test Accuracy**: **86.12%** (11,791 muestras evaluadas).
- **Muestras Correctas**: 10,154.
- **Muestras Incorrectas**: 1,637.
- **Eficiencia Computacional**:
    - **Latencia promedio**: 2.50 ms/imagen.
    - **Throughput**: 400 imágenes/segundo.

#### 4.2 Métricas Detalladas

Se implementó un sistema de evaluación que genera automáticamente métricas clave, exportadas a `evaluation_results.txt`.

**Matriz de Confusión:**

```text
Formato: Filas = Clase Real, Columnas = Clase Predicha

              AbdomenC  BreastMR   ChestCT       CXR      Hand    HeadCT
   AbdomenCT       482         0      1444         0         0         1
   BreastMRI         0      1799         0         0         0         0
     ChestCT         0         0      1985         0         0         0
         CXR         0         0         5      2055         2         2
        Hand         1         2        18        29      1932        17
      HeadCT         0         0        40         0        76      1901
```

**Métricas por Clase:**

```text
       Clase   Precision      Recall    F1-Score
------------------------------------------------
   AbdomenCT      99.79%      25.01%      40.00%
   BreastMRI      99.89%     100.00%      99.94%
     ChestCT      56.84%     100.00%      72.48%
         CXR      98.61%      99.56%      99.08%
        Hand      96.12%      96.65%      96.38%
      HeadCT      98.96%      94.25%      96.55%
```

#### 4.3 Discusión de Resultados

El modelo demuestra un rendimiento excelente en la mayoría de las clases, destacando particularmente en **BreastMRI** (99.94% F1) y **CXR** (99.08% F1). Sin embargo, se observa una confusión significativa entre **AbdomenCT** y **ChestCT**:

1.  **Caso AbdomenCT**: Tiene un Recall muy bajo (25.01%), lo que significa que el 75% de las imágenes de AbdomenCT fueron clasificadas erróneamente, casi exclusivamente como **ChestCT** (1444 falsos positivos).
2.  **Causa probable**: La similitud visual entre tomografías de abdomen y pecho en escala de grises de 64x64 es alta, y una red MLP simple carece de la capacidad de extracción de características espaciales finas que tendría una red convolucional (CNN).
3.  **Eficiencia**: Con una latencia de 2.50ms, el modelo es extremadamente rápido, lo que lo hace viable para procesamiento en tiempo real en hardware modesto.

---

### 5. Trabajo en equipo

| Tarea                          | Miembro                 | Rol                                                             |
| ------------------------------ |-------------------------| --------------------------------------------------------------- |
| Arquitectura del Sistema       | Hector Junior Caballero | Diseño de patrones, interfaces y estructura modular             |
| Implementación Modelo Neuronal | Kenzo Akira Shibuya     | Desarrollo de capas, optimizadores y algoritmo de entrenamiento |
| Investigación Teórica          | Javier Pariansullca     | Fundamentos matemáticos y estudio de backpropagation            |
| Testing y Benchmarking         | Kenzo Akira Shibuya     | Pruebas unitarias, métricas de rendimiento y validación         |
| Documentación y Casos de Uso   | Hector Junior Caballero | Elaboración de tutoriales, ejemplos y documentación técnica     |

---

### 6. Conclusiones

- **Logros**:
    - Implementación completa de red neuronal desde cero en C++ sin librerías externas
    - 86.12% de accuracy en Medical MNIST (58,954 imágenes médicas)
    - Sistema modular y extensible con patrones de diseño (Strategy, Factory, Composite)
    - Early stopping funcional que previene overfitting

- **Evaluación**:
    - Rendimiento excelente en 4 de 6 clases (F1 > 96%)
    - Confusión significativa entre AbdomenCT y ChestCT (requiere CNN para mejorar)
    - Latencia de 2.50ms/imagen permite procesamiento en tiempo real

- **Aprendizajes**:
    - Comprensión profunda de backpropagation y cálculo de gradientes
    - Importancia de normalización de gradientes y learning rate tuning
    - Diferencias entre optimizadores (SGD vs Adam)
    - Técnicas de early stopping para evitar overfitting

- **Recomendaciones**:
    - Implementar arquitectura CNN para mejorar clasificación de tomografías
    - Optimizar operaciones de matriz con SIMD o librerías BLAS
    - Agregar data augmentation para mejorar generalización
    - Implementar dropout y batch normalization

---

### 7. Bibliografía

[1] W. S. McCulloch and W. Pitts, "A logical calculus of the ideas immanent in nervous activity,"
The Bulletin of Mathematical Biophysics, vol. 5, pp. 115–133, 1943.
https://doi.org/10.1007/BF02478259

[2] F. Rosenblatt, "The perceptron: A probabilistic model for information storage and organization in the brain,"
Psychological Review, vol. 65, no. 6, pp. 386–408, 1958.
https://doi.org/10.1037/h0042519

[3] M. Minsky and S. Papert, _Perceptrons: An Introduction to Computational Geometry_,
Cambridge, MA, USA: MIT Press, 1969.
https://rodsmith.nz/wp-content/uploads/Minsky-and-Papert-Perceptrons.pdf

[4] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors,"
Nature, vol. 323, pp. 533–536, 1986.

https://doi.org/10.1038/323533a0

[5] G. E. Hinton, S. Osindero, and Y. W. Teh, "A fast learning algorithm for deep belief nets,"
Neural Computation, vol. 18, no. 7, pp. 1527–1554, 2006.
https://doi.org/10.1162/neco.2006.18.7.1527

[6] I. Goodfellow, Y. Bengio, and A. Courville, _Deep Learning_.
Cambridge, MA, USA: MIT Press, 2016.
https://www.deeplearningbook.org/

[7] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization,"
arXiv preprint arXiv:1412.6980, 2014.
https://arxiv.org/abs/1412.6980

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
