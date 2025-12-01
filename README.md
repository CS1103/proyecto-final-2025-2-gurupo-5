[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)

# Proyecto Final 2025-1: AI Neural Network

## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de un framework completo de redes neuronales en C++ para clasificación de imágenes médicas (Medical MNIST) y resolución de problemas lógicos complejos (XOR).

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
  - Shibuya Rengifo, Kenzo Akira – 209900003 (Implementación del modelo)
  - Pariansullca Ventrua Javier – 202420057 (Responsable de investigación teórica)
  - Muñuico Panti, Percy Eduardo – 209900004 (Pruebas y benchmarking)
  - Caballero Canchanya, Hector Junior – 202420043 (Documentación y demo)

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   - CMake 3.18+
   - Eigen 3.4
   - \[Otra librería opcional]

3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
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

* **Estructura de carpetas **:

  ```
  proyecto-final-2025-2-gurupo-5
  ├── cmake-build-debug
  ├── data
  │   └── processed
  │       ├── metadata.txt
  │       ├── test.csv
  │       └── train.csv
  │
  ├── examples
  │   ├── 1_train_xor.cpp
  │   ├── 2_train_medical_mnist.cpp
  │   └── 3_evaluate_model.cpp
  ├── include
  │   └── utec
  │       └── data
  │           └── medical_mnist_loader.h
  ├── scripts
  │   ├── convert_images_to_csv.cpp
  │   └── stb_image.h
  ├── src
  │   └── utec
  │       └── nn
  │           ├── neural_network.h
  │           ├── nn_activation.h
  │           ├── nn_dense.h
  │           ├── nn_interfaces.h
  │           ├── nn_loss.h
  │           ├── nn_optimizer.h
  │           └── tensor.h
  ├── .gitignore
  ├── CMakeLists.txt
  └── README.md


  ```

#### 2.2 Manual de uso y casos de prueba

- **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
- **Casos de prueba**:

  - Test unitario de capa densa.
  - Test de función de activación ReLU.
  - Test de convergencia en dataset de ejemplo.

> _Personalizar rutas, comandos y casos reales._

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

- **Métricas de ejemplo**:

  - Iteraciones: 1000 épocas.
  - Tiempo total de entrenamiento: 2m30s.
  - Precisión final: 92.5%.

- **Ventajas/Desventajas**:

  - - Código ligero y dependencias mínimas.
  - – Sin paralelización, rendimiento limitado.

- **Mejoras futuras**:

  - Uso de BLAS para multiplicaciones (Justificación).
  - Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                          | Miembro                 | Rol                                                             |
| ------------------------------ | ----------------------- | --------------------------------------------------------------- |
| Arquitectura del Sistema       | Hector Junior Caballero | Diseño de patrones, interfaces y estructura modular             |
| Implementación Modelo Neuronal | Kenzo Akira Shibuya     | Desarrollo de capas, optimizadores y algoritmo de entrenamiento |
| Investigación Teórica          | Javier Pariansullca     | Fundamentos matemáticos y estudio de backpropagation            |
| Testing y Benchmarking         | Percy Eduardo Muñuico   | Pruebas unitarias, métricas de rendimiento y validación         |
| Documentación y Casos de Uso   | Hector Junior Caballero | Elaboración de tutoriales, ejemplos y documentación técnica     |

---

### 6. Conclusiones

- **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
- **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
- **Aprendizajes**: Profundización en backpropagation y optimización.
- **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> _Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas._

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
