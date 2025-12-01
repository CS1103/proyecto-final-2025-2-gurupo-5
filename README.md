[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

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

* **Tema**: Redes Neuronales en AI
* **Grupo**: `gurupo-5`
* **Integrantes**:

  * Caballero Canchanya, Hector Junior – 202420043 (Desarrollo de la arquitectura)
  * Shibuya Rengifo, Kenzo Akira – 209900003 (Implementación del modelo)
  * Pariansullca Ventrua Javier – 202420057 (Responsable de investigación teórica)
  * Muñuico Panti, Percy Eduardo – 209900004 (Pruebas y benchmarking)
  * Caballero Canchanya, Hector Junior – 202420043 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

  1. Historia y evolución de las NNs.
  
     El desarrollo de las redes neuronales artificiales comenzó en 1943 cuando Warren McCulloch y Walter Pitts propusieron el primer modelo matemático de neurona artificial en su artículo "A Logical Calculus of the Ideas Immanent in Nervous Activity". Inspirándose en el funcionamiento del cerebro humano, crearon un modelo binario que podía realizar operaciones lógicas básicas, estableciendo los fundamentos teóricos para todo el campo de la inteligencia artificial.\
  \
     En 1958, Frank Rosenblatt, un psicólogo estadounidense, desarrolló el perceptrón en el Cornell Aeronautical Laboratory. Este fue el primer modelo con capacidad real de aprendizaje automático, capaz de reconocer patrones simples y clasificar datos. Rosenblatt no solo propuso el algoritmo matemático, sino que construyó una implementación física llamada "Mark I Perceptron", demostrando que las máquinas podían aprender de la experiencia. Este avance generó un gran entusiasmo en la comunidad científica.\
  \
    Sin embargo, el optimismo inicial se vio frenado en 1969 cuando Marvin Minsky y Seymour Papert publicaron su libro "Perceptrons", donde demostraron matemáticamente las limitaciones fundamentales del perceptrón simple. Probaron que no podía resolver problemas no lineales como la función XOR, lo que llevó a un escepticismo generalizado y a una drástica reducción en la financiación para investigación en redes neuronales, período conocido como "el invierno de la IA".\
  \
     El renacimiento llegó en la década de 1980 con el desarrollo del algoritmo de backpropagation. Aunque Paul Werbos había propuesto la idea en su tesis doctoral de 1974, fue el artículo de Rumelhart, Hinton y Williams en Nature en 1986 el que popularizó el método. Este algoritmo permitió entrenar redes multicapa de manera eficiente al calcular gradientes mediante la regla de la cadena, resolviendo finalmente el problema XOR y abriendo la puerta a arquitecturas más complejas.\
  \
     La revolución moderna del deep learning comenzó en 2006 con Geoffrey Hinton y su equipo, quienes publicaron "A Fast Learning Algorithm for Deep Belief Nets", demostrando cómo entrenar redes profundas de manera eficiente. Esto coincidió con la disponibilidad de grandes conjuntos de datos y poder computacional, llevando a avances espectaculares en reconocimiento de imágenes, procesamiento de lenguaje natural y otras áreas.\
  \
     En nuestro proyecto, esta evolución histórica se refleja directamente en nuestra implementación. Desde las interfaces básicas de neurona en nn_interfaces.h hasta el algoritmo completo de backpropagation en neural_network.h, hemos construido un framework que encapsula décadas de investigación. La resolución del problema XOR en examples/1_train_xor.cpp demuestra cómo en nuestro proyecto superamos las limitaciones que alguna vez detuvieron el progreso en este campo.

  2. Principales arquitecturas: MLP, CNN, RNN.
     - **MLP (Multi-Layer Perceptron)**\
     \
     El Multi-Layer Perceptron representa la arquitectura fundamental de las redes neuronales artificiales. Consiste en una serie de capas completamente conectadas donde cada neurona de una capa se conecta con todas las neuronas de la capa siguiente. La potencia de esta arquitectura radica en su capacidad de aproximar cualquier función continua gracias al teorema de aproximación universal. Las capas ocultas con funciones de activación no lineales permiten modelar relaciones complejas en los datos. En nuestro proyecto, esta arquitectura forma la base de nuestro clasificador Medical MNIST, implementado a través de las clases en nn_dense.h y nn_activation.h, demostrando cómo una estructura aparentemente simple puede resolver problemas complejos de clasificación de imágenes médicas.
     
     - **CNN (Convolutional Neural Networks)**\
     \
     Las Redes Neuronales Convolucionales representan una especialización evolutiva para el procesamiento de datos con estructura espacial, particularmente imágenes. A diferencia de las MLP que usan conexiones densas, las CNN emplean operaciones de convolución que preservan las relaciones espaciales mediante el uso de kernels deslizantes. Esta arquitectura introduce dos conceptos fundamentales: la compartición de pesos, que reduce significativamente el número de parámetros, y la conectividad local, que permite detectar características independientemente de su posición en la imagen. Aunque nuestro proyecto actual no implementa CNN, la estructura modular de nuestro framework en nn_interfaces.h está diseñada para permitir esta extensión futura, manteniendo la compatibilidad con las operaciones de tensor existentes.
     - **RNN (Recurrent Neural Networks)**\
     \
     Las Redes Neuronales Recurrentes abordan una clase diferente de problemas: aquellos que involucran datos secuenciales y dependencias temporales. La característica distintiva de las RNN es su capacidad de mantener un estado interno que actúa como memoria, permitiendo que la salida en cada paso temporal dependa no solo de la entrada actual sino también de estados anteriores. Arquitecturas más avanzadas como LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit) resolvieron el problema del gradiente vanishing mediante mecanismos de puertas que controlan el flujo de información. En nuestro framework actual, aunque nos enfocamos en problemas feedforward, la interfaz ILayer proporciona la base sobre la cual podrían implementarse capas recurrentes en futuras iteraciones.

  3. Algoritmos de entrenamiento: backpropagation, optimizadores.
     - **Backpropagation**\
     \
     El algoritmo de backpropagation representa el pilar fundamental del entrenamiento de redes neuronales modernas. Su funcionamiento se basa en la aplicación sistemática de la regla de la cadena del cálculo diferencial para calcular eficientemente los gradientes de la función de pérdida con respecto a todos los parámetros de la red. El proceso se divide en dos fases principales: durante el forward pass, las entradas se propagan a través de la red calculando las salidas de cada capa, mientras que en el backward pass, los gradientes se calculan desde la salida hacia la entrada, permitiendo actualizar los pesos en la dirección que minimiza el error. En nuestra implementación en neural_network.h, este algoritmo no solo funciona correctamente sino que incorpora optimizaciones como el mini-batch training que equilibra la eficiencia computacional con la estabilidad de la convergencia.

     - **Optimizadores** \
     \
     Los algoritmos de optimización determinan cómo se actualizan los parámetros de la red una vez calculados los gradientes. El Descenso de Gradiente Estocástico (SGD) representa la aproximación más directa, ajustando los pesos en dirección opuesta al gradiente con una tasa de aprendizaje constante. Sin embargo, métodos más avanzados como Adam (Adaptive Moment Estimation) combinan el concepto de momentum, que acelera la convergencia en direcciones consistentes, con learning rates adaptativos por parámetro que ajustan automáticamente el tamaño de paso basándose en estadísticas de gradientes anteriores. En nuestro proyecto, la implementación de ambos optimizadores en nn_optimizer.h mediante el patrón Strategy permite experimentar con diferentes enfoques, demostrando cómo Adam generalmente ofrece convergencia más rápida y robusta para una amplia variedad de problemas.


---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

  **Patrones de Diseño**
  En el desarrollo de nuestro framework de red neuronal, hemos adoptado varios patrones de diseño que nos han permitido crear una arquitectura modular, mantenible y extensible. Cada patrón resuelve problemas específicos de diseño que son comunes en sistemas de machine learning.
  
  - **Strategy Pattern:** Flexibilidad en los Algoritmos de Optimización
  
    Cuando diseñámos el sistema de entrenamiento, nos enfrentamos al desafío de querer experimentar con diferentes algoritmos de optimización sin tener que reescribir el código de entrenamiento cada vez. La solución llegó con el Strategy Pattern, que nos permite encapsular cada algoritmo de optimización detrás de una interfaz común.
    
    En la práctica, esto significa que nuestra interfaz IOptimizer en nn_interfaces.h define un contrato simple - el método update - que cualquier optimizador debe implementar. Así, cuando en neural_network.h necesitamos actualizar los pesos, simplemente llamamos a optimizer.update() sin preocuparnos por si estamos usando SGD, Adam o cualquier otro optimizador que implementemos en el futuro. Esta abstracción ha demostrado ser invaluable durante el desarrollo, permitiéndonos comparar el rendimiento de diferentes optimizadores con cambios mínimos en el código.
  
  - **Factory Pattern:** Creación Flexible de Capas
  
    La verdadera potencia de una red neuronal reside en su capacidad de combinar diferentes tipos de capas. Para manejar esta diversidad de manera elegante, aplicamos el Factory Pattern a través de nuestra interfaz ILayer en nn_interfaces.h.
  
    Cada tipo de capa - ya sea una capa densa en nn_dense.h, una función de activación ReLU en nn_activation.h, o cualquier otra que añadamos después - se convierte en una fábrica concreta que sabe cómo realizar sus propias operaciones forward y backward. Esto transforma la construcción de la red neuronal en un proceso de composición: simplemente ensamblamos diferentes fábricas de capas como si fueran bloques de construcción. La belleza de este enfoque es que podemos añadir nuevos tipos de capas (como convolucionales o recurrentes en el futuro) sin modificar ni una línea del código existente que utiliza las capas.
  
  - **Template Method Pattern:** El Esqueleto del Entrenamiento
  
    El algoritmo de entrenamiento de una red neuronal sigue siempre los mismos pasos fundamentales: forward pass, cálculo de pérdida, backward pass y actualización de parámetros. Sin embargo, los detalles específicos de cómo calcular la pérdida o cómo optimizar pueden variar. Aquí es donde el Template Method Pattern en neural_network.h brilla.
  
    En nuestro método train, definimos el esqueleto invariable del algoritmo de entrenamiento, pero dejamos "huecos" templateados para las partes que pueden cambiar: la función de pérdida y el optimizador. Esto nos da lo mejor de ambos mundos: la estabilidad de un algoritmo de entrenamiento probado y la flexibilidad de personalizar sus componentes clave. Es como tener una receta de cocina donde los pasos fundamentales siempre son los mismos, pero puedes elegir diferentes ingredientes según lo que quieras cocinar.
  
  - **Composite Pattern:** Gestión Unificada de la Arquitectura
  
    Una red neuronal es, en esencia, una composición de capas que trabajan en conjunto. El Composite Pattern nos permite tratar esta composición de manera elegante. En neural_network.h, manejamos la red como un todo a través de un vector de unique_ptr<ILayer>, pero también podemos acceder y manipular cada capa individualmente cuando es necesario. 
    
    Esto significa que operaciones como el forward pass se vuelven simples iteraciones sobre todas las capas, mientras que aún podemos invocar operaciones específicas en capas individuales cuando necesitamos, por ejemplo, actualizar solo los parámetros de las capas densas. Esta dualidad - tratar la red como un todo unificado y como una colección de partes individuales - es precisamente lo que el Composite Pattern maneja tan bien.
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

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

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

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro                     | Rol                                                       |
| ------------------------- | --------------------------- | --------------------------------------------------------- |
| Arquitectura del Sistema  | Hector Junior Caballero     | Diseño de patrones, interfaces y estructura modular       |
| Implementación Modelo Neuronal | Kenzo Akira Shibuya    | Desarrollo de capas, optimizadores y algoritmo de entrenamiento |
| Investigación Teórica     | Javier Pariansullca         | Fundamentos matemáticos y estudio de backpropagation      |
| Testing y Benchmarking    | Percy Eduardo Muñuico       | Pruebas unitarias, métricas de rendimiento y validación   |
| Documentación y Casos de Uso | Hector Junior Caballero    | Elaboración de tutoriales, ejemplos y documentación técnica |



---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

[1] W. S. McCulloch and W. Pitts, "A logical calculus of the ideas immanent in nervous activity,"
The Bulletin of Mathematical Biophysics, vol. 5, pp. 115–133, 1943.
 https://doi.org/10.1007/BF02478259

[2] F. Rosenblatt, "The perceptron: A probabilistic model for information storage and organization in the brain,"
Psychological Review, vol. 65, no. 6, pp. 386–408, 1958.
 https://doi.org/10.1037/h0042519

[3] M. Minsky and S. Papert, *Perceptrons: An Introduction to Computational Geometry*,
Cambridge, MA, USA: MIT Press, 1969.
https://rodsmith.nz/wp-content/uploads/Minsky-and-Papert-Perceptrons.pdf

[4] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors,"
Nature, vol. 323, pp. 533–536, 1986.

https://doi.org/10.1038/323533a0

[5] G. E. Hinton, S. Osindero, and Y. W. Teh, "A fast learning algorithm for deep belief nets,"
Neural Computation, vol. 18, no. 7, pp. 1527–1554, 2006.
https://doi.org/10.1162/neco.2006.18.7.1527

[6] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*.
Cambridge, MA, USA: MIT Press, 2016.
https://www.deeplearningbook.org/

[7] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization,"
arXiv preprint arXiv:1412.6980, 2014.
https://arxiv.org/abs/1412.6980

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
