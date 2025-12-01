### 1. Investigación teórica

- **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.

  1. ### Historia y evolución de las NNs.

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

  2. ### Principales arquitecturas: MLP, CNN, RNN.

     - **MLP (Multi-Layer Perceptron)**\
       \
       El Multi-Layer Perceptron representa la arquitectura fundamental de las redes neuronales artificiales. Consiste en una serie de capas completamente conectadas donde cada neurona de una capa se conecta con todas las neuronas de la capa siguiente. La potencia de esta arquitectura radica en su capacidad de aproximar cualquier función continua gracias al teorema de aproximación universal. Las capas ocultas con funciones de activación no lineales permiten modelar relaciones complejas en los datos. En nuestro proyecto, esta arquitectura forma la base de nuestro clasificador Medical MNIST, implementado a través de las clases en nn_dense.h y nn_activation.h, demostrando cómo una estructura aparentemente simple puede resolver problemas complejos de clasificación de imágenes médicas.

     - **CNN (Convolutional Neural Networks)**\
       \
       Las Redes Neuronales Convolucionales representan una especialización evolutiva para el procesamiento de datos con estructura espacial, particularmente imágenes. A diferencia de las MLP que usan conexiones densas, las CNN emplean operaciones de convolución que preservan las relaciones espaciales mediante el uso de kernels deslizantes. Esta arquitectura introduce dos conceptos fundamentales: la compartición de pesos, que reduce significativamente el número de parámetros, y la conectividad local, que permite detectar características independientemente de su posición en la imagen. Aunque nuestro proyecto actual no implementa CNN, la estructura modular de nuestro framework en nn_interfaces.h está diseñada para permitir esta extensión futura, manteniendo la compatibilidad con las operaciones de tensor existentes.
     - **RNN (Recurrent Neural Networks)**\
       \
       Las Redes Neuronales Recurrentes abordan una clase diferente de problemas: aquellos que involucran datos secuenciales y dependencias temporales. La característica distintiva de las RNN es su capacidad de mantener un estado interno que actúa como memoria, permitiendo que la salida en cada paso temporal dependa no solo de la entrada actual sino también de estados anteriores. Arquitecturas más avanzadas como LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit) resolvieron el problema del gradiente vanishing mediante mecanismos de puertas que controlan el flujo de información. En nuestro framework actual, aunque nos enfocamos en problemas feedforward, la interfaz ILayer proporciona la base sobre la cual podrían implementarse capas recurrentes en futuras iteraciones.

  3. ### Algoritmos de entrenamiento: backpropagation, optimizadores.

     - **Backpropagation**\
       \
       El algoritmo de backpropagation representa el pilar fundamental del entrenamiento de redes neuronales modernas. Su funcionamiento se basa en la aplicación sistemática de la regla de la cadena del cálculo diferencial para calcular eficientemente los gradientes de la función de pérdida con respecto a todos los parámetros de la red. El proceso se divide en dos fases principales: durante el forward pass, las entradas se propagan a través de la red calculando las salidas de cada capa, mientras que en el backward pass, los gradientes se calculan desde la salida hacia la entrada, permitiendo actualizar los pesos en la dirección que minimiza el error. En nuestra implementación en neural_network.h, este algoritmo no solo funciona correctamente sino que incorpora optimizaciones como el mini-batch training que equilibra la eficiencia computacional con la estabilidad de la convergencia.

     - **Optimizadores** \
       \
       Los algoritmos de optimización determinan cómo se actualizan los parámetros de la red una vez calculados los gradientes. El Descenso de Gradiente Estocástico (SGD) representa la aproximación más directa, ajustando los pesos en dirección opuesta al gradiente con una tasa de aprendizaje constante. Sin embargo, métodos más avanzados como Adam (Adaptive Moment Estimation) combinan el concepto de momentum, que acelera la convergencia en direcciones consistentes, con learning rates adaptativos por parámetro que ajustan automáticamente el tamaño de paso basándose en estadísticas de gradientes anteriores. En nuestro proyecto, la implementación de ambos optimizadores en nn_optimizer.h mediante el patrón Strategy permite experimentar con diferentes enfoques, demostrando cómo Adam generalmente ofrece convergencia más rápida y robusta para una amplia variedad de problemas.

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

### Patrones de Diseño

En el desarrollo de nuestro framework de red neuronal, hemos adoptado varios patrones de diseño que nos han permitido crear una arquitectura modular, mantenible y extensible. Cada patrón resuelve problemas específicos de diseño que son comunes en sistemas de machine learning.

- #### **Strategy Pattern:** Flexibilidad en los Algoritmos de Optimización

  Cuando diseñámos el sistema de entrenamiento, nos enfrentamos al desafío de querer experimentar con diferentes algoritmos de optimización sin tener que reescribir el código de entrenamiento cada vez. La solución llegó con el Strategy Pattern, que nos permite encapsular cada algoritmo de optimización detrás de una interfaz común.

  En la práctica, esto significa que nuestra interfaz IOptimizer en nn_interfaces.h define un contrato simple - el método update - que cualquier optimizador debe implementar. Así, cuando en neural_network.h necesitamos actualizar los pesos, simplemente llamamos a optimizer.update() sin preocuparnos por si estamos usando SGD, Adam o cualquier otro optimizador que implementemos en el futuro. Esta abstracción ha demostrado ser invaluable durante el desarrollo, permitiéndonos comparar el rendimiento de diferentes optimizadores con cambios mínimos en el código.

- #### **Factory Pattern:** Creación Flexible de Capas

  La verdadera potencia de una red neuronal reside en su capacidad de combinar diferentes tipos de capas. Para manejar esta diversidad de manera elegante, aplicamos el Factory Pattern a través de nuestra interfaz ILayer en nn_interfaces.h.

  Cada tipo de capa - ya sea una capa densa en nn_dense.h, una función de activación ReLU en nn_activation.h, o cualquier otra que añadamos después - se convierte en una fábrica concreta que sabe cómo realizar sus propias operaciones forward y backward. Esto transforma la construcción de la red neuronal en un proceso de composición: simplemente ensamblamos diferentes fábricas de capas como si fueran bloques de construcción. La belleza de este enfoque es que podemos añadir nuevos tipos de capas (como convolucionales o recurrentes en el futuro) sin modificar ni una línea del código existente que utiliza las capas.

- #### **Template Method Pattern:** El Esqueleto del Entrenamiento

  El algoritmo de entrenamiento de una red neuronal sigue siempre los mismos pasos fundamentales: forward pass, cálculo de pérdida, backward pass y actualización de parámetros. Sin embargo, los detalles específicos de cómo calcular la pérdida o cómo optimizar pueden variar. Aquí es donde el Template Method Pattern en neural_network.h brilla.

  En nuestro método train, definimos el esqueleto invariable del algoritmo de entrenamiento, pero dejamos "huecos" templateados para las partes que pueden cambiar: la función de pérdida y el optimizador. Esto nos da lo mejor de ambos mundos: la estabilidad de un algoritmo de entrenamiento probado y la flexibilidad de personalizar sus componentes clave. Es como tener una receta de cocina donde los pasos fundamentales siempre son los mismos, pero puedes elegir diferentes ingredientes según lo que quieras cocinar.

- #### **Composite Pattern:** Gestión Unificada de la Arquitectura

  Una red neuronal es, en esencia, una composición de capas que trabajan en conjunto. El Composite Pattern nos permite tratar esta composición de manera elegante. En neural_network.h, manejamos la red como un todo a través de un vector de unique_ptr<ILayer>, pero también podemos acceder y manipular cada capa individualmente cuando es necesario.

  Esto significa que operaciones como el forward pass se vuelven simples iteraciones sobre todas las capas, mientras que aún podemos invocar operaciones específicas en capas individuales cuando necesitamos, por ejemplo, actualizar solo los parámetros de las capas densas. Esta dualidad - tratar la red como un todo unificado y como una colección de partes individuales - es precisamente lo que el Composite Pattern maneja tan bien.
