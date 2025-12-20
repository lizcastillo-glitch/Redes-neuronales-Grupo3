# Reporte Técnico – Redes Neuronales para Análisis de Sentimientos

## 1. Introducción
Las redes neuronales artificiales constituyen uno de los pilares
fundamentales del aprendizaje profundo y han demostrado un desempeño
sobresaliente en problemas de clasificación con relaciones no lineales,
especialmente en tareas de procesamiento de lenguaje natural como el
análisis de sentimientos en texto no estructurado.

El presente reporte técnico documenta de manera detallada el diseño,
implementación, experimentación y evaluación de una red neuronal
feedforward desarrollada desde cero. Este trabajo forma parte del
proyecto de Inteligencia Artificial del Grupo 3 y tiene como objetivo
principal comprender en profundidad los mecanismos internos del
aprendizaje supervisado, más allá del uso directo de librerías de alto
nivel.

El desarrollo de este modelo sirve como base conceptual para justificar
la utilización posterior de arquitecturas más avanzadas, como los
modelos Transformer, empleados en la solución final del proyecto.

---

## 2. Descripción del Dataset
Se utilizó un conjunto de datos sintético de reseñas de productos,
compuesto por textos cortos y etiquetas categóricas de sentimiento,
definidas de la siguiente manera:

- 0: Negativo  
- 1: Neutro  
- 2: Positivo  

El dataset presenta un tamaño reducido y una distribución de clases
balanceada, lo que permitió evaluar de forma controlada el
comportamiento del modelo, minimizar sesgos por desbalance y facilitar
el análisis de los resultados obtenidos. Este tipo de dataset resulta
adecuado para fines académicos, ya que permite centrarse en la
comprensión del funcionamiento interno del modelo.

### Distribución de clases
<img width="546" height="402" alt="Distribución de clases" src="https://github.com/user-attachments/assets/3af8d655-d566-4b37-a8b3-4c1d32d503a3" />

---

## 3. Implementación de la Red Neuronal (Notebook 1)
La red neuronal fue implementada desde cero utilizando exclusivamente
NumPy, sin recurrir a frameworks de alto nivel como TensorFlow o PyTorch.
Esta decisión permitió comprender explícitamente cada etapa del proceso
de aprendizaje, desde el cálculo de activaciones hasta la propagación
del error.

### Características técnicas de la implementación
- Arquitectura feedforward con dos capas ocultas.
- Evaluación de funciones de activación: ReLU, Tanh y Sigmoid.
- Inicialización de pesos mediante los métodos Xavier y He.
- Función de pérdida: Cross-Entropy.
- Optimización mediante descenso por gradiente.
- Implementación explícita de forward propagation y backpropagation.

### Arquitectura de la red
La arquitectura de la red neuronal se definió de forma explícita en el
código del Notebook 1, a partir de la especificación del número de capas
y neuronas. La red corresponde a una arquitectura feedforward compuesta
por una capa de entrada, dos capas ocultas y una capa de salida con tres
neuronas, correspondientes a las clases de sentimiento.

La estructura conceptual de la red se deriva directamente de la
implementación en NumPy, donde se define el tamaño de cada capa y las
funciones de activación utilizadas, sin generar una visualización
gráfica automática de la arquitectura.

---

## 4. Entrenamiento y Convergencia
El entrenamiento del modelo se realizó durante múltiples épocas,
monitoreando de forma continua la evolución de la función de pérdida
(loss) y de la exactitud (accuracy) sobre el conjunto de entrenamiento.

El análisis de estas curvas permitió evaluar la estabilidad del proceso
de entrenamiento, detectar posibles problemas de subajuste o
sobreajuste, y verificar la correcta convergencia del modelo hacia una
solución óptima.

### Curva de entrenamiento
<img width="567" height="455" alt="Curva de pérdida" src="https://github.com/user-attachments/assets/d9b29551-162f-40f7-bcec-a8371d6bccbd" />

---

## 5. Experimentación y Comparación (Notebook 2)
Durante la fase de experimentación se evaluaron distintas
configuraciones de la red neuronal, variando de manera sistemática los
siguientes hiperparámetros:

- Número de neuronas por capa.
- Funciones de activación.
- Learning rate.
- Número de épocas de entrenamiento.

Adicionalmente, se entrenó un modelo baseline clásico basado en
Regresión Logística y representaciones TF-IDF, con el fin de establecer
un punto de referencia para la comparación del desempeño.

Los resultados obtenidos evidenciaron que la red neuronal presenta un
mejor desempeño frente al modelo baseline, especialmente en escenarios
con relaciones no lineales entre las características, validando el uso
de redes neuronales frente a modelos lineales tradicionales.

---

## 6. Evaluación y Análisis de Resultados (Notebook 3)
La evaluación final se realizó sobre un conjunto de prueba no visto,
utilizando métricas estándar de clasificación multiclase:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)  

Asimismo, se analizó la matriz de confusión para identificar posibles
patrones de error entre las distintas clases de sentimiento.

### Matriz de confusión
<img width="435" height="393" alt="Matriz de confusión" src="https://github.com/user-attachments/assets/cb4b33a0-4045-40a7-ba35-79f75a92d6b8" />

Los resultados muestran un desempeño sobresaliente en las clases
positiva y negativa. Aunque la clase neutra suele representar un mayor
desafío en problemas reales de análisis de sentimientos, en este
escenario el modelo logró una clasificación correcta sobre el conjunto
de prueba.

---

## 7. Análisis de Errores
Se realizó un análisis del desempeño del modelo sobre el conjunto de
prueba mediante la matriz de confusión y las métricas de evaluación
globales.

En este caso, no se identificaron errores de clasificación en el
conjunto de prueba, lo cual se evidencia en la matriz de confusión,
donde todos los valores se concentran en la diagonal principal. Este
resultado se debe principalmente al carácter sintético y balanceado
del dataset utilizado, así como a la simplicidad del escenario
evaluado.

Por este motivo, no se presenta una tabla de errores mal clasificados,
ya que no existen observaciones incorrectamente predichas en el
conjunto de prueba.

---

## 8. Conclusiones
El desarrollo de la red neuronal feedforward desde cero permitió una
comprensión profunda del funcionamiento interno del aprendizaje
supervisado y de los factores que influyen directamente en el desempeño
del modelo.

Los experimentos realizados validan que las redes neuronales superan a
modelos lineales clásicos en tareas de clasificación con relaciones no
lineales, justificando su uso como base conceptual para arquitecturas
más avanzadas, como los modelos Transformer, empleados en el proyecto
final de análisis de sentimientos.
