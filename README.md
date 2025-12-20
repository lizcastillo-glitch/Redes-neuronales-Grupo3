# 🧠 Red Neuronal Feedforward desde Cero (NumPy) - Grupo 3

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Academic%20Project-green)

Implementación completa de una **Red Neuronal Artificial (Multi-Layer Perceptron)** construida puramente con **NumPy**, sin utilizar frameworks de Deep Learning (como TensorFlow o PyTorch) para el cálculo de gradientes.

El objetivo principal es desmitificar la "caja negra" del aprendizaje profundo, implementando manualmente el **Forward Propagation**, **Backpropagation** y el **Descenso de Gradiente**.

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Características Técnicas](#características-técnicas)
- [Instalación y Uso](#instalación-y-uso)
- [Metodología](#metodología)
- [Resultados y Comparativa](#resultados-y-comparativa)
- [Autores](#autores)

## 📖 Descripción del Proyecto
Este proyecto aborda un problema de **Clasificación de Sentimientos** (Positivo, Negativo, Neutro) utilizando un pipeline completo de Machine Learning:
1.  **Preprocesamiento:** Limpieza de texto y Aumentación de Datos (Data Augmentation).
2.  **Vectorización:** TF-IDF (Term Frequency - Inverse Document Frequency).
3.  **Modelado:** Red Neuronal Feedforward con arquitectura dinámica construida desde cero.
4.  **Optimización:** Backpropagation manual con optimizador SGD (Stochastic Gradient Descent) por mini-batches.

## 📂 Estructura del Repositorio

## ⚙️ Características Técnicas
La clase `NeuralNetwork` implementada en `src/neural_network.py` soporta:
* **Arquitectura Dinámica:** Definición arbitraria de capas ocultas (ej. `[Input, 128, 64, Output]`).
* **Funciones de Activación:**
    * `ReLU` (optimizada con inicialización **He** para evitar *vanishing gradients*).
    * `Tanh` y `Sigmoid` (con inicialización **Xavier/Glorot**).
    * `Softmax` (para la capa de salida multi-clase).
* **Optimizador:** Mini-Batch Gradient Descent.
* **Función de Costo:** Cross-Entropy Loss (con estabilidad numérica).

## 🚀 Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/lizcastillo-glitch/Redes-neuronales-Grupo3.git](https://github.com/lizcastillo-glitch/Redes-neuronales-Grupo3.git)
    cd Redes-neuronales-Grupo3
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn
    ```

3.  **Ejecución:**
    Se recomienda abrir los notebooks en Google Colab o Jupyter Lab siguiendo el orden numérico para replicar el proceso desde la construcción de la clase hasta la experimentación.

## 🧪 Metodología
Para validar el modelo, se realizaron experimentos comparativos variando:
* **Arquitecturas:** Profundidad (2+ capas ocultas) y ancho de capas.
* **Hiperparámetros:** Learning Rate (0.05, 0.01, 0.005) y Epochs.
* **Data Augmentation:** Se implementó una técnica de generación de texto sintético para robustecer el dataset original pequeño.
* **Baseline:** Se comparó el rendimiento contra una **Regresión Logística** de Scikit-Learn.

## 📊 Resultados y Comparativa

Se encontró que la combinación de **ReLU + Inicialización He** ofreció la convergencia más rápida y estable. A continuación, un resumen de los hallazgos clave:

| Modelo | Activación | Init | F1-Score | Observación |
| :--- | :--- | :--- | :--- | :--- |
| **Red Neuronal (Propia)** | **ReLU** | **He** | **1.00** | Mejor rendimiento y convergencia rápida. |
| Red Neuronal (Propia) | Tanh | Xavier | 1.00 | Buen rendimiento, ligeramente más lenta en converger. |
| Red Neuronal (Propia) | Sigmoid | Xavier | 0.16 - 0.72 | Sufrió severamente de *Vanishing Gradient*. |
| Baseline (LogReg) | - | - | 1.00 | Modelo lineal simple, efectivo para este dataset separable. |

> **Nota:** El Accuracy perfecto (1.0) se debe a la naturaleza sintética y altamente separable del dataset de prueba. En un entorno de producción real con datos ruidosos, se esperarían métricas más variadas.

### Visualizaciones
Las curvas de aprendizaje y matrices de confusión generadas durante los experimentos se encuentran almacenadas en la carpeta `/results`.

## 👥 Autores
**Grupo 3**

* Liz Eliana Castillo Zamora

* Pablo Mauricio Castro Hinostroza

* Erick Sebastián Rivas

* Ángel Israel Romero Medina

---
*Este proyecto fue desarrollado con fines académicos.*
