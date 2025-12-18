import numpy as np

class NeuralNetwork:
    """
    Red neuronal feedforward multicapa (NumPy) para clasificación multiclase.
    - Inicialización Xavier / He
    - Activaciones: relu, tanh, sigmoid
    - Salida: softmax
    - Loss: cross-entropy
    - Backprop manual + mini-batch GD
    """

    def __init__(self, layers, activation="relu", init="he", seed=42, momentum=0.0):
        """
        layers: lista, ej: [n_features, 256, 128, 3]
        activation: 'relu' | 'tanh' | 'sigmoid'
        init: 'xavier' | 'he'
        momentum: 0.0 a 0.9 (extra para “excelencia técnica”)
        """
        self.layers = layers
        self.activation_name = activation.lower()
        self.init = init.lower()
        self.momentum = float(momentum)

        np.random.seed(seed)

        self.W = []
        self.b = []
        self.vW = []
        self.vb = []

        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i+1]

            if self.init == "he":
                scale = np.sqrt(2.0 / fan_in)
            else:  # xavier
                scale = np.sqrt(1.0 / fan_in)

            Wi = np.random.randn(fan_out, fan_in) * scale
            bi = np.zeros((fan_out, 1))

            self.W.append(Wi)
            self.b.append(bi)

            self.vW.append(np.zeros_like(Wi))
            self.vb.append(np.zeros_like(bi))

    # -------- Activaciones --------
    def _act(self, Z):
        if self.activation_name == "relu":
            return np.maximum(0, Z)
        if self.activation_name == "tanh":
            return np.tanh(Z)
        if self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        raise ValueError("Activación no soportada")

    def _act_deriv(self, Z):
        if self.activation_name == "relu":
            return (Z > 0).astype(float)
        if self.activation_name == "tanh":
            A = np.tanh(Z)
            return 1 - A**2
        if self.activation_name == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
            return A * (1 - A)
        raise ValueError("Activación no soportada")

    # -------- Softmax --------
    def _softmax(self, Z):
        Z_shift = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(Z_shift)
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    # -------- Forward --------
    def forward(self, X):
        """
        X: (n_samples, n_features)
        returns: probs (n_samples, n_classes)
        """
        A = X.T  # (n_features, n_samples)

        self.cache = {"A0": A}
        for i in range(len(self.W) - 1):
            Z = self.W[i] @ A + self.b[i]
            A = self._act(Z)
            self.cache[f"Z{i+1}"] = Z
            self.cache[f"A{i+1}"] = A

        # Capa salida
        ZL = self.W[-1] @ A + self.b[-1]
        AL = self._softmax(ZL)
        self.cache[f"Z{len(self.W)}"] = ZL
        self.cache[f"A{len(self.W)}"] = AL

        return AL.T

    # -------- Loss --------
    def _one_hot(self, y, n_classes):
        Y = np.zeros((n_classes, y.shape[0]))
        Y[y, np.arange(y.shape[0])] = 1
        return Y

    def loss(self, y_true, probs, eps=1e-12):
        """
        y_true: (n_samples,) labels 0..C-1
        probs: (n_samples, C)
        """
        n = y_true.shape[0]
        p = np.clip(probs[np.arange(n), y_true], eps, 1 - eps)
        return -np.mean(np.log(p))

    # -------- Backprop --------
    def backward(self, X, y):
        """
        Gradientes con cross-entropy + softmax:
        dZL = AL - Y
        """
        n_samples = X.shape[0]
        n_classes = self.layers[-1]

        AL = self.cache[f"A{len(self.W)}"]  # (C, n)
        Y = self._one_hot(y, n_classes)     # (C, n)

        dZ = (AL - Y) / n_samples

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        A_prev = self.cache[f"A{len(self.W)-1}"]
        grads_W[-1] = dZ @ A_prev.T
        grads_b[-1] = np.sum(dZ, axis=1, keepdims=True)

        dA_prev = self.W[-1].T @ dZ

        for i in reversed(range(len(self.W) - 1)):
            Z = self.cache[f"Z{i+1}"]
            dZ = dA_prev * self._act_deriv(Z)

            A_prev = self.cache[f"A{i}"]
            grads_W[i] = dZ @ A_prev.T
            grads_b[i] = np.sum(dZ, axis=1, keepdims=True)

            if i != 0:
                dA_prev = self.W[i].T @ dZ

        return grads_W, grads_b

    # -------- Train --------
    def train(self, X, y, epochs=20, lr=0.01, batch_size=64, verbose=True):
        history = {"loss": []}

        n = X.shape[0]
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]

            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb = Xs[start:end]
                yb = ys[start:end]

                probs = self.forward(Xb)
                gW, gb = self.backward(Xb, yb)

                # Update con momentum opcional
                for i in range(len(self.W)):
                    self.vW[i] = self.momentum * self.vW[i] - lr * gW[i]
                    self.vb[i] = self.momentum * self.vb[i] - lr * gb[i]
                    self.W[i] += self.vW[i]
                    self.b[i] += self.vb[i]

            probs_full = self.forward(X)
            L = self.loss(y, probs_full)
            history["loss"].append(L)

            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"Epoch {ep}/{epochs} | loss={L:.4f}")

        return history

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.forward(X)
