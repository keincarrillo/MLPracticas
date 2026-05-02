# perceptron_gradiente.py
# Perceptrón con función de activación sigmoide y gradiente descendente.
#
# A diferencia de la regla Delta, aquí la función de activación es DIFERENCIABLE,
# lo que permite calcular el gradiente exacto de la función de pérdida respecto
# a los pesos y actualizarlos en la dirección que reduce el error.
#
# Función de pérdida: Binary Cross-Entropy (BCE)
#   L = -mean( y*log(p) + (1-y)*log(1-p) )
# Su gradiente respecto a los pesos es: X.T @ (p - y) / n
#
# Modos disponibles:
#   "classic"     → gradiente calculado sobre todo el conjunto (batch GD)
#   "stochastic"  → gradiente calculado ejemplo a ejemplo (SGD)

import numpy as np
import time
from sklearn.metrics import accuracy_score
from utilidades import set_seed, sigmoid

class PerceptronGradient:

    # __init__(lr, epochs, mode, seed) -> None
    # Inicializa hiperparámetros y variables de seguimiento.
    # lr      : tasa de aprendizaje para el descenso de gradiente
    # epochs  : número máximo de épocas de entrenamiento
    # mode    : "classic" (batch) o "stochastic" (online SGD)
    # seed    : semilla para reproducibilidad de la inicialización de pesos
    def __init__(self, lr=0.1, epochs=300, mode="classic", seed=42):
        self.lr = lr
        self.epochs = epochs
        self.mode = mode
        self.seed = seed

        self.weights = None
        self.bias = None

        # Historial de métricas por época
        self.history = []           # BCE (pérdida) por época
        self.accuracy_history = []  # exactitud sobre entrenamiento por época
        self.weight_history = []    # norma L2 de los pesos por época

        self.train_time = 0.0
        self.converged_epoch = None

        # Variables internas para detección de convergencia
        self._prev_loss = float("inf")  # pérdida de la época anterior
        self._tol = 1e-6                # umbral mínimo de mejora para considerar convergencia

    # _bce(y: np.ndarray, p: np.ndarray) -> float
    # Calcula la pérdida de entropía cruzada binaria entre etiquetas y y
    # probabilidades predichas p. El clip evita log(0) que daría -infinito.
    def _bce(self, y, p):
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # fit(X: np.ndarray, y: np.ndarray) -> self
    # Entrena el modelo minimizando la BCE mediante gradiente descendente.
    # En modo batch actualiza una vez por época con el gradiente completo.
    # En modo estocástico actualiza una vez por cada ejemplo (orden aleatorio).
    def fit(self, X, y):
        set_seed(self.seed)
        n_samples, n_features = X.shape

        # Inicializar pesos con valores pequeños aleatorios para romper simetría.
        # Usar ceros haría que todos los pesos evolucionaran igual en batch.
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        start = time.perf_counter()

        for epoch in range(self.epochs):

            # Mezclar datos en modo estocástico para evitar correlaciones temporales
            if self.mode == "stochastic":
                idx = np.random.permutation(n_samples)
                X_ep, y_ep = X[idx], y[idx]
            else:
                X_ep, y_ep = X, y

            if self.mode == "stochastic":
                # Actualización online: un ajuste por cada ejemplo
                for xi, yi in zip(X_ep, y_ep):
                    # Calcular la probabilidad predicha con sigmoide
                    out = sigmoid(np.dot(xi, self.weights) + self.bias)
                    # Gradiente de BCE respecto a w es (out - yi) * xi
                    err = out - yi
                    self.weights -= self.lr * err * xi
                    self.bias    -= self.lr * err
            else:
                # Actualización batch: un solo ajuste usando el gradiente promedio
                out = sigmoid(X_ep @ self.weights + self.bias)  # vector de salidas
                err = out - y_ep                                 # vector de errores
                # Gradiente: X.T @ err promedia el gradiente sobre todos los ejemplos
                self.weights -= self.lr * (X_ep.T @ err) / n_samples
                self.bias    -= self.lr * np.mean(err)

            # Calcular métricas sobre todo el conjunto de entrenamiento
            prob = self.predict_proba(X)
            loss = self._bce(y, prob)
            y_pred = (prob >= 0.5).astype(float)
            acc = accuracy_score(y, y_pred)

            self.history.append(loss)
            self.accuracy_history.append(acc)
            self.weight_history.append(np.linalg.norm(self.weights))

            # Convergencia: si la mejora en pérdida es menor que la tolerancia,
            # el modelo ha llegado a un mínimo (local o global)
            if abs(self._prev_loss - loss) < self._tol and self.converged_epoch is None:
                self.converged_epoch = epoch + 1
            self._prev_loss = loss

        self.train_time = time.perf_counter() - start
        return self

    # predict_proba(X: np.ndarray) -> np.ndarray
    # Retorna la probabilidad de clase positiva (1) para cada muestra en X.
    # Valores cercanos a 1 indican alta confianza de que la muestra es positiva.
    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)

    # predict(X: np.ndarray) -> np.ndarray
    # Retorna etiquetas binarias (0 o 1) aplicando umbral 0.5 a las probabilidades.
    # Si P(y=1|x) >= 0.5 se clasifica como 1, de lo contrario como 0.
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(float)