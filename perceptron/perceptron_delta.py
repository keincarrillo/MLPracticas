# perceptron_delta.py
# Perceptrón con función de activación escalón y regla de aprendizaje Delta.
#
# La REGLA DELTA ajusta los pesos proporcionalmente al error cometido:
#   Δw = lr * (y_real - y_predicho) * x
# No requiere que la función de activación sea diferenciable.
#
# Modos disponibles:
#   "classic"     → procesa todos los ejemplos antes de guardar métricas (batch)
#   "stochastic"  → actualiza pesos ejemplo a ejemplo en orden aleatorio (online)

import numpy as np
import time
from sklearn.metrics import accuracy_score
from utilidades import set_seed, sigmoid, step

class PerceptronDelta:

    # __init__(lr, epochs, mode, seed) -> None
    # Inicializa los hiperparámetros y reserva el espacio para el historial.
    # lr     : tasa de aprendizaje; controla cuánto cambian los pesos por error
    # epochs : número de pasadas completas por el conjunto de entrenamiento
    # mode   : "classic" o "stochastic"
    # seed   : semilla para reproducibilidad
    def __init__(self, lr=0.01, epochs=200, mode="classic", seed=42):
        self.lr = lr
        self.epochs = epochs
        self.mode = mode
        self.seed = seed

        # Pesos del perceptrón; se inicializan en fit()
        self.weights = None
        # Término de sesgo (bias); permite desplazar el hiperplano de decisión
        self.bias = None

        # Listas que registran la evolución del modelo época a época
        self.history = []           # error cuadrático medio por época
        self.accuracy_history = []  # exactitud sobre el conjunto de entrenamiento
        self.weight_history = []    # norma L2 de los pesos (mide su magnitud)

        self.train_time = 0.0       # duración total del entrenamiento en segundos
        self.converged_epoch = None # época en que el error llegó a 0 (si ocurre)

    # fit(X: np.ndarray, y: np.ndarray) -> self
    # Entrena el perceptrón sobre el conjunto de datos X con etiquetas y.
    # X tiene forma (n_muestras, n_características), y tiene forma (n_muestras,).
    # Retorna self para permitir encadenamiento: modelo.fit(X, y).predict(X_test)
    def fit(self, X, y):
        set_seed(self.seed)
        n_samples, n_features = X.shape

        # Inicializar pesos en cero; el sesgo también en cero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        start = time.perf_counter()

        for epoch in range(self.epochs):

            # En modo estocástico se mezcla el orden de los ejemplos cada época
            # para evitar que el modelo aprenda el orden de presentación
            if self.mode == "stochastic":
                idx = np.random.permutation(n_samples)
                X_ep, y_ep = X[idx], y[idx]
            else:
                # En modo clásico se usan todos los ejemplos en orden original
                X_ep, y_ep = X, y

            epoch_loss = 0.0

            for xi, yi in zip(X_ep, y_ep):
                # Calcular la salida neta: suma ponderada de entradas + sesgo
                net = np.dot(xi, self.weights) + self.bias

                # Aplicar la función escalón: 1 si net >= 0, 0 si net < 0
                out = step(np.array([net]))[0]

                # El error es la diferencia entre la salida deseada y la obtenida
                # Si out == yi, error = 0 y los pesos no cambian
                error = yi - out

                # Regla Delta: actualizar pesos proporcional al error y la entrada
                # Un error positivo (debía ser 1, salió 0) empuja w hacia xi
                # Un error negativo (debía ser 0, salió 1) aleja w de xi
                self.weights += self.lr * error * xi
                self.bias    += self.lr * error

                # Acumular error cuadrático para la métrica de la época
                epoch_loss += error ** 2

            # Guardar métricas al final de cada época
            y_pred = self.predict(X)
            acc = accuracy_score(y, y_pred)
            self.history.append(epoch_loss / n_samples)     # MSE de la época
            self.accuracy_history.append(acc)
            self.weight_history.append(np.linalg.norm(self.weights))

            # Detectar convergencia: si todos los ejemplos se clasifican bien
            if epoch_loss == 0 and self.converged_epoch is None:
                self.converged_epoch = epoch + 1

        self.train_time = time.perf_counter() - start
        return self

    # predict(X: np.ndarray) -> np.ndarray
    # Retorna las etiquetas predichas (0 o 1) para cada fila de X.
    # Calcula la salida neta vectorialmente y aplica la función escalón.
    def predict(self, X):
        net = X @ self.weights + self.bias
        return step(net)

    # predict_proba(X: np.ndarray) -> np.ndarray
    # Retorna una estimación de probabilidad suave en el rango (0, 1).
    # La función escalón no produce probabilidades naturales, por lo que
    # se usa la sigmoide sobre la salida neta como aproximación continua.
    # Útil para calcular AUC-ROC y graficar curvas ROC.
    def predict_proba(self, X):
        net = X @ self.weights + self.bias
        return sigmoid(net)