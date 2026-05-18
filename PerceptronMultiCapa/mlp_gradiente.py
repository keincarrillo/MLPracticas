# mlp_gradiente.py
# Red neuronal multicapa (MLP) entrenada con retropropagación del error (backprop).
#
# DIFERENCIA CLAVE respecto al perceptrón simple:
#   El perceptrón simple tiene UNA sola capa y aprende con la regla Delta.
#   El MLP tiene MÚLTIPLES capas y necesita backpropagation para calcular
#   cómo contribuye cada peso al error final, sin importar cuán profunda esté la capa.
#
# FORWARD PASS: los datos fluyen de entrada → capas ocultas → salida.
#   Cada capa aplica: activación(pesos @ entrada + sesgo)
#
# BACKWARD PASS: el error fluye de salida → capas ocultas → entrada.
#   Cada capa calcula su gradiente usando la regla de la cadena.
#
# Función de pérdida: Entropía cruzada categórica (multiclase)
# Activación oculta: Sigmoide
# Activación salida: Softmax (produce distribución de probabilidad sobre las 3 clases)
#
# Modos disponibles:
#   "classic"     → gradiente calculado sobre todo el conjunto (batch GD)
#   "stochastic"  → gradiente calculado con mini-lotes aleatorios (SGD)

import numpy as np
import time
from sklearn.metrics import accuracy_score

from utilidades import (
    set_seed, sigmoid, sigmoid_derivative,
    softmax, cross_entropy, to_one_hot, compute_metrics
)

class MLPGradiente:

    # __init__(hidden_layers, lr, epochs, mode, batch_size, seed) -> None
    # Inicializa la arquitectura y los hiperparámetros de la red.
    #
    # hidden_layers : lista con el número de neuronas por capa oculta.
    #                 Ejemplo: [8] = una capa oculta con 8 neuronas.
    #                          [16, 8] = dos capas ocultas con 16 y 8 neuronas.
    # lr            : tasa de aprendizaje; controla el tamaño del paso en cada actualización.
    #                 Muy grande → los pesos oscilan y divergen.
    #                 Muy pequeña → converge muy lento.
    # epochs        : número de pasadas completas por el conjunto de entrenamiento.
    # mode          : "classic" (batch GD) o "stochastic" (mini-batch SGD).
    # batch_size    : tamaño del mini-lote en modo estocástico.
    # seed          : semilla para reproducibilidad de la inicialización de pesos.
    def __init__(self, hidden_layers=(8,), lr=0.1, epochs=300,
                 mode="classic", batch_size=16, seed=42):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed

        # Pesos y sesgos de cada capa; se inicializan en fit()
        self.weights = []   # lista de matrices W por capa
        self.biases  = []   # lista de vectores b por capa

        # Historial de métricas por época para graficar curvas de aprendizaje
        self.history          = []   # entropía cruzada por época
        self.accuracy_history = []   # exactitud sobre entrenamiento por época
        self.weight_history   = []   # norma Frobenius total de todos los pesos

        self.train_time      = 0.0   # duración total del entrenamiento en segundos
        self.converged_epoch = None  # época en que la mejora fue menor a la tolerancia

        self._prev_loss = float("inf")
        self._tol       = 1e-5

    # _build_layers(n_in: int, n_out: int) -> None
    # Construye la lista de matrices de pesos y sesgos para toda la red.
    # La arquitectura completa es: n_in → hidden_layers[0] → ... → n_out
    #
    # Inicialización de Xavier (Glorot):
    #   w ~ U(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
    # Mantiene la varianza de las activaciones estable a lo largo de las capas.
    # Sin esto, los gradientes pueden explotar o desvanecerse en redes profundas.
    def _build_layers(self, n_in, n_out):
        self.weights = []
        self.biases  = []
        dims = [n_in] + list(self.hidden_layers) + [n_out]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            # Inicialización Xavier: distribuye uniformemente en [-limit, limit]
            W = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

    # _forward(X: np.ndarray) -> (list, list)
    # Pasa los datos X por todas las capas de la red (forward pass).
    # Retorna:
    #   activations : lista con la activación de cada capa (incluye entrada)
    #   nets        : lista con la suma ponderada (antes de activación) de cada capa
    #
    # Las capas ocultas usan sigmoide; la capa de salida usa softmax.
    # Se guardan las activaciones porque el backward pass las necesita para los gradientes.
    def _forward(self, X):
        activations = [X]          # activaciones[0] = datos de entrada
        nets        = []           # salidas netas antes de la activación
        current     = X

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Suma ponderada: cada neurona recibe la combinación lineal de sus entradas
            net = current @ W + b
            nets.append(net)

            is_last = (i == len(self.weights) - 1)
            if is_last:
                # Capa de salida: softmax convierte logits en probabilidades
                current = softmax(net)
            else:
                # Capas ocultas: sigmoide introduce la no-linealidad necesaria
                current = sigmoid(net)
            activations.append(current)

        return activations, nets

    # _backward(activations: list, nets: list, y_oh: np.ndarray) -> (list, list)
    # Calcula los gradientes de la pérdida respecto a cada peso usando la regla de la cadena.
    # Fluye el error desde la capa de salida hacia la entrada (retropropagación).
    #
    # y_oh : etiquetas en formato one-hot, forma (n_muestras, n_clases)
    # Retorna:
    #   grads_W : lista de gradientes dL/dW para cada capa
    #   grads_b : lista de gradientes dL/db para cada capa
    def _backward(self, activations, nets, y_oh):
        n = y_oh.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # δ (delta) de la capa de salida:
        # Para softmax + entropía cruzada, el gradiente combinado simplifica a (ŷ - y).
        # Esta simplificación elegante es una de las razones para elegir softmax+CE juntos.
        delta = activations[-1] - y_oh   # forma: (n_muestras, n_clases)

        # Calcular gradientes de la última capa
        grads_W[-1] = activations[-2].T @ delta / n   # dL/dW = a_anterior.T @ δ / n
        grads_b[-1] = delta.mean(axis=0)              # dL/db = promedio de δ

        # Propagar el error hacia atrás por las capas ocultas
        for i in range(len(self.weights) - 2, -1, -1):
            # Propagar delta hacia la capa anterior a través de los pesos transpuestos
            delta = (delta @ self.weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
            # La derivada de sigmoid indica cuánto "pasó" la señal por esa neurona

            grads_W[i] = activations[i].T @ delta / n
            grads_b[i] = delta.mean(axis=0)

        return grads_W, grads_b

    # fit(X: np.ndarray, y: np.ndarray) -> self
    # Entrena la red sobre el conjunto de datos X con etiquetas enteras y.
    # En modo batch: usa todo X para calcular el gradiente en cada época.
    # En modo estocástico: divide X en mini-lotes y actualiza con cada lote.
    def fit(self, X, y):
        set_seed(self.seed)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self._build_layers(n_features, n_classes)

        # Convertir etiquetas enteras a one-hot para calcular la pérdida
        y_oh = to_one_hot(y, n_classes)

        start = time.perf_counter()

        for epoch in range(self.epochs):

            if self.mode == "stochastic":
                # Mini-batch SGD: mezclar datos y procesar en lotes pequeños
                # Introduce ruido en el gradiente → puede escapar mínimos locales
                idx = np.random.permutation(n_samples)
                for start_b in range(0, n_samples, self.batch_size):
                    batch = idx[start_b: start_b + self.batch_size]
                    Xb, yb_oh = X[batch], y_oh[batch]

                    acts, nets = self._forward(Xb)
                    gW, gb    = self._backward(acts, nets, yb_oh)

                    # Descenso de gradiente: moverse en dirección contraria al gradiente
                    for i in range(len(self.weights)):
                        self.weights[i] -= self.lr * gW[i]
                        self.biases[i]  -= self.lr * gb[i]
            else:
                # Batch GD: gradiente exacto sobre todo el conjunto
                # Más estable pero más lento por época en datasets grandes
                acts, nets = self._forward(X)
                gW, gb    = self._backward(acts, nets, y_oh)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * gW[i]
                    self.biases[i]  -= self.lr * gb[i]

            # Calcular métricas sobre todo el conjunto al final de cada época
            prob = self._forward(X)[0][-1]         # salida softmax
            loss = cross_entropy(y_oh, prob)
            y_pred = prob.argmax(axis=1)
            acc = accuracy_score(y, y_pred)

            self.history.append(loss)
            self.accuracy_history.append(acc)
            # Norma de Frobenius total: mide la magnitud acumulada de todos los pesos
            self.weight_history.append(
                sum(np.linalg.norm(W) for W in self.weights)
            )

            # Convergencia: si la mejora en pérdida es menor que la tolerancia
            if abs(self._prev_loss - loss) < self._tol and self.converged_epoch is None:
                self.converged_epoch = epoch + 1
            self._prev_loss = loss

        self.train_time = time.perf_counter() - start
        return self

    # predict_proba(X: np.ndarray) -> np.ndarray
    # Retorna la distribución de probabilidad sobre las 3 clases para cada muestra.
    # La suma por fila siempre es 1 (propiedad de softmax).
    def predict_proba(self, X):
        return self._forward(X)[0][-1]

    # predict(X: np.ndarray) -> np.ndarray
    # Retorna la clase predicha (0, 1 o 2) para cada muestra.
    # Elige la clase con mayor probabilidad (argmax sobre softmax).
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
