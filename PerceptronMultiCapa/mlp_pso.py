# mlp_pso.py
# Red neuronal multicapa (MLP) entrenada con Optimización por Enjambre de Partículas (PSO).
#
# DIFERENCIA respecto al perceptrón PSO anterior:
#   Antes: cada partícula era un vector de pesos de UN perceptrón (n_features + 1 valores).
#   Ahora: cada partícula codifica TODOS los pesos y sesgos de la red multicapa.
#          Un MLP 4→8→3 tiene (4*8 + 8) + (8*3 + 3) = 40 + 27 = 67 parámetros.
#   El espacio de búsqueda es mucho más grande → se necesitan más partículas o épocas.
#
# El PSO NO usa gradientes. En cambio, evalúa la calidad de cada conjunto de pesos
# directamente pasándolos por la red y midiendo la pérdida.
# Ventaja: puede escapar mínimos locales que atrapan al gradiente descendente.
# Desventaja: es más lento por época porque evalúa N_PARTICLES redes completas.
#
# Modos disponibles:
#   "classic"     → fitness evaluado sobre todo el conjunto de datos
#   "stochastic"  → fitness evaluado sobre un mini-lote aleatorio por generación

import numpy as np
import time
from sklearn.metrics import accuracy_score

from utilidades import (
    set_seed, sigmoid, softmax,
    cross_entropy, to_one_hot, compute_metrics
)

class MLPPSO:

    # __init__(...) -> None
    # Inicializa el enjambre y la arquitectura de la red.
    #
    # hidden_layers : lista con neuronas por capa oculta (igual que MLPGradiente)
    # n_particles   : número de partículas en el enjambre.
    #                 Más partículas → mejor exploración pero más lento.
    # epochs        : número de generaciones del enjambre.
    # w             : inercia; qué tanto conserva la partícula su velocidad anterior.
    #                 w > 1 → exploración amplia. w < 0.5 → explotación local.
    # c1            : coeficiente cognitivo; atracción hacia el mejor propio (pbest).
    # c2            : coeficiente social; atracción hacia el mejor global (gbest).
    # mode          : "classic" o "stochastic"
    # batch_size    : tamaño del mini-lote en modo estocástico
    # seed          : semilla de reproducibilidad
    def __init__(self, hidden_layers=(8,), n_particles=30, epochs=200,
                 w=0.7, c1=1.5, c2=1.5, mode="classic", batch_size=32, seed=42):
        self.hidden_layers = hidden_layers
        self.n_particles   = n_particles
        self.epochs        = epochs
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.mode       = mode
        self.batch_size = batch_size
        self.seed       = seed

        # Arquitectura interna (se define en fit cuando se conocen los datos)
        self._layer_dims  = []    # dimensiones de cada capa: [(fan_in, fan_out), ...]
        self._param_dim   = 0     # total de parámetros que codifica cada partícula

        # Mejor solución encontrada por el enjambre
        self.weights = []         # matrices de pesos del mejor MLP
        self.biases  = []         # sesgos del mejor MLP

        self.train_time      = 0.0
        self.converged_epoch = None

        # Historial de métricas globales
        self.history          = []   # entropía cruzada del gbest por generación
        self.accuracy_history = []   # exactitud del gbest por generación
        self.weight_history   = []   # norma total del gbest por generación

        # Métricas del enjambre
        self.swarm_fitness_history  = []
        self.diversity_history      = []
        self.personal_best_fitness  = []

    # _compute_dims(n_in: int, n_out: int) -> None
    # Calcula las dimensiones de cada capa y el número total de parámetros.
    # Se llama en fit() una vez que se conocen n_features y n_classes.
    # Necesario para saber cómo "empaquetar" y "desempaquetar" los pesos de una partícula.
    def _compute_dims(self, n_in, n_out):
        dims = [n_in] + list(self.hidden_layers) + [n_out]
        self._layer_dims = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        # Total de parámetros = suma de (fan_in * fan_out + fan_out) por capa
        self._param_dim = sum(fi * fo + fo for fi, fo in self._layer_dims)

    # _unpack(pos: np.ndarray) -> (list, list)
    # Convierte el vector plano de una partícula en listas de matrices W y vectores b.
    # Una partícula es un vector de longitud _param_dim que contiene todos los parámetros.
    # Desempaquetar = reconstruir la estructura de la red para poder hacer forward pass.
    def _unpack(self, pos):
        weights, biases = [], []
        offset = 0
        for fan_in, fan_out in self._layer_dims:
            size_W = fan_in * fan_out
            W = pos[offset: offset + size_W].reshape(fan_in, fan_out)
            offset += size_W
            b = pos[offset: offset + fan_out]
            offset += fan_out
            weights.append(W)
            biases.append(b)
        return weights, biases

    # _forward(X: np.ndarray, weights: list, biases: list) -> np.ndarray
    # Pasa los datos por la red definida por los pesos dados y retorna la salida softmax.
    # A diferencia del MLPGradiente, aquí recibe pesos explícitos porque evaluamos
    # múltiples partículas (redes diferentes) en cada generación.
    def _forward(self, X, weights, biases):
        current = X
        for i, (W, b) in enumerate(zip(weights, biases)):
            net = current @ W + b
            if i == len(weights) - 1:
                current = softmax(net)   # capa de salida: distribución de probabilidad
            else:
                current = sigmoid(net)   # capas ocultas: no-linealidad
        return current

    # _evaluate(pos: np.ndarray, X: np.ndarray, y_oh: np.ndarray) -> float
    # Evalúa la calidad (fitness) de una partícula dada su posición en el espacio de pesos.
    # Desempaqueta los pesos, hace un forward pass y calcula la pérdida de entropía cruzada.
    # El PSO minimiza este valor → menor pérdida = mejor partícula.
    def _evaluate(self, pos, X, y_oh):
        W_list, b_list = self._unpack(pos)
        prob = self._forward(X, W_list, b_list)
        return cross_entropy(y_oh, prob)

    # fit(X: np.ndarray, y: np.ndarray) -> self
    # Ejecuta el PSO para encontrar los pesos del MLP que minimizan la entropía cruzada.
    # Cada generación:
    #   1. Actualiza velocidades con inercia + atracción cognitiva + atracción social.
    #   2. Mueve partículas.
    #   3. Evalúa fitness de cada partícula.
    #   4. Actualiza pbest y gbest.
    def fit(self, X, y):
        set_seed(self.seed)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self._compute_dims(n_features, n_classes)
        y_oh = to_one_hot(y, n_classes)

        dim = self._param_dim   # dimensionalidad del espacio de búsqueda

        # Inicializar el enjambre con posiciones y velocidades pequeñas aleatorias.
        # Posiciones cercanas a 0 son un buen punto de partida (como Xavier).
        pos = np.random.randn(self.n_particles, dim) * 0.1
        vel = np.random.randn(self.n_particles, dim) * 0.01

        pbest_pos = pos.copy()
        # Evaluar fitness inicial de cada partícula sobre todo el conjunto
        pbest_fit = np.array([self._evaluate(p, X, y_oh) for p in pos])

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        start = time.perf_counter()

        for epoch in range(self.epochs):

            # En modo estocástico se evalúa sobre un mini-lote para acelerar cada generación.
            # El ruido introducido puede ayudar a escapar de mínimos locales.
            if self.mode == "stochastic":
                idx = np.random.choice(n_samples, self.batch_size, replace=False)
                Xb, yb_oh = X[idx], y_oh[idx]
            else:
                Xb, yb_oh = X, y_oh

            # r1 y r2: factores aleatorios independientes por partícula y dimensión.
            # Introducen estocasticidad en la dirección de atracción hacia pbest y gbest.
            r1 = np.random.rand(self.n_particles, dim)
            r2 = np.random.rand(self.n_particles, dim)

            # Actualizar velocidades:
            #   w * vel   → inercia: conservar dirección actual
            #   c1*r1*... → componente cognitiva: volver a donde la partícula fue mejor
            #   c2*r2*... → componente social: ir hacia donde el enjambre fue mejor
            vel = (
                self.w  * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            pos = pos + vel

            # Evaluar todas las partículas en esta generación
            fits = np.array([self._evaluate(p, Xb, yb_oh) for p in pos])

            # Actualizar pbest: si la partícula mejoró su récord personal
            improved = fits < pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fits[improved]

            # Actualizar gbest: si alguna partícula superó el récord global
            best_idx = np.argmin(pbest_fit)
            if pbest_fit[best_idx] < gbest_fit:
                gbest_fit = pbest_fit[best_idx]
                gbest_pos = pbest_pos[best_idx].copy()

            # Diversidad del enjambre: desviación estándar promedio de las posiciones.
            # Alta → las partículas están dispersas (explorando).
            # Baja → las partículas están agrupadas (convergiendo).
            diversity = np.mean(np.std(pos, axis=0))

            # Calcular métricas globales usando el mejor conjunto de pesos (gbest)
            W_g, b_g = self._unpack(gbest_pos)
            prob_all = self._forward(X, W_g, b_g)
            loss_all = cross_entropy(y_oh, prob_all)
            acc_all  = accuracy_score(y, prob_all.argmax(axis=1))

            self.history.append(loss_all)
            self.accuracy_history.append(acc_all)
            self.weight_history.append(
                sum(np.linalg.norm(W_g[i]) for i in range(len(W_g)))
            )
            self.swarm_fitness_history.append(gbest_fit)
            self.diversity_history.append(diversity)

            # Convergencia: cuando el fitness global cae por debajo de un umbral bajo
            if gbest_fit < 0.05 and self.converged_epoch is None:
                self.converged_epoch = epoch + 1

        self.train_time = time.perf_counter() - start

        # Guardar estado final del enjambre para análisis
        self.personal_best_fitness = pbest_fit.tolist()

        # Extraer los mejores pesos encontrados y guardarlos en la red
        self.weights, self.biases = self._unpack(gbest_pos)
        return self

    # predict_proba(X: np.ndarray) -> np.ndarray
    # Retorna probabilidades sobre las 3 clases usando los pesos del gbest.
    def predict_proba(self, X):
        return self._forward(X, self.weights, self.biases)

    # predict(X: np.ndarray) -> np.ndarray
    # Retorna la clase predicha (0, 1 o 2) como el argmax de las probabilidades.
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
