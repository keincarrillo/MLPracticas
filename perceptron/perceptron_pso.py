# perceptron_pso.py
# Perceptrón con función de activación sigmoide entrenado mediante PSO.
#
# PSO (Particle Swarm Optimization) es un algoritmo bioinspirado que simula
# el comportamiento colectivo de un enjambre (bandada de pájaros, banco de peces).
# Cada "partícula" representa un conjunto completo de pesos del perceptrón.
# Las partículas se mueven en el espacio de búsqueda atraídas por:
#   - Su mejor posición personal histórica (pbest) → componente cognitiva
#   - La mejor posición global encontrada por cualquier partícula (gbest) → componente social
#
# Ecuación de actualización de velocidad (por dimensión d):
#   v_d = w*v_d + c1*r1*(pbest_d - pos_d) + c2*r2*(gbest_d - pos_d)
#   pos_d = pos_d + v_d
#
# Hiperparámetros clave:
#   w  (inercia)   : qué tanto conserva cada partícula su velocidad anterior
#   c1 (cognitivo) : qué tan fuerte es la atracción hacia el mejor propio
#   c2 (social)    : qué tan fuerte es la atracción hacia el mejor global
#
# Modos disponibles:
#   "classic"     → el fitness se evalúa sobre todo el conjunto de datos
#   "stochastic"  → cada generación el fitness se evalúa sobre un mini-lote aleatorio

import numpy as np
import time
from sklearn.metrics import accuracy_score
from utilidades import set_seed, sigmoid

class PerceptronPSO:

    # __init__(n_particles, epochs, w, c1, c2, mode, batch_size, seed) -> None
    # Inicializa el enjambre con sus hiperparámetros.
    # n_particles : número de partículas en el enjambre
    # epochs      : número de generaciones (iteraciones del enjambre)
    # w           : factor de inercia; valor típico entre 0.4 y 0.9
    # c1          : coeficiente cognitivo; controla atracción al pbest propio
    # c2          : coeficiente social; controla atracción al gbest global
    # mode        : "classic" o "stochastic"
    # batch_size  : tamaño del mini-lote en modo estocástico
    # seed        : semilla para reproducibilidad
    def __init__(
        self,
        n_particles=30,
        epochs=200,
        w=0.7,
        c1=1.5,
        c2=1.5,
        mode="classic",
        batch_size=64,
        seed=42
    ):
        self.n_particles = n_particles
        self.epochs = epochs
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed

        # Pesos y sesgo del mejor perceptrón encontrado por el enjambre
        self.weights = None
        self.bias = None

        self.train_time = 0.0
        self.converged_epoch = None

        # Historial de métricas globales (sobre todo el conjunto de datos)
        self.history = []               # BCE del gbest por generación
        self.accuracy_history = []      # exactitud del gbest por generación
        self.weight_history = []        # norma L2 del gbest por generación

        # Métricas específicas del enjambre
        self.swarm_fitness_history = [] # mejor fitness global (gbest) por generación
        self.diversity_history = []     # dispersión media del enjambre por generación

        # Resultados finales del enjambre (después de entrenar)
        self.personal_best_fitness = [] # mejor fitness personal de cada partícula
        self.particle_positions_final = None   # posiciones finales de todas las partículas
        self.particle_velocities_final = None  # velocidades finales de todas las partículas

    # _bce(y: np.ndarray, prob: np.ndarray) -> float
    # Calcula la Binary Cross-Entropy; es la función de fitness que minimiza el PSO.
    # Un valor más bajo indica que los pesos de esa partícula clasifican mejor.
    def _bce(self, y, prob):
        eps = 1e-15
        p = np.clip(prob, eps, 1 - eps)
        return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))

    # _evaluate(pos: np.ndarray, X: np.ndarray, y: np.ndarray) -> float
    # Evalúa el fitness de una partícula dada su posición en el espacio de pesos.
    # pos contiene [w_1, w_2, ..., w_n, bias] como un vector plano de dimensión n+1.
    # Retorna la pérdida BCE; menor es mejor (el PSO minimiza esta función).
    def _evaluate(self, pos, X, y):
        w, b = pos[:-1], pos[-1]
        prob = sigmoid(X @ w + b)
        return self._bce(y, prob)

    # fit(X: np.ndarray, y: np.ndarray) -> self
    # Ejecuta el PSO para encontrar los pesos que minimizan la BCE.
    # El enjambre explora el espacio de pesos colectivamente durante 'epochs' generaciones.
    def fit(self, X, y):
        set_seed(self.seed)
        n_samples, n_features = X.shape

        # Cada partícula es un vector de dimensión n_features + 1 (pesos + sesgo)
        dim = n_features + 1

        # Inicializar posiciones y velocidades con valores pequeños aleatorios.
        # Posiciones cercanas a 0 para empezar desde la región central del espacio.
        pos = np.random.randn(self.n_particles, dim) * 0.1
        vel = np.random.randn(self.n_particles, dim) * 0.01

        # pbest: mejor posición personal que cada partícula ha visitado
        pbest_pos = pos.copy()
        # Evaluar el fitness inicial de cada partícula sobre el conjunto completo
        pbest_fit = np.array([self._evaluate(p, X, y) for p in pos])

        # gbest: posición global con menor fitness entre todas las partículas
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        start = time.perf_counter()

        for epoch in range(self.epochs):

            # En modo estocástico se usa un subconjunto aleatorio para evaluar fitness.
            # Esto acelera cada generación y añade ruido que puede ayudar a escapar
            # de mínimos locales (similar al SGD vs GD en gradiente descendente).
            if self.mode == "stochastic":
                idx = np.random.choice(n_samples, self.batch_size, replace=False)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y

            # r1 y r2 son factores de aleatoriedad independientes por partícula y dimensión.
            # Introducen estocasticidad en la atracción hacia pbest y gbest.
            r1 = np.random.rand(self.n_particles, dim)
            r2 = np.random.rand(self.n_particles, dim)

            # Actualizar velocidades: mezcla de inercia + atracción cognitiva + atracción social
            vel = (
                self.w * vel                          # inercia: tendencia a seguir moviéndose igual
                + self.c1 * r1 * (pbest_pos - pos)   # cognitivo: volver a donde fui mejor
                + self.c2 * r2 * (gbest_pos - pos)   # social: ir hacia donde el enjambre fue mejor
            )
            # Mover partículas según su nueva velocidad
            pos = pos + vel

            # Evaluar fitness de todas las partículas en la generación actual
            fits = np.array([self._evaluate(p, Xb, yb) for p in pos])

            # Actualizar pbest: si la nueva posición es mejor que la histórica personal
            improved = fits < pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fits[improved]

            # Actualizar gbest: si alguna partícula superó al mejor global conocido
            new_best_idx = np.argmin(pbest_fit)
            if pbest_fit[new_best_idx] < gbest_fit:
                gbest_fit = pbest_fit[new_best_idx]
                gbest_pos = pbest_pos[new_best_idx].copy()

            # Diversidad: desviación estándar promedio de las posiciones del enjambre.
            # Un valor alto indica que el enjambre está explorando ampliamente;
            # un valor bajo indica convergencia (partículas agrupadas en zona similar).
            diversity = np.mean(np.std(pos, axis=0))

            # Evaluar métricas globales con el mejor conjunto de pesos (gbest)
            w_g, b_g = gbest_pos[:-1], gbest_pos[-1]
            prob_all = sigmoid(X @ w_g + b_g)
            loss_all = self._bce(y, prob_all)
            acc_all = accuracy_score(y, (prob_all >= 0.5).astype(float))

            self.history.append(loss_all)
            self.accuracy_history.append(acc_all)
            self.weight_history.append(np.linalg.norm(w_g))
            self.swarm_fitness_history.append(gbest_fit)
            self.diversity_history.append(diversity)

            # Convergencia: cuando el mejor fitness cae por debajo de un umbral
            if gbest_fit < 0.1 and self.converged_epoch is None:
                self.converged_epoch = epoch + 1

        self.train_time = time.perf_counter() - start

        # Guardar estado final del enjambre para análisis post-entrenamiento
        self.personal_best_fitness = pbest_fit.tolist()
        self.particle_positions_final = pos.copy()
        self.particle_velocities_final = vel.copy()

        # El modelo final usa los pesos del mejor global encontrado
        self.weights = gbest_pos[:-1]
        self.bias = gbest_pos[-1]
        return self

    # predict_proba(X: np.ndarray) -> np.ndarray
    # Retorna la probabilidad de clase positiva usando los pesos del gbest.
    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)

    # predict(X: np.ndarray) -> np.ndarray
    # Retorna etiquetas binarias (0 o 1) con umbral 0.5 sobre las probabilidades.
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(float)