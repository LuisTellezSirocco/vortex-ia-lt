"""Ejemplo de uso de este script:

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

# Crear un dataset de regresión sintético
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar dos modelos de regresión
model1 = LinearRegression().fit(X_train, y_train)
model2 = Ridge().fit(X_train, y_train)

# Predecir con los modelos
y_pred1 = model1.predict(X_val)
y_pred2 = model2.predict(X_val)
y_preds = [y_pred1, y_pred2]

# Crear una instancia de OptunaWeights sin especificar la métrica
ow = OptunaWeights(random_state=42, n_trials=100, direction='minimize')

# Ajustar la instancia a los datos
ow.fit(y_val, y_preds)

# Obtener las predicciones ponderadas
predictions = ow.predict(y_preds)

# Verificar los pesos optimizados
print("Pesos optimizados:", ow.get_weights())

"""

import numpy as np
import optuna
import pandas as pd
from pandas import DataFrame
from functools import partial
from sklearn.metrics import mean_squared_error

class OptunaWeights:
    """
    OptunaWeights is a class for optimizing the weights of predictions from multiple models using Optuna.

    :param random_state: Seed for the random number generator.
    :param metric: A callable that takes y_true and y_pred as inputs and returns a score.
    :param n_trials: The number of trials for the optimization.
    :param direction: The direction of optimization - 'minimize' or 'maximize'.
    """

    def __init__(self, random_state, metric=None, n_trials=100, direction="minimize"):
        # Inicializa los atributos de la clase.
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials
        self.metric = metric if metric is not None else self.rmse
        self.direction = direction

    @staticmethod
    def rmse(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

    @staticmethod
    def _validate_inputs(y_true, y_preds):
        """
        Private helper function to validate the inputs.
        Raises ValueError if the inputs are not valid.

        :param y_true: Array-like of true values.
        :param y_preds: List of array-like predictions from different models.
        """
        # Valida las entradas y lanza excepciones con mensajes de error útiles si no son válidas.
        if not isinstance(y_preds, list):
            raise ValueError("`y_preds` should be a list of predictions.")
        if not all(isinstance(y, np.ndarray) for y in y_preds):
            raise ValueError("Each element of `y_preds` should be a numpy array.")
        if not isinstance(y_true, np.ndarray):
            raise ValueError("`y_true` should be a numpy array.")
        if len(y_true) != len(y_preds[0]):
            raise ValueError("`y_true` and `y_preds` should have the same length.")

    def _objective(self, trial, y_true, y_preds):
        """
        Private helper function to define the objective for the optimization.

        :param trial: Trial instance.
        :param y_true: Array-like of true values.
        :param y_preds: List of array-like predictions from different models.
        :return: Score calculated based on the predictions and the provided metric.
        """
        # Define la función objetivo para la optimización, que calcula una puntuación basada en las predicciones ponderadas.
        weights = [
            trial.suggest_float(f"weight{n}", 1e-15, 1) for n in range(len(y_preds))
        ]
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
        score = self.metric(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        """
        Fits the model, performing the optimization to find the best weights.

        :param y_true: Array-like of true values.
        :param y_preds: List of array-like predictions from different models.
        """
        # Ajusta el modelo, realizando la optimización para encontrar los mejores pesos.
        self._validate_inputs(y_true, y_preds)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            study_name="OptunaWeights",
            direction=self.direction,
        )
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [
            self.study.best_params[f"weight{n}"] for n in range(len(y_preds))
        ]

    def predict(self, y_preds):
        """
        Makes a prediction using the optimized weights.

        :param y_preds: List of array-like predictions from different models.
        :return: Array-like of weighted predictions.
        """
        # Realiza una predicción con los pesos optimizados.
        if self.weights is None:
            raise Exception("OptunaWeights error, must be fitted before predict")
        if not isinstance(y_preds, list) or not all(
            isinstance(y, np.ndarray) for y in y_preds
        ):
            raise ValueError("`y_preds` should be a list of numpy arrays.")
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        """
        Fits the model and then makes a prediction.

        :param y_true: Array-like of true values.
        :param y_preds: List of array-like predictions from different models.
        :return: Array-like of weighted predictions.
        """
        # Ajusta el modelo y luego realiza una predicción.
        self.fit(y_true, y_preds)
        return self.predict(y_preds)

    def get_weights(self):
        """
        Returns the optimized weights.

        :return: List of optimized weights.
        """
        # Devuelve los pesos optimizados.
        return self.weights