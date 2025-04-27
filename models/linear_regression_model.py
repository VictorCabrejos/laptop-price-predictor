import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from models.prediction_model import PredictionModel

class LinearRegressionModel(PredictionModel):
    """Modelo de regresión lineal para predicción de precios"""

    def __init__(self):
        self.model = LinearRegression()
        self.mae = None
        self.r2 = None

    def train(self, X_train, y_train):
        """Entrena el modelo de regresión lineal"""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        y_pred = self.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        return {'mae': self.mae, 'r2': self.r2, 'predictions': y_pred}

    def get_feature_importances(self, feature_names):
        """Obtiene los coeficientes del modelo como importancia de características"""
        importances = {}
        for i, feature in enumerate(feature_names):
            importances[feature] = float(self.model.coef_[i])
        return importances

    def __str__(self):
        return "Regresión Lineal"