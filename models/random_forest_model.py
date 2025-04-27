import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from models.prediction_model import PredictionModel

class RandomForestModel(PredictionModel):
    """Modelo de Random Forest para predicción de precios"""

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.mae = None
        self.r2 = None

    def train(self, X_train, y_train):
        """Entrena el modelo de Random Forest"""
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
        """Obtiene la importancia de características del Random Forest"""
        importances = {}
        for i, feature in enumerate(feature_names):
            importances[feature] = float(self.model.feature_importances_[i])
        return importances

    def __str__(self):
        return "Random Forest"