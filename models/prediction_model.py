from abc import ABC, abstractmethod

class PredictionModel(ABC):
    """
    Clase abstracta que define la interfaz para todos los modelos de predicción.
    Siguiendo el patrón Factory, esta es la interfaz que deben implementar
    todas las clases concretas de modelos.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """Entrena el modelo con los datos proporcionados"""
        pass

    @abstractmethod
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        pass

    @abstractmethod
    def get_feature_importances(self, feature_names):
        """Obtiene la importancia de las características"""
        pass