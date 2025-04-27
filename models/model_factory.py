from models.linear_regression_model import LinearRegressionModel
from models.random_forest_model import RandomForestModel

class ModelFactory:
    """
    Factory para crear diferentes tipos de modelos de predicción.

    Esta clase implementa el patrón Factory Method para devolver
    instancias de diferentes modelos de predicción según el tipo solicitado.
    """

    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Crea y devuelve un modelo según el tipo especificado.

        Args:
            model_type (str): Tipo de modelo a crear ('linear', 'linear_regression', o 'random_forest')
            **kwargs: Parámetros adicionales para la creación del modelo

        Returns:
            PredictionModel: Una instancia del modelo solicitado

        Raises:
            ValueError: Si el tipo de modelo no es reconocido
        """
        if model_type == 'linear' or model_type == 'linear_regression':
            return LinearRegressionModel()
        elif model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            random_state = kwargs.get('random_state', 42)
            return RandomForestModel(n_estimators, random_state)
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {model_type}")