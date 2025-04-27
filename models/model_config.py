import copy
import json

class ModelConfig:
    """
    Clase que actúa como prototipo para configuraciones de modelos.
    Permite crear y clonar configuraciones para diferentes tipos de modelos.
    Implementa el patrón Prototype.
    """

    def __init__(self, model_type='linear', params=None, features=None):
        # Tipo de modelo
        self.model_type = model_type

        # Parámetros del modelo (valores por defecto)
        self.params = {
            'test_size': 0.15,
            'random_state': 42
        }

        # Parámetros específicos según el tipo de modelo
        if model_type == 'random_forest':
            self.params.update({
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            })

        # Actualizar con parámetros personalizados si se proporcionan
        if params:
            self.params.update(params)

        # Características para el modelo
        self.features = features or [
            'Weight', 'TypeName_Gaming', 'TypeName_Notebook',
            'screen_width', 'screen_height', 'GHz', 'Ram'
        ]

        # Configuración de visualización
        self.visualization = {
            'figsize': (10, 6),
            'color': 'blue',
            'alpha': 0.6,
            'title_fontsize': 14,
            'save_path': 'static/'
        }

        # Configuración de preprocesamiento
        self.preprocessing = {
            'encoding': 'ISO-8859-1',
            'scale_features': False,
            'handle_missing': 'median',
            'drop_outliers': False
        }

    def clone(self):
        """
        Crea un clon profundo de la configuración actual.
        Implementa el patrón Prototype.
        """
        return copy.deepcopy(self)

    def to_dict(self):
        """Convierte la configuración a un diccionario"""
        return {
            'model_type': self.model_type,
            'params': self.params,
            'features': self.features,
            'visualization': self.visualization,
            'preprocessing': self.preprocessing
        }

    def save(self, filepath):
        """Guarda la configuración en un archivo JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"Configuración guardada en {filepath}")

    @classmethod
    def load(cls, filepath):
        """Carga la configuración desde un archivo JSON"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = cls(
            model_type=config_dict['model_type'],
            params=config_dict['params'],
            features=config_dict['features']
        )

        config.visualization = config_dict['visualization']
        config.preprocessing = config_dict['preprocessing']

        return config

    def __str__(self):
        """Representación en texto de la configuración"""
        return f"ModelConfig(type={self.model_type}, features={len(self.features)})"