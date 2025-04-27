from models.model_config import ModelConfig

class ModelConfigRegistry:
    """
    Registro de prototipos de configuraciones de modelos.
    Permite almacenar y recuperar configuraciones predefinidas.
    Parte del patrón Prototype.
    """

    def __init__(self):
        self.prototypes = {}
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Inicializa configuraciones predeterminadas"""
        # Configuración básica de regresión lineal
        self.prototypes['basic_linear'] = ModelConfig(
            model_type='linear',
            params={'fit_intercept': True}
        )

        # Configuración para Random Forest optimizado
        self.prototypes['optimized_rf'] = ModelConfig(
            model_type='random_forest',
            params={
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 4,
                'random_state': 42
            }
        )

        # Configuración para características reducidas
        reduced_features = [
            'Weight', 'Ram', 'TypeName_Gaming', 'screen_width', 'screen_height'
        ]

        self.prototypes['reduced_features'] = ModelConfig(
            model_type='random_forest',
            features=reduced_features
        )

    def register(self, name, prototype):
        """Registra un nuevo prototipo con el nombre dado"""
        self.prototypes[name] = prototype
        print(f"Prototipo '{name}' registrado con éxito.")

    def unregister(self, name):
        """Elimina un prototipo del registro"""
        if name in self.prototypes:
            del self.prototypes[name]
            print(f"Prototipo '{name}' eliminado.")
        else:
            print(f"No se encontró el prototipo '{name}'.")

    def get(self, name):
        """
        Obtiene un clon del prototipo con el nombre especificado.
        Implementa el patrón Prototype.
        """
        prototype = self.prototypes.get(name)
        if prototype:
            return prototype.clone()
        return None

    def list_prototypes(self):
        """Lista todos los prototipos disponibles"""
        return {name: str(prototype) for name, prototype in self.prototypes.items()}