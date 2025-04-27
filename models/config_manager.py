import os
import json

class ConfigManager:
    """
    Implementación del patrón Singleton para gestionar la configuración
    de la aplicación de predicción de precios de laptops.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializa las configuraciones por defecto"""
        self.config = {
            'dataset_path': 'laptop_price.csv',
            'encoding': 'ISO-8859-1',
            'test_size': 0.15,
            'random_state': 42,
            'features': ['Weight', 'TypeName_Gaming', 'TypeName_Notebook',
                         'screen_width', 'screen_height', 'GHz', 'Ram'],
            'target': 'Price_euros',
            'model_type': 'linear_regression',
            'static_dir': 'static'
        }

        # Cargar configuración desde archivo si existe
        self._load_config()

        # Crear directorios necesarios
        os.makedirs(self.config['static_dir'], exist_ok=True)

    def _load_config(self):
        """Carga configuración desde un archivo JSON si existe"""
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                print(f"Error al cargar la configuración: {e}")

    def save_config(self):
        """Guarda la configuración actual en un archivo JSON"""
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error al guardar la configuración: {e}")
            return False

    def get(self, key, default=None):
        """Obtiene un valor de configuración específico"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Establece un valor de configuración específico"""
        self.config[key] = value
        return self