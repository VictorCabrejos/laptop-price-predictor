# Implementación del Patrón Singleton para el proyecto Laptop Price Predictor
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


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


class ModelManager:
    """
    Implementación del patrón Singleton para gestionar el modelo
    de predicción de precios y sus datos asociados.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializa las variables del gestor de modelos"""
        self.model = None
        self.feature_importances = {}
        self.mae = 0
        self.r2 = 0
        self.config_manager = ConfigManager()  # Utiliza el Singleton de configuración

    def load_data(self):
        """Carga y preprocesa los datos desde el archivo CSV"""
        config = self.config_manager.config

        # Carga el dataset
        df = pd.read_csv(config['dataset_path'], encoding=config['encoding'])

        # Limpia y transforma los datos
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extrae la resolución de pantalla
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(
            r'(\d{3,4})x(\d{3,4})'
        ).astype(float)

        # Extrae la velocidad del CPU
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        # Selección de características
        X = df[config['features']]
        y = df[config['target']]

        # Divide los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state']
        )

        return X_train, X_test, y_train, y_test

    def train_model(self):
        """Entrena el modelo con los datos cargados"""
        X_train, X_test, y_train, y_test = self.load_data()
        config = self.config_manager.config
        features = config['features']

        # Selecciona el tipo de modelo según la configuración
        if config['model_type'] == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=config['random_state']
            )
        elif config['model_type'] == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=config['random_state']
            )
        else:
            # Por defecto, usa regresión lineal
            self.model = LinearRegression()

        # Entrena el modelo
        self.model.fit(X_train, y_train)

        # Evalúa el modelo
        y_pred = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

        # Obtiene importancia de características
        if hasattr(self.model, 'coef_'):
            for i, feature in enumerate(features):
                self.feature_importances[feature] = float(self.model.coef_[i])
        elif hasattr(self.model, 'feature_importances_'):
            for i, feature in enumerate(features):
                self.feature_importances[feature] = float(self.model.feature_importances_[i])

        print(f"Modelo entrenado con MAE: {self.mae:.2f} y R²: {self.r2:.2f}")

        return self.model

    def predict(self, input_data):
        """Realiza una predicción con el modelo entrenado"""
        if self.model is None:
            self.train_model()

        # Asegúrate de que input_data sea un array numpy
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Reshape si es necesario
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)

        return self.model.predict(input_data)[0]

    def get_feature_importances(self):
        """Devuelve la importancia de cada característica"""
        if self.model is None:
            self.train_model()

        return self.feature_importances

    def get_model_metrics(self):
        """Devuelve las métricas del modelo"""
        if self.model is None:
            self.train_model()

        return {
            'mae': self.mae,
            'r2': self.r2
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Uso del Singleton de configuración
    config1 = ConfigManager()
    config2 = ConfigManager()

    print("¿Son el mismo objeto?", config1 is config2)  # Debería ser True

    # Modificar la configuración
    config1.set('test_size', 0.2)
    print("Valor actualizado en config2:", config2.get('test_size'))  # Debería ser 0.2

    # Uso del Singleton del modelo
    model_manager1 = ModelManager()
    model_manager2 = ModelManager()

    print("¿Son el mismo objeto?", model_manager1 is model_manager2)  # Debería ser True

    # Entrenar el modelo
    model_manager1.train_model()

    # Realizar una predicción
    laptop_features = [
        2.2,    # Weight (kg)
        1,      # TypeName_Gaming
        0,      # TypeName_Notebook
        1920,   # screen_width
        1080,   # screen_height
        2.5,    # GHz
        8       # Ram
    ]

    precio_predicho = model_manager2.predict(laptop_features)
    print(f"Precio predicho: €{precio_predicho:.2f}")