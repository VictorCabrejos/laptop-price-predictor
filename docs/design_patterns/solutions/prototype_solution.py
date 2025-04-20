# Implementación del Patrón Prototype para el proyecto Laptop Price Predictor
import copy
import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class ModelConfig:
    """
    Clase que actúa como prototipo para configuraciones de modelos.
    Permite crear y clonar configuraciones para diferentes tipos de modelos.
    """

    def __init__(self, model_type='linear_regression', params=None, features=None):
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
        elif model_type == 'gradient_boosting':
            self.params.update({
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 1.0
            })
        elif model_type == 'linear_regression':
            self.params.update({
                'fit_intercept': True
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

    def create_model(self):
        """Crea una instancia del modelo basado en la configuración actual"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth'),
                min_samples_split=self.params.get('min_samples_split', 2),
                min_samples_leaf=self.params.get('min_samples_leaf', 1),
                random_state=self.params.get('random_state', 42)
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=self.params.get('n_estimators', 100),
                learning_rate=self.params.get('learning_rate', 0.1),
                max_depth=self.params.get('max_depth', 3),
                subsample=self.params.get('subsample', 1.0),
                random_state=self.params.get('random_state', 42)
            )
        else:
            # Por defecto, usa regresión lineal
            return LinearRegression(
                fit_intercept=self.params.get('fit_intercept', True)
            )


class ModelPrototypeRegistry:
    """
    Registro de prototipos de configuraciones de modelos.
    Permite almacenar y recuperar configuraciones predefinidas.
    """

    def __init__(self):
        self.prototypes = {}
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Inicializa configuraciones predeterminadas"""
        # Configuración básica de regresión lineal
        self.prototypes['basic_linear'] = ModelConfig(
            model_type='linear_regression',
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

        # Configuración para Gradient Boosting optimizado
        self.prototypes['optimized_gb'] = ModelConfig(
            model_type='gradient_boosting',
            params={
                'n_estimators': 120,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
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
        print("Prototipos disponibles:")
        for name, prototype in self.prototypes.items():
            print(f" - {name}: {prototype}")


class LaptopPricePredictor:
    """
    Clase principal que utiliza prototipos de configuración para
    entrenar y evaluar modelos de predicción de precios de laptops.
    """

    def __init__(self, config=None, data_path='laptop_price.csv'):
        """
        Inicializa el predictor con una configuración y datos

        Args:
            config (ModelConfig): Configuración del modelo
            data_path (str): Ruta al archivo CSV de datos
        """
        self.data_path = data_path

        # Si no se proporciona configuración, usa la predeterminada
        if config is None:
            self.config = ModelConfig()
        else:
            self.config = config

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Carga y preprocesa los datos según la configuración"""
        # Carga el dataset
        df = pd.read_csv(
            self.data_path,
            encoding=self.config.preprocessing['encoding']
        )

        # Limpia y transforma los datos
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)

        # One-hot encoding para características categóricas
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extrae la resolución de pantalla
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(
            r'(\d{3,4})x(\d{3,4})'
        ).astype(float)

        # Extrae la velocidad del CPU
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        # Manejo de valores faltantes según configuración
        if self.config.preprocessing['handle_missing'] == 'median':
            for col in self.config.features:
                if col in df.columns and df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)

        # Eliminar valores atípicos si está configurado
        if self.config.preprocessing['drop_outliers']:
            for col in self.config.features:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

        # Verificar que todas las características existen en el DataFrame
        valid_features = [f for f in self.config.features if f in df.columns]

        if len(valid_features) < len(self.config.features):
            missing = set(self.config.features) - set(valid_features)
            print(f"Advertencia: Algunas características no están disponibles: {missing}")
            self.config.features = valid_features

        # Selección de características y target
        X = df[self.config.features]
        y = df['Price_euros']

        # Divide los datos según la configuración
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.params['test_size'],
            random_state=self.config.params['random_state']
        )

        print(f"Datos cargados: {X.shape[0]} filas, {X.shape[1]} características")
        return self

    def train(self):
        """Entrena el modelo según la configuración"""
        # Verifica si los datos están cargados
        if self.X_train is None:
            self.load_data()

        # Crea el modelo a partir de la configuración
        self.model = self.config.create_model()

        # Entrena el modelo
        self.model.fit(self.X_train, self.y_train)

        print(f"Modelo {self.config.model_type} entrenado con éxito.")
        return self

    def evaluate(self):
        """Evalúa el modelo y muestra los resultados"""
        if self.model is None:
            print("Error: Primero debes entrenar el modelo.")
            return None

        # Realiza predicciones en el conjunto de prueba
        y_pred = self.model.predict(self.X_test)

        # Calcula métricas
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Evaluación del modelo {self.config.model_type}:")
        print(f" - Error Absoluto Medio (MAE): €{mae:.2f}")
        print(f" - Coeficiente de Determinación (R²): {r2:.4f}")

        # Genera visualizaciones
        self._generate_plots(y_pred)

        return {'mae': mae, 'r2': r2, 'predictions': y_pred}

    def predict(self, features):
        """Realiza una predicción para nuevas características"""
        if self.model is None:
            print("Error: Primero debes entrenar el modelo.")
            return None

        # Convierte a array numpy si es necesario
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        # Reshape si es necesario
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Verifica que el número de características sea el correcto
        if features.shape[1] != len(self.config.features):
            print(f"Error: Se esperaban {len(self.config.features)} características, "
                  f"pero se recibieron {features.shape[1]}")
            return None

        # Realiza la predicción
        prediction = self.model.predict(features)[0]
        print(f"Precio predicho: €{prediction:.2f}")

        return prediction

    def _generate_plots(self, y_pred):
        """Genera visualizaciones según la configuración"""
        # Crea el directorio para guardar imágenes si no existe
        os.makedirs(self.config.visualization['save_path'], exist_ok=True)

        # Configuración de la visualización
        viz_config = self.config.visualization
        plt.figure(figsize=viz_config['figsize'])

        # Gráfico de predicciones vs valores reales
        sns.scatterplot(
            x=self.y_test,
            y=y_pred,
            color=viz_config['color'],
            alpha=viz_config['alpha']
        )
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            'k--'
        )
        plt.xlabel('Precio real (€)')
        plt.ylabel('Precio predicho (€)')
        plt.title(
            f'Predicción vs Real - {self.config.model_type}',
            fontsize=viz_config['title_fontsize']
        )

        # Guarda el gráfico
        filename = f"{self.config.model_type}_prediction_plot.png"
        save_path = os.path.join(viz_config['save_path'], filename)
        plt.savefig(save_path)
        print(f"Gráfico guardado en {save_path}")

        # Si tiene importancia de características, muestra un gráfico
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()

    def _plot_feature_importance(self):
        """Genera un gráfico de importancia de características"""
        viz_config = self.config.visualization
        plt.figure(figsize=viz_config['figsize'])

        # Crea un DataFrame con la importancia
        importances = {}
        for i, feature in enumerate(self.config.features):
            importances[feature] = self.model.feature_importances_[i]

        # Ordena las características por importancia
        features_df = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importance': list(importances.values())
        }).sort_values('Importance', ascending=False)

        # Genera el gráfico
        sns.barplot(x='Importance', y='Feature', data=features_df)
        plt.title(
            f'Importancia de características - {self.config.model_type}',
            fontsize=viz_config['title_fontsize']
        )
        plt.tight_layout()

        # Guarda el gráfico
        filename = f"{self.config.model_type}_feature_importance.png"
        save_path = os.path.join(viz_config['save_path'], filename)
        plt.savefig(save_path)
        print(f"Gráfico de importancia guardado en {save_path}")

    def save_model_config(self, filepath):
        """Guarda la configuración actual del modelo"""
        self.config.save(filepath)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un registro de prototipos
    registry = ModelPrototypeRegistry()

    # Mostrar los prototipos disponibles
    registry.list_prototypes()

    # Obtener un prototipo del registro
    rf_config = registry.get('optimized_rf')

    # Modificar el prototipo sin afectar al original
    rf_config.params['n_estimators'] = 200
    rf_config.preprocessing['drop_outliers'] = True
    rf_config.visualization['color'] = 'green'

    # Crear un predictor con la configuración clonada
    predictor = LaptopPricePredictor(config=rf_config)

    # Entrenar y evaluar el modelo
    predictor.load_data().train().evaluate()

    # Realizar una predicción con nuevas características
    laptop_features = [
        2.2,    # Weight (kg)
        1,      # TypeName_Gaming
        0,      # TypeName_Notebook
        1920,   # screen_width
        1080,   # screen_height
    ]

    predictor.predict(laptop_features)

    # Guardar la configuración personalizada
    predictor.save_model_config('custom_rf_config.json')

    print("\n--- Demostración de clonación ---")

    # Demostrar que el prototipo original no ha sido modificado
    original_rf = registry.get('optimized_rf')
    print(f"Prototipo original - n_estimators: {original_rf.params['n_estimators']}")
    print(f"Configuración modificada - n_estimators: {rf_config.params['n_estimators']}")

    # Crear una nueva configuración personalizada
    high_precision_config = ModelConfig(
        model_type='gradient_boosting',
        params={
            'n_estimators': 300,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.7
        }
    )

    # Cambiar la configuración de visualización
    high_precision_config.visualization['color'] = 'purple'
    high_precision_config.visualization['alpha'] = 0.8

    # Registrar el nuevo prototipo
    registry.register('high_precision', high_precision_config)

    # Obtener una copia del prototipo y usarla
    hp_config = registry.get('high_precision')
    predictor2 = LaptopPricePredictor(config=hp_config)
    predictor2.load_data().train().evaluate()