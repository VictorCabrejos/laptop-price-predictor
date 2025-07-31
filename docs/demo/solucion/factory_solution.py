# Implementación del Patrón Factory para el proyecto Laptop Price Predictor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Interfaces abstractas
class ModelInterface:
    """Interfaz para todos los modelos de predicción de precios"""

    def train(self, X_train, y_train):
        """Entrena el modelo con los datos proporcionados"""
        pass

    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        pass

    def evaluate(self, X_test, y_test):
        """Evalúa el rendimiento del modelo"""
        pass

    def get_feature_importances(self, feature_names):
        """Obtiene la importancia de las características"""
        pass


# Implementaciones concretas de modelos
class LinearRegressionModel(ModelInterface):
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
        return {'mae': self.mae, 'r2': self.r2}

    def get_feature_importances(self, feature_names):
        """Obtiene los coeficientes del modelo como importancia de características"""
        importances = {}
        for i, feature in enumerate(feature_names):
            importances[feature] = float(self.model.coef_[i])
        return importances

    def __str__(self):
        return "Regresión Lineal"


class RandomForestModel(ModelInterface):
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
        return {'mae': self.mae, 'r2': self.r2}

    def get_feature_importances(self, feature_names):
        """Obtiene la importancia de características del Random Forest"""
        importances = {}
        for i, feature in enumerate(feature_names):
            importances[feature] = float(self.model.feature_importances_[i])
        return importances

    def __str__(self):
        return "Random Forest"


class GradientBoostingModel(ModelInterface):
    """Modelo de Gradient Boosting para predicción de precios"""

    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.mae = None
        self.r2 = None

    def train(self, X_train, y_train):
        """Entrena el modelo de Gradient Boosting"""
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
        return {'mae': self.mae, 'r2': self.r2}

    def get_feature_importances(self, feature_names):
        """Obtiene la importancia de características del Gradient Boosting"""
        importances = {}
        for i, feature in enumerate(feature_names):
            importances[feature] = float(self.model.feature_importances_[i])
        return importances

    def __str__(self):
        return "Gradient Boosting"


class SVRModel(ModelInterface):
    """Modelo de Support Vector Regression para predicción de precios"""

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.mae = None
        self.r2 = None

    def train(self, X_train, y_train):
        """Entrena el modelo SVR"""
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
        return {'mae': self.mae, 'r2': self.r2}

    def get_feature_importances(self, feature_names):
        """SVR no proporciona importancia de características directamente"""
        return {feature: 0.0 for feature in feature_names}

    def __str__(self):
        return f"SVR (kernel={self.model.kernel})"


# Factory para crear modelos
class ModelFactory:
    """Factory para crear diferentes tipos de modelos de predicción"""

    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Crea y devuelve un modelo según el tipo especificado

        Args:
            model_type (str): Tipo de modelo a crear ('linear', 'random_forest', 'gradient_boosting', 'svr')
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            ModelInterface: Una instancia del modelo solicitado

        Raises:
            ValueError: Si el tipo de modelo no es reconocido
        """
        if model_type == 'linear':
            return LinearRegressionModel()
        elif model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            random_state = kwargs.get('random_state', 42)
            return RandomForestModel(n_estimators, random_state)
        elif model_type == 'gradient_boosting':
            n_estimators = kwargs.get('n_estimators', 100)
            learning_rate = kwargs.get('learning_rate', 0.1)
            random_state = kwargs.get('random_state', 42)
            return GradientBoostingModel(n_estimators, learning_rate, random_state)
        elif model_type == 'svr':
            kernel = kwargs.get('kernel', 'rbf')
            C = kwargs.get('C', 1.0)
            epsilon = kwargs.get('epsilon', 0.1)
            return SVRModel(kernel, C, epsilon)
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {model_type}")


# Clase de servicio para la predicción
class PredictionService:
    """
    Servicio para entrenar modelos, evaluar y hacer predicciones
    Utiliza el patrón Factory para crear los modelos
    """

    def __init__(self, data_path='laptop_price.csv', encoding='ISO-8859-1'):
        self.data_path = data_path
        self.encoding = encoding
        self.features = ['Weight', 'TypeName_Gaming', 'TypeName_Notebook',
                        'screen_width', 'screen_height', 'GHz', 'Ram']
        self.target = 'Price_euros'
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """Carga y preprocesa los datos desde el archivo CSV"""
        # Carga el dataset
        df = pd.read_csv(self.data_path, encoding=self.encoding)

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
        X = df[self.features]
        y = df[self.target]

        # Divide los datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

    def train_model(self, model_type, **kwargs):
        """
        Entrena un modelo específico utilizando el Factory

        Args:
            model_type (str): Tipo de modelo a entrenar
            **kwargs: Parámetros para la creación del modelo

        Returns:
            ModelInterface: El modelo entrenado
        """
        # Crear modelo utilizando el Factory
        model = ModelFactory.create_model(model_type, **kwargs)

        # Entrenar el modelo
        model.train(self.X_train, self.y_train)

        # Evaluar el modelo
        metrics = model.evaluate(self.X_test, self.y_test)
        print(f"Modelo {model} entrenado con MAE: {metrics['mae']:.2f} y R²: {metrics['r2']:.2f}")

        # Guardar el modelo
        self.models[model_type] = model

        # Generar gráficos
        self._generate_plots(model, model_type)

        return model

    def predict(self, model_type, input_data):
        """
        Realiza una predicción utilizando el modelo especificado

        Args:
            model_type (str): Tipo de modelo a utilizar
            input_data (list or array): Datos de entrada para la predicción

        Returns:
            float: Precio predicho
        """
        # Verificar si el modelo existe o debe ser entrenado
        if model_type not in self.models:
            self.train_model(model_type)

        model = self.models[model_type]

        # Asegúrate de que input_data sea un array numpy
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Reshape si es necesario
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)

        # Realizar la predicción
        return model.predict(input_data)[0]

    def get_feature_importances(self, model_type):
        """
        Obtiene la importancia de las características para un modelo específico

        Args:
            model_type (str): Tipo de modelo

        Returns:
            dict: Diccionario con la importancia de cada característica
        """
        # Verificar si el modelo existe o debe ser entrenado
        if model_type not in self.models:
            self.train_model(model_type)

        model = self.models[model_type]

        # Obtener importancia de características
        return model.get_feature_importances(self.features)

    def compare_models(self, model_types=None, **kwargs):
        """
        Compara el rendimiento de diferentes modelos

        Args:
            model_types (list, optional): Lista de tipos de modelos a comparar.
                       Si es None, compara todos los modelos disponibles.
            **kwargs: Parámetros para la creación de los modelos

        Returns:
            dict: Métricas de rendimiento de cada modelo
        """
        if model_types is None:
            model_types = ['linear', 'random_forest', 'gradient_boosting', 'svr']

        results = {}

        # Entrenar y evaluar cada modelo
        for model_type in model_types:
            model = self.train_model(model_type, **kwargs)
            metrics = model.evaluate(self.X_test, self.y_test)
            results[model_type] = {
                'model': model,
                'metrics': metrics
            }

        # Mostrar resultados
        print("\nComparación de modelos:")
        for model_type, result in results.items():
            print(f"{model_type}: MAE = {result['metrics']['mae']:.2f}, R² = {result['metrics']['r2']:.2f}")

        # Generar gráfico de comparación
        self._generate_comparison_plot(results)

        return results

    def _generate_plots(self, model, model_type):
        """Genera gráficos para visualizar el rendimiento del modelo"""
        # Crear directorio para gráficos si no existe
        os.makedirs('static', exist_ok=True)

        # Predicciones vs reales
        y_pred = model.predict(self.X_test)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--')
        plt.xlabel('Precio real')
        plt.ylabel('Precio predicho')
        plt.title(f'Predicción vs Real - {model}')
        plt.savefig(f'static/prediction_plot_{model_type}.png')

        # Importancia de características
        if hasattr(model, 'get_feature_importances'):
            importances = model.get_feature_importances(self.features)
            features_df = pd.DataFrame({
                'Feature': list(importances.keys()),
                'Importance': list(importances.values())
            })
            features_df = features_df.sort_values('Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=features_df)
            plt.title(f'Importancia de características - {model}')
            plt.tight_layout()
            plt.savefig(f'static/feature_importance_{model_type}.png')

    def _generate_comparison_plot(self, results):
        """Genera gráficos para comparar diferentes modelos"""
        # Métricas de rendimiento
        models = list(results.keys())
        mae_values = [results[m]['metrics']['mae'] for m in models]
        r2_values = [results[m]['metrics']['r2'] for m in models]

        # Gráfico de MAE
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(x=models, y=mae_values)
        plt.title('Error Absoluto Medio (MAE)')
        plt.ylabel('MAE (menor es mejor)')
        plt.xticks(rotation=45)

        # Gráfico de R²
        plt.subplot(1, 2, 2)
        sns.barplot(x=models, y=r2_values)
        plt.title('Coeficiente de Determinación (R²)')
        plt.ylabel('R² (mayor es mejor)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('static/model_comparison.png')


# Ejemplo de uso
if __name__ == "__main__":
    # Crear servicio de predicción
    service = PredictionService()

    # Entrenar algunos modelos
    service.train_model('linear')
    service.train_model('random_forest', n_estimators=150)

    # Comparar modelos
    results = service.compare_models(['linear', 'random_forest', 'gradient_boosting'])

    # Realizar predicción con diferentes modelos
    laptop_features = [
        2.2,    # Weight (kg)
        1,      # TypeName_Gaming
        0,      # TypeName_Notebook
        1920,   # screen_width
        1080,   # screen_height
        2.5,    # GHz
        8       # Ram
    ]

    price_linear = service.predict('linear', laptop_features)
    price_rf = service.predict('random_forest', laptop_features)
    price_gb = service.predict('gradient_boosting', laptop_features)

    print("\nPredicciones para la misma laptop usando diferentes modelos:")
    print(f"Regresión Lineal: €{price_linear:.2f}")
    print(f"Random Forest: €{price_rf:.2f}")
    print(f"Gradient Boosting: €{price_gb:.2f}")