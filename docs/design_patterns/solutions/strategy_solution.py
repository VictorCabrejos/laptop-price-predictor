import pandas as pd
import numpy as np
from preprocessors.data_preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split

class StandardPreprocessor(DataPreprocessor):
    """
    Implementación estándar del preprocesador de datos.
    Realiza las transformaciones básicas como en la implementación original.
    """

    def preprocess_training_data(self, df):
        # Crear una copia para no modificar el original
        df = df.copy()

        # Clean and transform data
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extract screen resolution
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(float)

        # Extract CPU speed
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        # Feature selection
        X = df[self.get_feature_names()]
        y = df['Price_euros']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        return X_train, X_test, y_train, y_test

    def preprocess_input_data(self, input_data):
        # Asumimos que input_data ya está en formato adecuado para predecir
        # El formato debe coincidir con get_feature_names()
        return input_data

    def get_feature_names(self):
        return ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']


class AdvancedPreprocessor(DataPreprocessor):
    """
    Implementación avanzada del preprocesador de datos.
    Incluye normalización de características y detección de valores atípicos.
    """

    def __init__(self):
        self.mean_values = None
        self.std_values = None
        self.outlier_threshold = 2.5  # Número de desviaciones estándar para considerar un valor atípico

    def preprocess_training_data(self, df):
        # Crear una copia para no modificar el original
        df = df.copy()

        # Preprocesamiento básico
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extract screen resolution
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(float)

        # Extract CPU speed
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        # Seleccionar características
        features = self.get_feature_names()
        X = df[features]

        # Guardar estadísticas para la normalización
        self.mean_values = X.mean()
        self.std_values = X.std()

        # Normalizar características (Z-score normalization)
        X = (X - self.mean_values) / self.std_values

        # Detectar y tratar valores atípicos
        X = self._handle_outliers(X)

        # Variable objetivo
        y = df['Price_euros']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        return X_train, X_test, y_train, y_test

    def _handle_outliers(self, X):
        """Reemplazar valores atípicos con los límites de las distribuciones"""
        X_copy = X.copy()

        for column in X_copy.columns:
            lower_bound = -self.outlier_threshold
            upper_bound = self.outlier_threshold

            # Reemplazar valores atípicos con los límites
            X_copy.loc[X_copy[column] < lower_bound, column] = lower_bound
            X_copy.loc[X_copy[column] > upper_bound, column] = upper_bound

        return X_copy

    def preprocess_input_data(self, input_data):
        """Preprocesa datos de entrada para la predicción usando las estadísticas guardadas"""
        if self.mean_values is None or self.std_values is None:
            raise ValueError("El preprocesador no ha sido entrenado. Ejecute preprocess_training_data primero.")

        # Convertir a DataFrame si es un array
        if isinstance(input_data, np.ndarray):
            input_data = pd.DataFrame(input_data, columns=self.get_feature_names())

        # Normalizar usando las estadísticas del conjunto de entrenamiento
        normalized_data = (input_data - self.mean_values) / self.std_values

        # Manejar valores atípicos
        normalized_data = self._handle_outliers(normalized_data)

        return normalized_data.values

    def get_feature_names(self):
        return ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']


class MinimalPreprocessor(DataPreprocessor):
    """
    Implementación minimalista del preprocesador de datos.
    Utiliza solo un subconjunto de características para la predicción.
    """

    def preprocess_training_data(self, df):
        # Crear una copia para no modificar el original
        df = df.copy()

        # Clean and transform data - solo las características más importantes
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extract CPU speed
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        # Feature selection - solo las más importantes
        X = df[self.get_feature_names()]
        y = df['Price_euros']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        return X_train, X_test, y_train, y_test

    def preprocess_input_data(self, input_data):
        # Para MinimalPreprocessor, solo necesitamos las columnas específicas
        # Asumimos que el input_data está en formato array con todas las características
        if isinstance(input_data, np.ndarray) and input_data.shape[1] > len(self.get_feature_names()):
            # Seleccionar solo las columnas que necesitamos
            # Esto es una simplificación y requeriría ajuste en una implementación real
            indices = [0, 1, 2, 5, 6]  # Índices correspondientes a Weight, TypeName_Gaming, TypeName_Notebook, GHz, Ram
            return input_data[:, indices]
        return input_data

    def get_feature_names(self):
        # Usar solo un subconjunto de características
        return ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'GHz', 'Ram']


# Ejemplo de contexto que utiliza el patrón Strategy
class ModelManager:
    """
    Clase que actúa como contexto para el patrón Strategy.
    Utiliza una estrategia de preprocesamiento para preparar los datos para el modelo.
    """

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or StandardPreprocessor()
        self.model = None
        self.feature_importances = {}

    def set_preprocessor(self, preprocessor):
        """Cambiar la estrategia de preprocesamiento en tiempo de ejecución"""
        self.preprocessor = preprocessor

    def train_model(self, data_path):
        """Entrenar el modelo usando la estrategia de preprocesamiento seleccionada"""
        from sklearn.linear_model import LinearRegression

        # Cargar datos
        df = pd.read_csv(data_path, encoding='ISO-8859-1')

        # Usar la estrategia de preprocesamiento
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_training_data(df)

        # Entrenar el modelo
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Guardar las importancias de las características
        for i, feature in enumerate(self.preprocessor.get_feature_names()):
            self.feature_importances[feature] = float(self.model.coef_[i])

        # Evaluar el modelo
        y_pred = self.model.predict(X_test)

        return X_test, y_test, y_pred

    def predict(self, input_data):
        """Realizar predicción usando el modelo entrenado y la estrategia de preprocesamiento"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecute train_model primero.")

        # Preprocesar los datos de entrada usando la estrategia actual
        processed_input = self.preprocessor.preprocess_input_data(input_data)

        # Realizar la predicción
        return self.model.predict(processed_input)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un gestor de modelo con una estrategia estándar
    manager = ModelManager(preprocessor=StandardPreprocessor())

    # Entrenar el modelo
    manager.train_model('laptop_price.csv')

    # Realizar una predicción
    sample_input = np.array([[2.1, 1, 0, 1920, 1080, 2.5, 16]])
    prediction = manager.predict(sample_input)
    print(f"Predicción con preprocesador estándar: ${prediction[0]:.2f}")

    # Cambiar a la estrategia avanzada
    manager.set_preprocessor(AdvancedPreprocessor())

    # Reentrenar el modelo con la nueva estrategia
    manager.train_model('laptop_price.csv')

    # Realizar predicción con la nueva estrategia
    prediction = manager.predict(sample_input)
    print(f"Predicción con preprocesador avanzado: ${prediction[0]:.2f}")