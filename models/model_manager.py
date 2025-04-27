import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

from models.config_manager import ConfigManager
from models.model_factory import ModelFactory

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

        return X_train, X_test, y_train, y_test, df

    def train_model(self):
        """Entrena el modelo con los datos cargados"""
        X_train, X_test, y_train, y_test, _ = self.load_data()
        config = self.config_manager.config
        features = config['features']

        # Usa el patrón Factory para crear el modelo según la configuración
        self.model = ModelFactory.create_model(config['model_type'])

        # Entrena el modelo
        self.model.train(X_train, y_train)

        # Evalúa el modelo
        result = self.model.evaluate(X_test, y_test)
        self.mae = result['mae']
        self.r2 = result['r2']
        y_pred = result.get('predictions', [])

        # Obtiene importancia de características
        self.feature_importances = self.model.get_feature_importances(features)

        # Genera gráficos para visualizar el rendimiento del modelo
        self._generate_plots(X_test, y_test, y_pred)

        print(f"Modelo entrenado con MAE: {self.mae:.2f} y R²: {self.r2:.2f}")

        return self.model

    def _generate_plots(self, X_test, y_test, y_pred):
        """Genera gráficos para visualizar el rendimiento del modelo"""
        config = self.config_manager.config
        static_dir = config['static_dir']

        # Generate a plot for the model
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Laptop Prices')
        plt.savefig(os.path.join(static_dir, 'prediction_plot.png'))
        plt.close()

        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        features_df = pd.DataFrame({
            'Feature': list(self.feature_importances.keys()),
            'Importance': list(self.feature_importances.values())
        })
        features_df = features_df.sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=features_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, 'feature_importance.png'))
        plt.close()

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

        # Si ya tenemos importancias calculadas, las devolvemos
        if self.feature_importances:
            return self.feature_importances

        # Si no, recalculamos las importancias usando las características de la configuración
        config = self.config_manager.config
        self.feature_importances = self.model.get_feature_importances(config['features'])
        return self.feature_importances

    def get_model_metrics(self):
        """Devuelve las métricas del modelo"""
        if self.model is None:
            self.train_model()

        return {
            'mae': self.mae,
            'r2': self.r2
        }

    def find_similar_laptops(self, price, count=3):
        """Encuentra laptops similares basadas en el precio"""
        _, _, _, _, df = self.load_data()
        df['price_diff'] = abs(df['Price_euros'] - price)
        similar = df.sort_values('price_diff').head(count)
        result = []
        for _, row in similar.iterrows():
            result.append({
                'company': row['Company'],
                'product': row['Product'],
                'price': f"{row['Price_euros']:,.2f}"
            })
        return result