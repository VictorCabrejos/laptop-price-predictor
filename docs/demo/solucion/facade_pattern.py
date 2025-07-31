"""
Patrón de Diseño: Facade
-----------------------
Este archivo muestra cómo se puede implementar el patrón Facade
para simplificar la interfaz del sistema de predicción de precios de laptops.

Versión Original (Spaguetti) vs Versión con Facade
"""

# ----------------- VERSIÓN ORIGINAL (SPAGUETTI) -----------------
"""
En esta versión, el cliente (API) tiene que interactuar directamente
con múltiples subsistemas, lo que genera complejidad y acoplamiento.
"""
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("laptop_price_predictor")

# Inicializar FastAPI
app_original = FastAPI()
templates = Jinja2Templates(directory="templates")
model = None
features = ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']

# Versión original con interacción directa con los subsistemas
@app.post("/predict_original", response_class=HTMLResponse)
async def predict_original(
    request: Request,
    weight: float = Form(...),
    is_gaming: int = Form(...),
    is_notebook: int = Form(...),
    screen_width: int = Form(...),
    screen_height: int = Form(...),
    ghz: float = Form(...),
    ram: int = Form(...)
):
    # El cliente interactúa directamente con múltiples subsistemas

    # 1. Validar datos manualmente
    if weight <= 0 or weight > 5:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Peso inválido: {weight}. Debe estar entre 0 y 5 kg."
        })

    # ...más validaciones

    # 2. Preparar datos para predicción
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])

    # 3. Verificar si el modelo está cargado
    if model is None:
        # 4. Cargar y preparar datos para el modelo
        df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')

        # Limpiar y transformar datos
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extraer resolución de pantalla
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(float)

        # Extraer velocidad de CPU
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        # Entrenar modelo
        X = df[features]
        y = df['Price_euros']
        global model
        model = LinearRegression()
        model.fit(X, y)

    # 5. Hacer predicción
    try:
        prediction = model.predict(input_data)[0]
        logger.info(f"Predicción exitosa: {prediction:.2f}€")
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Error al realizar la predicción: {str(e)}"
        })

    # 6. Buscar laptops similares
    df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
    df['price_diff'] = abs(df['Price_euros'] - prediction)
    similar = df.sort_values('price_diff').head(3)

    similar_laptops = []
    for _, row in similar.iterrows():
        similar_laptops.append({
            'company': row['Company'],
            'product': row['Product'],
            'price': float(row['Price_euros'])
        })

    # 7. Generar visualización
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(8, 4))
    companies = df.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)
    sns.barplot(x=companies.index, y=companies.values)
    plt.title('Precio Promedio por Fabricante')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/companies_price.png')

    # 8. Devolver respuesta
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "price": f"{prediction:,.2f}",
        "weight": weight,
        "ram": ram,
        "is_gaming": "Sí" if is_gaming == 1 else "No",
        "is_notebook": "Sí" if is_notebook == 1 else "No",
        "screen_resolution": f"{screen_width}x{screen_height}",
        "ghz": ghz,
        "similar_laptops": similar_laptops
    })


# ----------------- VERSIÓN CON PATRÓN FACADE -----------------
"""
En esta versión, creamos una fachada que encapsula toda la complejidad
y provee una interfaz simplificada para el cliente (API).
"""
import logging
import time
from typing import Dict, Any, List, Tuple
import os
import json
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel

# Modelos de datos
class LaptopFeatures(BaseModel):
    weight: float
    is_gaming: int
    is_notebook: int
    screen_width: int
    screen_height: int
    ghz: float
    ram: int

class SimilarLaptop(BaseModel):
    company: str
    product: str
    price: float

class PredictionResult(BaseModel):
    price: float
    similar_laptops: List[SimilarLaptop]
    currency: str = "EUR"

# La Fachada - Oculta toda la complejidad del sistema
class LaptopPredictionFacade:
    """
    Fachada que encapsula toda la complejidad del sistema de predicción de precios.
    Proporciona una interfaz simplificada para el cliente.
    """
    def __init__(self):
        self.logger = logging.getLogger("laptop_prediction_facade")
        self.model = None
        self.data_processor = None
        self.similar_finder = None
        self.metrics_calculator = None
        self.visualization_engine = None
        self.initialize()

    def initialize(self):
        """Inicializa todos los componentes necesarios"""
        self.logger.info("Inicializando componentes de la fachada")

        # Inicializar procesador de datos
        self.data_processor = self._create_data_processor()

        # Inicializar modelo
        self.model = self._create_model()

        # Inicializar buscador de laptops similares
        self.similar_finder = self._create_similar_finder()

        # Inicializar calculador de métricas
        self.metrics_calculator = self._create_metrics_calculator()

        # Inicializar motor de visualización
        self.visualization_engine = self._create_visualization_engine()

        self.logger.info("Fachada inicializada correctamente")

    def _create_data_processor(self):
        """Crea el componente de procesamiento de datos"""
        return {
            'features': ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram'],
            'dataset_path': 'laptop_price.csv'
        }

    def _create_model(self):
        """Crea y entrena el modelo (o lo carga si existe)"""
        from sklearn.linear_model import LinearRegression
        import pickle
        import os

        model_path = 'laptop_price_model.pkl'

        # Si existe modelo previo, cargarlo
        if os.path.exists(model_path):
            self.logger.info("Cargando modelo existente")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model

        # Si no existe, entrenar nuevo
        self.logger.info("Entrenando nuevo modelo")

        # Cargar y procesar datos
        processed_df = self._load_and_process_data()

        # Dividir en entrenamiento/prueba
        from sklearn.model_selection import train_test_split
        X = processed_df[self.data_processor['features']]
        y = processed_df['Price_euros']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Guardar modelo
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        return model

    def _load_and_process_data(self):
        """Carga y preprocesa los datos"""
        df = pd.read_csv(self.data_processor['dataset_path'], encoding='ISO-8859-1')

        # Limpiar y transformar datos
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

        # Extraer resolución de pantalla
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(float)

        # Extraer velocidad de CPU
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        return df

    def _create_similar_finder(self):
        """Crea el componente para buscar laptops similares"""
        return {
            'data': pd.read_csv(self.data_processor['dataset_path'], encoding='ISO-8859-1')
        }

    def _create_metrics_calculator(self):
        """Crea el componente para calcular métricas del modelo"""
        from sklearn.metrics import mean_absolute_error, r2_score
        return {
            'mae_func': mean_absolute_error,
            'r2_func': r2_score
        }

    def _create_visualization_engine(self):
        """Crea el componente de visualizaciones"""
        os.makedirs("static", exist_ok=True)
        return {
            'output_dir': 'static'
        }

    def predict_price(self, features: dict) -> dict:
        """
        Método principal de la fachada para predicción.
        Coordina todos los subsistemas para realizar la predicción.
        """
        try:
            # 1. Validar datos de entrada
            self._validate_input(features)

            # 2. Preparar datos para predicción
            input_data = self._prepare_prediction_input(features)

            # 3. Realizar predicción
            price = float(self.model.predict(input_data)[0])

            # 4. Buscar laptops similares
            similar = self._find_similar_laptops(price)

            # 5. Generar visualización (si es necesario)
            self._generate_visualizations(features, price)

            # 6. Devolver resultado completo
            return {
                'success': True,
                'price': price,
                'similar_laptops': similar,
                'currency': 'EUR'
            }

        except ValueError as e:
            self.logger.error(f"Error en validación: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        except Exception as e:
            self.logger.exception("Error inesperado en predicción")
            return {
                'success': False,
                'error': 'Error interno del servidor'
            }

    def _validate_input(self, features):
        """Valida los datos de entrada"""
        if features['weight'] <= 0 or features['weight'] > 5:
            raise ValueError(f"Peso inválido: {features['weight']}. Debe estar entre 0 y 5 kg.")

        if features['screen_width'] < 800 or features['screen_height'] < 600:
            raise ValueError(f"Resolución inválida: {features['screen_width']}x{features['screen_height']}")

        if features['ram'] <= 0 or features['ram'] > 128:
            raise ValueError(f"RAM inválida: {features['ram']}. Debe estar entre 1 y 128 GB.")

        if features['ghz'] <= 0 or features['ghz'] > 6:
            raise ValueError(f"Frecuencia de CPU inválida: {features['ghz']}. Debe estar entre 0.1 y 6 GHz.")

    def _prepare_prediction_input(self, features):
        """Prepara los datos para la predicción"""
        return np.array([[
            features['weight'],
            features['is_gaming'],
            features['is_notebook'],
            features['screen_width'],
            features['screen_height'],
            features['ghz'],
            features['ram']
        ]])

    def _find_similar_laptops(self, price, count=3):
        """Encuentra laptops similares por precio"""
        df = self.similar_finder['data']
        df['price_diff'] = abs(df['Price_euros'] - price)
        similar = df.sort_values('price_diff').head(count)

        result = []
        for _, row in similar.iterrows():
            result.append({
                'company': row['Company'],
                'product': row['Product'],
                'price': float(row['Price_euros'])
            })
        return result

    def _generate_visualizations(self, features, price):
        """Genera visualizaciones relacionadas con la predicción"""
        df = self.similar_finder['data']

        # Visualización de precios por fabricante
        plt.figure(figsize=(8, 4))
        companies = df.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)
        sns.barplot(x=companies.index, y=companies.values)
        plt.title('Precio Promedio por Fabricante')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.visualization_engine['output_dir']}/companies_price.png")

    def get_model_metrics(self):
        """Obtiene métricas del modelo para visualización"""
        # Cargar y procesar datos
        df = self._load_and_process_data()

        # Dividir datos
        X = df[self.data_processor['features']]
        y = df['Price_euros']

        # Predecir
        y_pred = self.model.predict(X)

        # Calcular métricas
        mae = self.metrics_calculator['mae_func'](y, y_pred)
        r2 = self.metrics_calculator['r2_func'](y, y_pred)

        # Obtener importancia de características
        importances = {}
        for i, feature in enumerate(self.data_processor['features']):
            importances[feature] = float(self.model.coef_[i])

        return {
            'mae': mae,
            'r2': r2,
            'feature_importances': importances
        }

# Demo para uso con FastAPI
app_facade = FastAPI()
templates = Jinja2Templates(directory="templates")

# Crear nuestra fachada
facade = LaptopPredictionFacade()

# Endpoint API JSON
@app_facade.post("/api/predict")
async def predict_api(laptop: LaptopFeatures):
    result = facade.predict_price(laptop.dict())
    if result['success']:
        return result
    else:
        raise HTTPException(status_code=400, detail=result['error'])

# Endpoint web con HTML
@app_facade.post("/predict", response_class=HTMLResponse)
async def predict_web(
    request: Request,
    weight: float = Form(...),
    is_gaming: int = Form(...),
    is_notebook: int = Form(...),
    screen_width: int = Form(...),
    screen_height: int = Form(...),
    ghz: float = Form(...),
    ram: int = Form(...)
):
    # Preparar datos para la fachada
    features = {
        'weight': weight,
        'is_gaming': is_gaming,
        'is_notebook': is_notebook,
        'screen_width': screen_width,
        'screen_height': screen_height,
        'ghz': ghz,
        'ram': ram
    }

    # Usar la fachada para hacer predicción
    result = facade.predict_price(features)

    if not result['success']:
        # Manejar error
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": result['error']
        })

    # Renderizar template con resultados
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "price": f"{result['price']:,.2f}",
        "weight": weight,
        "ram": ram,
        "is_gaming": "Sí" if is_gaming == 1 else "No",
        "is_notebook": "Sí" if is_notebook == 1 else "No",
        "screen_resolution": f"{screen_width}x{screen_height}",
        "ghz": ghz,
        "similar_laptops": result['similar_laptops']
    })

# Endpoint para información del modelo
@app_facade.get("/model_info", response_class=HTMLResponse)
async def model_info(request: Request):
    metrics = facade.get_model_metrics()

    # Preparar datos para la plantilla
    feature_data = []
    for feature, importance in metrics['feature_importances'].items():
        feature_data.append({
            "feature": feature,
            "importance": f"{importance:.4f}"
        })

    return templates.TemplateResponse("model_info.html", {
        "request": request,
        "mae": f"{metrics['mae']:.2f}",
        "r2": f"{metrics['r2']:.4f}",
        "feature_importances": feature_data
    })


"""
Demo simplificado para ejecutar y demostrar la diferencia
"""
if __name__ == "__main__":
    # Demostración simplificada
    print("=== Demo del Patrón Facade ===")

    # Caso de uso simplificado para demostración
    print("\nDemo 1: Acceso a subsistemas sin Facade (código cliente complejo)")
    print("- Cliente debe conocer cada subsistema")
    print("- Cliente debe coordinar los subsistemas")
    print("- Cliente debe manejar errores de cada subsistema")
    print("- Ejemplo: Ver función predict_original() con múltiples subsistemas acoplados\n")

    # Demo con Facade
    print("\nDemo 2: Acceso a subsistemas con Facade")
    facade_demo = LaptopPredictionFacade()

    # Caso de prueba simple
    laptop_features = {
        'weight': 1.5,
        'is_gaming': 0,
        'is_notebook': 1,
        'screen_width': 1920,
        'screen_height': 1080,
        'ghz': 2.8,
        'ram': 16
    }

    print("Cliente usando Facade:")
    print("facade.predict_price(laptop_features)")

    result = facade_demo.predict_price(laptop_features)
    print(f"Resultado: {result['price']:.2f}€")
    print(f"Laptops similares: {len(result['similar_laptops'])} encontradas")

    print("\nMétricas del modelo a través de la fachada:")
    metrics = facade_demo.get_model_metrics()
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")

    print("\nVentajas del patrón Facade:")
    print("1. Simplificación: Interfaz simplificada a un subsistema complejo")
    print("2. Desacoplamiento: Cliente desacoplado de los subsistemas")
    print("3. Mantenimiento: Cambios en subsistemas no afectan al cliente")
    print("4. API unificada: Un solo punto de acceso para múltiples operaciones")