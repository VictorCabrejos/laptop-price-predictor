"""
Patrón de Diseño: Adapter
--------------------------
Este archivo muestra cómo se puede implementar el patrón Adapter
para hacer compatible el modelo de scikit-learn con la API de FastAPI.

Versión Original (Spaguetti) vs Versión con Adapter
"""

# ----------------- VERSIÓN ORIGINAL (SPAGUETTI) -----------------
"""
En esta versión, la predicción está directamente acoplada a la API y al modelo
"""
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import numpy as np

app = FastAPI()
model = None  # Se asume que se carga en algún punto

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
    # Prepare input for prediction - Acoplado directamente al formato del modelo
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])

    # Make prediction - Acoplado a la API de scikit-learn
    prediction = model.predict(input_data)[0]

    # Uso directo del resultado
    return {"price": prediction}


# ----------------- VERSIÓN CON PATRÓN ADAPTER -----------------
"""
En esta versión, usamos un Adapter para desacoplar la API del modelo
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression  # Asumimos que es el modelo que usamos
import pandas as pd
import numpy as np

# Definimos los modelos de datos de entrada y salida
class LaptopFeatures(BaseModel):
    weight: float
    is_gaming: int
    is_notebook: int
    screen_width: int
    screen_height: int
    ghz: float
    ram: int

class PredictionResult(BaseModel):
    price: float
    similar_laptops: list

# Adaptador para el modelo de scikit-learn
class LaptopPriceModelAdapter:
    """
    Adaptador que convierte entre la interfaz de la API web y el modelo ML
    """
    def __init__(self, model):
        self.model = model
        self.features = ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']

    def predict(self, laptop_data: dict) -> float:
        """
        Convierte los datos del formato de la API al formato que espera el modelo
        y realiza la predicción.
        """
        # Convertir al formato que espera el modelo
        input_array = np.array([[
            laptop_data["weight"],
            laptop_data["is_gaming"],
            laptop_data["is_notebook"],
            laptop_data["screen_width"],
            laptop_data["screen_height"],
            laptop_data["ghz"],
            laptop_data["ram"]
        ]])

        # Realizar predicción
        return self.model.predict(input_array)[0]

    def get_similar_laptops(self, price: float, count: int = 3) -> list:
        """
        Encuentra laptops similares basadas en precio.
        """
        # Lógica para encontrar laptops similares
        df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
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

# Demostración de uso con FastAPI
app_adapter = FastAPI()

# Función de inicialización (simplificada para el ejemplo)
def train_model():
    # ... código de entrenamiento existente
    return LinearRegression()  # Modelo entrenado

# Inicializar
model_sklearn = train_model()
adapter = LaptopPriceModelAdapter(model_sklearn)

# Endpoint de predicción utilizando el adaptador
@app.post("/api/predict", response_model=PredictionResult)
async def predict(laptop: LaptopFeatures):
    # Usar el adaptador para hacer la predicción
    price = adapter.predict(laptop.dict())

    # Obtener laptops similares a través del adaptador
    similar_laptops = adapter.get_similar_laptops(price)

    # Devolver resultado
    return PredictionResult(price=price, similar_laptops=similar_laptops)

# Versión web para retrocompatibilidad (ahora usa el adaptador)
@app.post("/predict", response_class=HTMLResponse)
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
    # Crear diccionario de características
    laptop_data = {
        "weight": weight,
        "is_gaming": is_gaming,
        "is_notebook": is_notebook,
        "screen_width": screen_width,
        "screen_height": screen_height,
        "ghz": ghz,
        "ram": ram
    }

    # Usar el adaptador para hacer la predicción
    price = adapter.predict(laptop_data)
    similar_laptops = adapter.get_similar_laptops(price)

    # Devolver resultado en el formato adecuado para la plantilla
    return {"price": price, "similar_laptops": similar_laptops}


"""
Demo simplificado para ejecutar y demostrar la diferencia
"""
if __name__ == "__main__":
    # Demostración simplificada
    print("=== Demo del Patrón Adapter ===")

    # Crear modelo simple para demo
    model_demo = LinearRegression()
    X = [[1, 0, 1, 1920, 1080, 2.4, 8], [2, 1, 0, 1366, 768, 3.2, 16]]
    y = [800, 1200]
    model_demo.fit(X, y)

    print("Versión original (acoplada):")
    input_data = np.array([[1.5, 0, 1, 1920, 1080, 2.8, 16]])
    prediction = model_demo.predict(input_data)[0]
    print(f"Predicción directa: {prediction:.2f} €")

    print("\nVersión con Adapter:")
    adapter_demo = LaptopPriceModelAdapter(model_demo)
    laptop_data = {
        "weight": 1.5,
        "is_gaming": 0,
        "is_notebook": 1,
        "screen_width": 1920,
        "screen_height": 1080,
        "ghz": 2.8,
        "ram": 16
    }
    price = adapter_demo.predict(laptop_data)
    print(f"Predicción con adapter: {price:.2f} €")