"""
Patrón de Diseño: Decorator
---------------------------
Este archivo muestra cómo se puede implementar el patrón Decorator
para añadir funcionalidades adicionales a la predicción de precios de laptops.

Versión Original (Spaguetti) vs Versión con Decorator
"""

# ----------------- VERSIÓN ORIGINAL (SPAGUETTI) -----------------
"""
En esta versión, la predicción no tiene separación de responsabilidades,
mezclando validación, logging, y la predicción en un solo método.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# Modelo scikit-learn simple para la demostración
def create_simple_model():
    model = LinearRegression()
    X = [[1, 0, 1, 1920, 1080, 2.4, 8], [2, 1, 0, 1366, 768, 3.2, 16]]
    y = [800, 1200]
    model.fit(X, y)
    return model

# Versión original con todo mezclado
def predict_price_original(model, weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram):
    # Validación mezclada con la lógica de negocio
    if weight <= 0 or weight > 5:
        raise ValueError(f"Peso inválido: {weight}. Debe estar entre 0 y 5 kg.")

    if screen_width < 800 or screen_height < 600:
        raise ValueError(f"Resolución inválida: {screen_width}x{screen_height}")

    if ram <= 0 or ram > 128:
        raise ValueError(f"RAM inválida: {ram}. Debe estar entre 1 y 128 GB.")

    if ghz <= 0 or ghz > 6:
        raise ValueError(f"Frecuencia de CPU inválida: {ghz}. Debe estar entre 0.1 y 6 GHz.")

    # Logging mezclado con la lógica de negocio
    print(f"Iniciando predicción para laptop: {weight}kg, Gaming: {is_gaming}, Notebook: {is_notebook}, "
          f"Resolución: {screen_width}x{screen_height}, CPU: {ghz}GHz, RAM: {ram}GB")

    start_time = time.time()

    # Preparar datos y predecir
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])
    result = model.predict(input_data)[0]

    # Más logging mezclado
    elapsed_time = time.time() - start_time
    print(f"Predicción completada en {elapsed_time:.4f} segundos. Resultado: {result:.2f}€")

    return result


# ----------------- VERSIÓN CON PATRÓN DECORATOR -----------------
"""
En esta versión, usamos el patrón Decorator para separar responsabilidades
y poder añadir funcionalidades de forma modular.
"""
import logging
import time
from typing import Dict, Any

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("laptop_price_predictor")

# Componente base - define la interfaz
class PricePredictor:
    """Interfaz base para los predictores de precios"""
    def predict(self, features: Dict[str, Any]) -> float:
        """Predice el precio basado en las características de la laptop"""
        pass

# Implementación concreta del predictor
class SklearnPredictor(PricePredictor):
    """Implementación concreta que usa un modelo scikit-learn"""
    def __init__(self, model):
        self.model = model

    def predict(self, features: Dict[str, Any]) -> float:
        # Extraer características en el orden correcto
        input_data = np.array([[
            features['weight'],
            features['is_gaming'],
            features['is_notebook'],
            features['screen_width'],
            features['screen_height'],
            features['ghz'],
            features['ram']
        ]])
        return float(self.model.predict(input_data)[0])

# Decorador base
class PredictorDecorator(PricePredictor):
    """Clase base para todos los decoradores de predictores"""
    def __init__(self, predictor: PricePredictor):
        self._predictor = predictor

    def predict(self, features: Dict[str, Any]) -> float:
        return self._predictor.predict(features)

# Decorador para validación de datos
class ValidationDecorator(PredictorDecorator):
    """Decorador que añade validación de datos a cualquier predictor"""
    def predict(self, features: Dict[str, Any]) -> float:
        # Validar peso
        if features['weight'] <= 0 or features['weight'] > 5:
            raise ValueError(f"Peso inválido: {features['weight']}. Debe estar entre 0 y 5 kg.")

        # Validar resolución
        if features['screen_width'] < 800 or features['screen_height'] < 600:
            raise ValueError(f"Resolución inválida: {features['screen_width']}x{features['screen_height']}")

        # Validar RAM
        if features['ram'] <= 0 or features['ram'] > 128:
            raise ValueError(f"RAM inválida: {features['ram']}. Debe estar entre 1 y 128 GB.")

        # Validar GHz
        if features['ghz'] <= 0 or features['ghz'] > 6:
            raise ValueError(f"Frecuencia de CPU inválida: {features['ghz']}. Debe estar entre 0.1 y 6 GHz.")

        # Si pasa todas las validaciones, continuar
        return self._predictor.predict(features)

# Decorador para logging
class LoggingDecorator(PredictorDecorator):
    """Decorador que añade registro de eventos a cualquier predictor"""
    def predict(self, features: Dict[str, Any]) -> float:
        logger.info(f"Iniciando predicción con características: {features}")
        start_time = time.time()

        result = self._predictor.predict(features)

        elapsed_time = time.time() - start_time
        logger.info(f"Predicción completada en {elapsed_time:.4f} segundos. Resultado: {result:.2f}€")

        return result

# Decorador para caché
class CacheDecorator(PredictorDecorator):
    """Decorador que añade caché a cualquier predictor"""
    def __init__(self, predictor: PricePredictor):
        super().__init__(predictor)
        self._cache = {}

    def predict(self, features: Dict[str, Any]) -> float:
        # Crear key para el caché (simplificado para el ejemplo)
        cache_key = str(sorted(features.items()))

        # Verificar si ya existe en caché
        if cache_key in self._cache:
            logger.info(f"Usando resultado en caché para: {features}")
            return self._cache[cache_key]

        # Si no está en caché, calcular y almacenar
        result = self._predictor.predict(features)
        self._cache[cache_key] = result

        return result

# Demo para uso con FastAPI (simplificado)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class LaptopFeatures(BaseModel):
    weight: float
    is_gaming: int
    is_notebook: int
    screen_width: int
    screen_height: int
    ghz: float
    ram: int

app = FastAPI()

# Crear predictor configurado con decoradores
def create_decorated_predictor():
    model = create_simple_model()

    # Crear predictor base y aplicar decoradores
    base_predictor = SklearnPredictor(model)

    # Aplicar decoradores en orden específico
    predictor = ValidationDecorator(base_predictor)  # Primero validamos
    predictor = LoggingDecorator(predictor)  # Luego hacemos logging
    predictor = CacheDecorator(predictor)  # Finalmente usamos caché

    return predictor

predictor = create_decorated_predictor()

@app.post("/predict")
async def predict_price(laptop: LaptopFeatures):
    try:
        price = predictor.predict(laptop.dict())
        return {"price": price}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


"""
Demo simplificado para ejecutar y demostrar la diferencia
"""
if __name__ == "__main__":
    print("=== Demo del Patrón Decorator ===")

    # Crear modelo simple para demo
    model = create_simple_model()

    print("Versión Original (sin decoradores):")
    try:
        precio = predict_price_original(model, 1.5, 0, 1, 1920, 1080, 2.8, 16)
        print(f"Resultado: {precio:.2f}€")

        # Esto debería fallar
        print("\nProbando con datos inválidos:")
        precio_invalido = predict_price_original(model, -1, 0, 1, 1920, 1080, 2.8, 16)
    except ValueError as e:
        print(f"Error (esperado): {e}")

    # Demo con decoradores
    print("\nVersión con Decoradores:")

    # Configuración incremental para demostrar la flexibilidad
    base_predictor = SklearnPredictor(model)
    print("\n1. Solo predictor base:")
    laptop_data = {
        "weight": 1.5,
        "is_gaming": 0,
        "is_notebook": 1,
        "screen_width": 1920,
        "screen_height": 1080,
        "ghz": 2.8,
        "ram": 16
    }
    price = base_predictor.predict(laptop_data)
    print(f"Resultado: {price:.2f}€")

    print("\n2. Con validación:")
    validated_predictor = ValidationDecorator(base_predictor)
    try:
        price = validated_predictor.predict(laptop_data)
        print(f"Resultado: {price:.2f}€")

        # Probando validación
        print("Probando validación con datos incorrectos:")
        laptop_data_invalid = laptop_data.copy()
        laptop_data_invalid["weight"] = -1  # Peso inválido
        validated_predictor.predict(laptop_data_invalid)
    except ValueError as e:
        print(f"Error de validación (esperado): {e}")

    print("\n3. Con validación y logging:")
    logged_predictor = LoggingDecorator(validated_predictor)
    price = logged_predictor.predict(laptop_data)
    print(f"Resultado (vea los logs): {price:.2f}€")

    print("\n4. Predictor completo (validación, logging y caché):")
    cached_predictor = CacheDecorator(logged_predictor)

    print("Primera llamada (sin caché):")
    price1 = cached_predictor.predict(laptop_data)
    print(f"Resultado: {price1:.2f}€")

    print("\nSegunda llamada (debe usar caché):")
    price2 = cached_predictor.predict(laptop_data)
    print(f"Resultado (desde caché): {price2:.2f}€")