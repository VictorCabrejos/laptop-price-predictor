# Ejercicio de Patrón de Diseño: Factory

## Introducción

El patrón Factory (Fábrica) es un patrón de diseño creacional que proporciona una interfaz para crear objetos en una superclase, pero permite a las subclases alterar el tipo de objetos que se crearán. Este patrón es útil cuando necesitamos crear diferentes tipos de objetos sin acoplar el código cliente a las clases concretas.

## Problema en el Código Actual

En el archivo `main.py`, observarás que solo se utiliza un tipo de modelo de regresión lineal:

```python
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
```

Esto presenta varios problemas:

1. **Limitación a un solo tipo de modelo**: No podemos cambiar fácilmente entre diferentes algoritmos de ML
2. **Acoplamiento rígido**: El código está fuertemente acoplado a la implementación de LinearRegression
3. **Falta de extensibilidad**: Agregar nuevos tipos de modelos requiere modificar código existente
4. **Dificultad para pruebas**: No se pueden probar fácilmente diferentes tipos de modelos

## Ejercicio

Tu tarea es refactorizar el código para implementar el patrón Factory para la creación de modelos de machine learning:

1. **Crear una interfaz/clase base abstracta para modelos**:
   - Implementa una clase base `Model` o `PredictionModel` con métodos comunes
   - Define métodos como `train()`, `predict()`, `evaluate()`

2. **Implementar diferentes tipos de modelos concretos**:
   - Implementa `LinearRegressionModel` que hereda de la clase base
   - Implementa `RandomForestModel` que también hereda de la clase base
   - Opcionalmente, implementa otros tipos de modelos (SVR, GradientBoosting, etc.)

3. **Crear una Factory para generar los modelos**:
   - Implementa una clase `ModelFactory` que genera diferentes tipos de modelos
   - Proporciona un método para crear modelos según un parámetro de tipo

## Directivas de Implementación

### Para la clase base de modelos:

1. Crea un nuevo archivo `base_model.py` con una clase abstracta para modelos:
```python
from abc import ABC, abstractmethod

class PredictionModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass
```

### Para los modelos concretos:

1. Crea un archivo `linear_model.py` para implementar `LinearRegressionModel`
2. Crea un archivo `random_forest_model.py` para implementar `RandomForestModel`
3. Implementa los métodos abstractos en cada clase concreta

### Para la Factory:

1. Crea un archivo `model_factory.py` con la clase `ModelFactory`
2. Implementa un método `create_model(model_type)` que devuelva una instancia del modelo solicitado

## Ejemplo de Implementación

La estructura básica podría ser:

```python
# model_factory.py
from linear_model import LinearRegressionModel
from random_forest_model import RandomForestModel

class ModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == "linear":
            return LinearRegressionModel()
        elif model_type == "random_forest":
            return RandomForestModel()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
```

## Cómo Verificar tu Solución

Una implementación correcta del patrón Factory debe demostrar que:

1. Puedes crear diferentes tipos de modelos a través de la misma interfaz
2. El código cliente (como las rutas en `main.py`) no está acoplado a las clases concretas
3. Es fácil añadir nuevos tipos de modelos sin modificar el código existente

Ejemplo de prueba:

```python
# Uso de la factory en main.py
factory = ModelFactory()
model = factory.create_model(model_type="linear")  # o "random_forest"
model.train(X_train, y_train)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

## Beneficios Esperados

Al implementar el patrón Factory:

1. **Flexibilidad**: Facilidad para cambiar entre diferentes tipos de modelos
2. **Extensibilidad**: Capacidad para agregar nuevos modelos fácilmente
3. **Mantenibilidad**: Código más organizado y modular
4. **Pruebas**: Facilita probar múltiples tipos de modelos
5. **Separación de responsabilidades**: Cada clase tiene un único propósito

## Mejoras Adicionales

- Considera usar un método de registro de modelos en la Factory para permitir el registro dinámico de nuevos tipos
- Implementa un mecanismo para configurar hiperparámetros en la creación del modelo
- Piensa en cómo podría combinarse este patrón con el Singleton implementado anteriormente
- Reflexiona sobre cómo este patrón facilita el cumplimiento del principio de Inversión de Dependencias (parte de SOLID)