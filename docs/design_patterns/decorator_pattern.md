# Patrón de Diseño: Decorator

## Descripción

El **patrón Decorator** (Decorador) permite añadir funcionalidades a objetos existentes dinámicamente durante la ejecución, sin alterar su estructura. Este patrón crea una serie de clases decoradoras que envuelven al objeto original, añadiendo nuevos comportamientos.

En el contexto de una aplicación de machine learning, el patrón Decorator es especialmente útil para agregar funcionalidades transversales como validación, logging, caché o monitoreo a los modelos de predicción.

## Problema que resuelve

En el proyecto Laptop Price Predictor, nos encontramos con el siguiente problema:

- La función de predicción se volvió monolítica con múltiples responsabilidades
- Hay código mezclado para validación, logging y la predicción en sí
- Para añadir nuevas funcionalidades, hay que modificar el código existente
- Es difícil reutilizar componentes específicos (como la validación) en otros contextos

Esto viola varios principios SOLID:
- Principio de Responsabilidad Única
- Principio Abierto/Cerrado
- Principio de Segregación de Interfaces

## Solución: Patrón Decorator

El patrón Decorator resuelve este problema mediante:

1. La definición de una interfaz común (`PricePredictor`)
2. La implementación de una clase base concreta (`SklearnPredictor`)
3. La creación de decoradores que implementan la misma interfaz pero delegan al componente decorado
4. La composición dinámica de los decoradores según las necesidades

## Implementación

### Versión Original (Acoplada)

```python
def predict_price_original(model, weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram):
    # Validación mezclada con la lógica de negocio
    if weight <= 0 or weight > 5:
        raise ValueError(f"Peso inválido: {weight}. Debe estar entre 0 y 5 kg.")

    # Más validaciones...

    # Logging mezclado con la lógica de negocio
    print(f"Iniciando predicción para laptop: {weight}kg, Gaming: {is_gaming}...")

    start_time = time.time()

    # Preparar datos y predecir
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])
    result = model.predict(input_data)[0]

    # Más logging mezclado
    elapsed_time = time.time() - start_time
    print(f"Predicción completada en {elapsed_time:.4f} segundos. Resultado: {result:.2f}€")

    return result
```

### Versión con Decorator

```python
# 1. Interfaz común
class PricePredictor:
    def predict(self, features: Dict[str, Any]) -> float:
        pass

# 2. Implementación concreta
class SklearnPredictor(PricePredictor):
    def __init__(self, model):
        self.model = model

    def predict(self, features: Dict[str, Any]) -> float:
        # Lógica de predicción básica
        input_data = np.array([[
            features['weight'],
            # ...otras características
        ]])
        return float(self.model.predict(input_data)[0])

# 3. Decorador base
class PredictorDecorator(PricePredictor):
    def __init__(self, predictor: PricePredictor):
        self._predictor = predictor

    def predict(self, features: Dict[str, Any]) -> float:
        return self._predictor.predict(features)

# 4. Decoradores concretos
class ValidationDecorator(PredictorDecorator):
    def predict(self, features: Dict[str, Any]) -> float:
        # Lógica de validación
        if features['weight'] <= 0 or features['weight'] > 5:
            raise ValueError(f"Peso inválido: {features['weight']}")
        # ...más validaciones
        return self._predictor.predict(features)

class LoggingDecorator(PredictorDecorator):
    def predict(self, features: Dict[str, Any]) -> float:
        logger.info(f"Iniciando predicción...")
        start_time = time.time()
        result = self._predictor.predict(features)
        elapsed_time = time.time() - start_time
        logger.info(f"Predicción completada en {elapsed_time:.4f} segundos")
        return result
```

## Uso con Composición

El poder del patrón Decorator reside en la capacidad de componer funcionalidades:

```python
# Crear predictor base
base_predictor = SklearnPredictor(model)

# Opción 1: Solo validación
validated_predictor = ValidationDecorator(base_predictor)

# Opción 2: Solo logging
logged_predictor = LoggingDecorator(base_predictor)

# Opción 3: Validación + logging
validated_and_logged = LoggingDecorator(ValidationDecorator(base_predictor))

# Opción 4: Validación + logging + caché
full_predictor = CacheDecorator(LoggingDecorator(ValidationDecorator(base_predictor)))
```

## Ventajas

1. **Separación de Responsabilidades**: Cada decorador tiene una única responsabilidad
2. **Extensibilidad**: Podemos añadir nuevas funcionalidades sin modificar código existente
3. **Composición Dinámica**: Permite configurar las funcionalidades en tiempo de ejecución
4. **Reutilización**: Los decoradores pueden usarse en diferentes contextos
5. **Principio Abierto/Cerrado**: Extendemos el comportamiento sin modificar las clases
6. **Principio de Responsabilidad Única**: Cada clase tiene una sola razón para cambiar

## Cuándo usar el patrón Decorator

Este patrón es especialmente útil cuando:

- Necesitas añadir responsabilidades a objetos individuales dinámicamente
- Quieres evitar la explosión de subclases para proveer combinaciones de funcionalidades
- La herencia no es una opción viable (ej. clases finales)
- Necesitas funcionalidades transversales que se apliquen a múltiples clases
- Las responsabilidades pueden componerse de forma flexible

## Diagrama UML

```
        +----------------+
        |  PricePredictor|
        +----------------+
        | + predict()    |
        +----------------+
               ▲
               |
    +----------+----------+
    |                     |
+----------------+  +----------------+
|SklearnPredictor|  |PredictorDecorator|
+----------------+  +----------------+
| + predict()    |  | - _predictor   |
+----------------+  | + predict()    |
                    +----------------+
                         ▲
          +--------------|-------------+
          |              |             |
+-------------------+  +----------------+  +----------------+
|ValidationDecorator|  |LoggingDecorator|  |CacheDecorator  |
+-------------------+  +----------------+  +----------------+
| + predict()       |  | + predict()    |  | + predict()    |
+-------------------+  +----------------+  +----------------+
```

## Ejemplo completo

Revisa el archivo `decorator_pattern.py` para ver un ejemplo completo y funcional del patrón Decorator aplicado al predictor de precios de laptops.