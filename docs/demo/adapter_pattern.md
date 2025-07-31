# Patrón de Diseño: Adapter

## Descripción

El **patrón Adapter** (Adaptador) actúa como un puente entre dos interfaces incompatibles. Este patrón permite que clases con interfaces incompatibles trabajen juntas al crear una clase intermedia que sirve como convertidor.

En el contexto de una aplicación de machine learning, el patrón Adapter es especialmente útil para desacoplar los modelos de ML de la API web que los expone.

## Problema que resuelve

En el proyecto Laptop Price Predictor, nos encontramos con el siguiente problema:

- El modelo de scikit-learn espera datos en formato numpy array con una estructura específica
- La API web recibe datos en un formato diferente (JSON o formularios HTML)
- Cada cambio en el modelo requiere cambios en el código de la API y viceversa

Esto crea un **acoplamiento fuerte** entre la API y el modelo de ML, lo que dificulta:
- Cambiar el modelo sin afectar la API
- Probar cada componente de forma aislada
- Reutilizar el modelo en diferentes contextos

## Solución: Patrón Adapter

El patrón Adapter resuelve este problema mediante la creación de una clase intermedia (`LaptopPriceModelAdapter`) que:

1. Actúa como traductor entre la API web y el modelo de ML
2. Convierte los datos del formato de la API al formato que espera el modelo
3. Encapsula la lógica de interacción con el modelo

## Implementación

### Versión Original (Acoplada)

```python
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    weight: float = Form(...),
    # ... otros parámetros
    ram: int = Form(...)
):
    # Formato acoplado directamente al modelo
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])

    # Llamada directa al modelo
    prediction = model.predict(input_data)[0]

    # Uso directo del resultado
    return {"price": prediction}
```

### Versión con Adapter

```python
# 1. Definir interfaces claras
class LaptopFeatures(BaseModel):
    weight: float
    is_gaming: int
    # ... otras características

# 2. Crear el adaptador
class LaptopPriceModelAdapter:
    def __init__(self, model):
        self.model = model

    def predict(self, laptop_data: dict) -> float:
        # Convertir al formato que espera el modelo
        input_array = np.array([[
            laptop_data["weight"],
            # ... otras características
        ]])

        # Realizar predicción
        return self.model.predict(input_array)[0]

# 3. Usar el adaptador en la API
@app.post("/api/predict")
async def predict(laptop: LaptopFeatures):
    # Usar el adaptador para hacer la predicción
    price = adapter.predict(laptop.dict())
    return {"price": price}
```

## Ventajas

1. **Desacoplamiento**: La API y el modelo pueden evolucionar independientemente
2. **Testabilidad**: Cada componente puede probarse por separado
3. **Flexibilidad**: Permite cambiar el modelo subyacente sin cambiar la API
4. **Reutilización**: El adaptador puede usarse en diferentes contextos (API web, CLI, etc.)
5. **Claridad**: Las responsabilidades están claramente separadas

## Cuándo usar el patrón Adapter

Este patrón es especialmente útil cuando:

- Necesitas integrar un sistema existente (como un modelo de ML) con una nueva interfaz
- Quieres desacoplar componentes para facilitar pruebas y mantenimiento
- Trabajas con bibliotecas de terceros que no puedes modificar
- Necesitas transformar datos entre diferentes formatos o estructuras
- Quieres crear una interfaz estable para un componente que puede cambiar

## Diagrama UML

```
+----------------+    +-------------------+    +--------------+
|                |    |                   |    |              |
| Cliente (API)  |----| Adapter           |--->| Adaptado     |
|                |    | (ModelAdapter)    |    | (Modelo ML)  |
+----------------+    +-------------------+    +--------------+
```

## Ejemplo completo

Revisa el archivo `adapter_pattern.py` para ver un ejemplo completo y funcional del patrón Adapter aplicado al predictor de precios de laptops.