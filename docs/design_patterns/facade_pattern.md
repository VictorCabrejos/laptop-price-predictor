# Patrón de Diseño: Facade

## Descripción

El **patrón Facade** (Fachada) proporciona una interfaz unificada y simplificada a un conjunto de interfaces en un subsistema. Define una interfaz de nivel superior que hace que el subsistema sea más fácil de usar al ocultar su complejidad.

En el contexto de una aplicación de machine learning, el patrón Facade es especialmente útil para encapsular múltiples subsistemas relacionados con el entrenamiento, preprocesamiento, evaluación y predicción en una interfaz simple y coherente.

## Problema que resuelve

En el proyecto Laptop Price Predictor, nos encontramos con el siguiente problema:

- La API necesita interactuar con múltiples subsistemas: preprocesamiento de datos, modelos de ML, búsqueda de laptops similares, visualizaciones, etc.
- Los controladores de la API tienen que conocer los detalles de implementación de cada subsistema
- El código se vuelve monolítico y difícil de mantener
- Si un subsistema cambia, hay que modificar múltiples partes del código

Esto genera acoplamiento y complejidad excesiva en el cliente (la API), dificultando:
- El mantenimiento del código
- Las pruebas unitarias
- La reutilización de componentes
- La comprensión del flujo de trabajo

## Solución: Patrón Facade

El patrón Facade resuelve este problema mediante la creación de una clase unificadora (`LaptopPredictionFacade`) que:

1. Proporciona una interfaz simplificada para todas las operaciones complejas
2. Coordina los subsistemas internos
3. Abstrae la complejidad de los subsistemas
4. Centraliza el manejo de errores y flujos de trabajo

## Implementación

### Versión Original (Acoplada)

```python
@app.post("/predict", response_class=HTMLResponse)
async def predict_original(
    request: Request,
    weight: float = Form(...),
    # ... otros parámetros
):
    # 1. Validar datos manualmente
    if weight <= 0 or weight > 5:
        return templates.TemplateResponse("error.html", {...})

    # 2. Preparar datos para predicción
    input_data = np.array([[weight, is_gaming, is_notebook, ...]])

    # 3. Verificar si el modelo está cargado
    if model is None:
        # 4. Cargar y preparar datos para el modelo
        df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
        # Múltiples líneas de preprocesamiento...

        # Entrenar modelo
        model = LinearRegression()
        model.fit(X, y)

    # 5. Hacer predicción
    prediction = model.predict(input_data)[0]

    # 6. Buscar laptops similares
    df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
    # Más código para buscar similares...

    # 7. Generar visualización
    plt.figure(figsize=(8, 4))
    # Más código para visualización...

    # 8. Devolver respuesta
    return templates.TemplateResponse("prediction.html", {...})
```

### Versión con Facade

```python
# 1. Crear la fachada que encapsula todos los subsistemas
class LaptopPredictionFacade:
    def __init__(self):
        self.initialize()

    def initialize(self):
        """Inicializa todos los componentes necesarios"""
        # Inicializar procesador de datos
        self.data_processor = self._create_data_processor()
        # Inicializar modelo
        self.model = self._create_model()
        # Inicializar otros subsistemas...

    def predict_price(self, features: dict) -> dict:
        """Método principal, coordina todos los subsistemas"""
        try:
            # 1. Validar datos de entrada
            self._validate_input(features)
            # 2. Preparar datos para predicción
            input_data = self._prepare_prediction_input(features)
            # 3. Realizar predicción
            price = float(self.model.predict(input_data)[0])
            # 4. Buscar laptops similares
            similar = self._find_similar_laptops(price)
            # 5. Generar visualización
            self._generate_visualizations(features, price)
            # 6. Devolver resultado completo
            return {
                'success': True,
                'price': price,
                'similar_laptops': similar
            }
        except Exception as e:
            # Manejo centralizado de errores
            return {'success': False, 'error': str(e)}

    # Métodos privados para manejar subsistemas
    def _validate_input(self, features): ...
    def _prepare_prediction_input(self, features): ...
    def _find_similar_laptops(self, price): ...
    def _generate_visualizations(self, features, price): ...

# 2. Uso simplificado desde el cliente (API)
@app.post("/predict", response_class=HTMLResponse)
async def predict_web(request: Request, ...):
    # Preparar datos
    features = {
        'weight': weight,
        'is_gaming': is_gaming,
        # ... otros parámetros
    }

    # Un solo punto de contacto con la fachada
    result = facade.predict_price(features)

    # Verificar resultado y responder
    if not result['success']:
        return templates.TemplateResponse("error.html", {...})

    return templates.TemplateResponse("prediction.html", {...})
```

## Ventajas

1. **Simplicidad**: Proporciona una interfaz unificada y simple para operaciones complejas
2. **Desacoplamiento**: El cliente no necesita conocer los detalles internos del subsistema
3. **Mantenimiento**: Los cambios en los subsistemas no afectan al cliente
4. **Cohesión**: Agrupa operaciones relacionadas en una sola clase
5. **Manejabilidad**: Centraliza el manejo de errores y flujos de trabajo complejos
6. **Testabilidad**: Facilita las pruebas al aislar subsistemas
7. **Reutilización**: La fachada puede ser utilizada por diferentes clientes

## Cuándo usar el patrón Facade

Este patrón es especialmente útil cuando:

- Necesitas proporcionar una interfaz simple para un subsistema complejo
- Hay muchas dependencias entre clientes y clases de implementación
- Quieres organizar un subsistema en capas
- Deseas reducir el acoplamiento entre subsistemas y clientes
- Tienes múltiples pasos en un proceso que deben coordinarse

## Diagrama UML

```
+----------------+         +----------------------+
|                |         |                      |
| Cliente (API)  |-------->| Facade               |
|                |         | (LaptopPredictionFa.)|
+----------------+         +----------------------+
                              |         |         |
                              |         |         |
                              v         v         v
                   +----------+  +------+  +------+
                   | Subsist. |  | Subs.|  | Subs.|
                   | Data     |  | Model|  | Viz  |
                   +----------+  +------+  +------+
```

## Ejemplo completo

Revisa el archivo `facade_pattern.py` para ver un ejemplo completo y funcional del patrón Facade aplicado al predictor de precios de laptops.