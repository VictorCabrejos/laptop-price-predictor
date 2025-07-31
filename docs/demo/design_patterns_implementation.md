# Implementación de Patrones de Diseño en el Proyecto Laptop Price Predictor

Este documento describe la implementación de tres patrones de diseño (Singleton, Factory y Prototype) en el proyecto Laptop Price Predictor, transformándolo desde un código monolítico (spaghetti code) a una arquitectura modular y mantenible.

## Estructura del Proyecto Refactorizado

La nueva estructura del proyecto es:

```
laptop-price-predictor/
├── main.py           # Archivo principal refactorizado
├── main.old.py       # Versión original (spaghetti code) como referencia
├── models/           # Directorio para modelos y patrones de diseño
│   ├── config_manager.py        # Implementación del Singleton para configuración
│   ├── model_manager.py         # Implementación del Singleton para gestión de modelos
│   ├── model_factory.py         # Implementación del Factory Method
│   ├── prediction_model.py      # Interfaz base para el Factory
│   ├── linear_regression_model.py # Implementación concreta para Factory
│   ├── random_forest_model.py     # Implementación concreta para Factory
│   ├── model_config.py          # Implementación del Prototype para configuraciones
│   └── model_config_registry.py   # Registro para gestionar prototipos
├── static/           # Directorio para archivos estáticos
├── templates/        # Directorio para templates HTML
└── docs/             # Documentación
```

## 1. Patrón Singleton

### Implementación

Hemos implementado dos clases Singleton:

1. **ConfigManager** (`models/config_manager.py`):
   - Gestiona la configuración global de la aplicación
   - Mantiene una única instancia accesible desde cualquier parte del código
   - Centraliza la lectura y escritura de parámetros de configuración

2. **ModelManager** (`models/model_manager.py`):
   - Gestiona la carga, entrenamiento y uso del modelo de predicción
   - Mantiene una única instancia del gestor de modelos
   - Coordina las operaciones con el modelo entrenado

### Metodología de Implementación

Ambas clases utilizan el mismo enfoque para implementar el patrón Singleton:

1. Una variable de clase privada `_instance` para almacenar la única instancia
2. Sobreescritura del método `__new__` para controlar la creación de instancias
3. Un método `_initialize()` que se llama solo la primera vez que se instancia la clase

Ejemplo de implementación:

```python
class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Inicialización de la única instancia
        self.config = { /* valores por defecto */ }
```

### Beneficios

1. **Acceso global controlado**: Proporcionamos un punto de acceso global sin usar variables globales
2. **Estado consistente**: Garantizamos que toda la aplicación use la misma configuración
3. **Inicialización perezosa**: El modelo se entrena solo cuando se necesita por primera vez
4. **Control centralizado**: Los cambios en la configuración afectan a todo el sistema

## 2. Patrón Factory Method

### Implementación

Para permitir diferentes algoritmos de machine learning, implementamos el patrón Factory Method:

1. **Interfaz Base** (`models/prediction_model.py`):
   - Define la interfaz común para todos los modelos de predicción
   - Declara métodos abstractos como `train()`, `predict()`, `evaluate()`

2. **Clases Concretas**:
   - `LinearRegressionModel`: Implementación de regresión lineal
   - `RandomForestModel`: Implementación de Random Forest

3. **ModelFactory** (`models/model_factory.py`):
   - Crea instancias de modelos según el tipo solicitado
   - Oculta los detalles de construcción de los modelos

### Metodología de Implementación

1. Diseñamos una interfaz base con métodos abstractos usando `ABC`
2. Cada algoritmo se implementa en una clase separada que hereda de la interfaz
3. La Factory usa un método estático para crear la instancia adecuada según el tipo

Ejemplo:

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == 'linear':
            return LinearRegressionModel()
        elif model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            random_state = kwargs.get('random_state', 42)
            return RandomForestModel(n_estimators, random_state)
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {model_type}")
```

### Beneficios

1. **Extensibilidad**: Podemos añadir nuevos tipos de modelos sin modificar el código existente
2. **Desacoplamiento**: El código cliente trabaja con interfaces, no con implementaciones
3. **Encapsulación**: Los detalles de construcción de modelos están centralizados
4. **Experimentación**: Facilita probar diferentes algoritmos y comparar rendimiento

## 3. Patrón Prototype

### Implementación

Para facilitar la experimentación con diferentes configuraciones de modelos:

1. **ModelConfig** (`models/model_config.py`):
   - Define configuraciones clonables para modelos
   - Implementa el método `clone()` usando `copy.deepcopy()`
   - Incluye parámetros para el modelo, características, visualización, etc.

2. **ModelConfigRegistry** (`models/model_config_registry.py`):
   - Almacena configuraciones predefinidas como prototipos
   - Proporciona método `get()` que devuelve un clon del prototipo solicitado
   - Inicializa por defecto varias configuraciones útiles

### Metodología de Implementación

1. Creamos una clase para configuraciones con método `clone()`
2. Implementamos un registro para almacenar y recuperar prototipos
3. Utilizamos `copy.deepcopy()` para garantizar clones independientes
4. Predefinimos algunas configuraciones útiles como punto de partida

Ejemplo:

```python
def clone(self):
    """Crea un clon profundo de la configuración actual"""
    return copy.deepcopy(self)

# Uso del patrón
original_config = config_registry.get('optimized_rf')
modified_config = original_config.clone()
modified_config.params['n_estimators'] = 200
# El original sigue intacto
```

### Beneficios

1. **Reutilización**: Podemos crear nuevas configuraciones basadas en existentes
2. **No-destructivo**: Las modificaciones no afectan a los prototipos originales
3. **Flexibilidad**: Facilita la experimentación con diferentes parámetros
4. **Eficiencia**: Evita reinicializar configuraciones complejas desde cero

## Integración de los Patrones en `main.py`

El archivo `main.py` refactorizado integra estos tres patrones:

1. **Inicialización**:
   - Obtiene instancias Singleton de `ConfigManager` y `ModelManager`
   - Inicializa el registro de prototipos de configuración

2. **Endpoints**:
   - `/`: Muestra la página principal con métricas del modelo
   - `/model_info`: Muestra información detallada del modelo
   - `/predict`: Realiza predicciones con los datos ingresados
   - `/retrain`: Fuerza el reentrenamiento del modelo
   - `/change_model/{model_type}`: Cambia el tipo de modelo usando Factory
   - `/model_configs`: Lista configuraciones disponibles (Prototype)
   - `/use_model_config/{config_name}`: Usa una configuración clonada

### Ejemplo de Integración

```python
@app.get("/change_model/{model_type}")
async def change_model(model_type: str):
    # Actualiza configuración usando Singleton
    config_manager.set('model_type', model_type)

    # Fuerza reentrenamiento con Factory (indirectamente a través de ModelManager)
    model_manager.train_model()

    # Obtiene métricas del Singleton ModelManager
    metrics = model_manager.get_model_metrics()
    return {
        "message": f"Model changed to {model_type} successfully",
        "metrics": metrics
    }
```

## Ventajas de la Refactorización

1. **Modularidad**: Código organizado en componentes con responsabilidades claras
2. **Mantenibilidad**: Cambios localizados sin afectar al sistema completo
3. **Reutilización**: Componentes que pueden usarse en diferentes contextos
4. **Extensibilidad**: Facilidad para añadir nuevas funcionalidades
5. **Comprensibilidad**: Código más claro y autodocumentado

## Cómo Probar los Patrones

1. **Singleton**:
   - Ejecuta `python -c "from models.config_manager import ConfigManager; a = ConfigManager(); b = ConfigManager(); print(a is b)"`
   - Debería imprimir `True` indicando que son la misma instancia

2. **Factory Method**:
   - Accede a `/change_model/random_forest` para cambiar a un modelo Random Forest
   - Accede a `/change_model/linear` para cambiar a regresión lineal

3. **Prototype**:
   - Accede a `/model_configs` para ver las configuraciones disponibles
   - Accede a `/use_model_config/optimized_rf` para usar la configuración de Random Forest optimizada

## Conclusión

La refactorización ha transformado un código inicial monolítico y difícil de mantener en una arquitectura modular y extensible mediante la aplicación de tres patrones de diseño fundamentales. Este enfoque no solo mejora la calidad del código sino que también facilita la adición de nuevas características y la experimentación con diferentes algoritmos de machine learning.