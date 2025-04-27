# Guía Práctica: Implementando Patrones de Diseño en Proyectos de Machine Learning

## Introducción

Esta guía está diseñada para ayudarte a transformar tu proyecto de machine learning desde un "código espagueti" a una arquitectura de software organizada y mantenible utilizando patrones de diseño. Nos centraremos en tres patrones fundamentales:

1. **Patrón Factory (Fábrica)**: Para crear diferentes tipos de objetos sin especificar su clase concreta
2. **Patrón Singleton**: Para garantizar que una clase tenga una única instancia
3. **Patrón Prototype (Prototipo)**: Para crear nuevos objetos duplicando objetos existentes

## ¿Por qué necesitas patrones de diseño en machine learning?

Los proyectos de machine learning suelen comenzar como notebooks de Jupyter con código exploratoria, pero cuando quieres convertirlo en un producto o aplicación real, necesitas una estructura de software adecuada. Los patrones de diseño te ayudarán a:

- **Organizar** tu código en componentes reutilizables
- **Facilitar** el mantenimiento y las actualizaciones
- **Permitir** la extensión con nuevos modelos o funcionalidades
- **Mejorar** la legibilidad y comprensión del código

## Paso 1: Identificar oportunidades para patrones de diseño

Antes de implementar patrones, debes identificar dónde aplicarlos. Analiza tu código y hazle estas preguntas:

### ¿Dónde aplicar Factory?

Pregúntate:
- ¿Tengo **diferentes algoritmos o modelos** que hacen básicamente lo mismo?
- ¿Necesito **crear visualizaciones similares** pero con pequeñas diferencias?
- ¿Tengo código que **decide qué tipo de objeto crear** basado en parámetros?

**Ejemplo**: En nuestro proyecto, identificamos dos oportunidades para Factory:
1. Para crear diferentes tipos de modelos de ML (LinearRegression, RandomForest)
2. Para crear diferentes tipos de visualizaciones (scatter plots, bar charts, histogramas)

### ¿Dónde aplicar Singleton?

Pregúntate:
- ¿Tengo recursos o servicios que **deben existir solo una vez** en mi aplicación?
- ¿Necesito **acceso global** a ciertos objetos desde diferentes partes del código?
- ¿Hay **configuraciones o estados** que deben ser consistentes en toda la aplicación?

**Ejemplo**: En nuestro proyecto, usamos Singleton para:
1. El `ConfigManager` que gestiona parámetros globales
2. El `ModelManager` que controla el estado del modelo entrenado

### ¿Dónde aplicar Prototype?

Pregúntate:
- ¿Necesito **crear variaciones** de un objeto base?
- ¿Tengo **configuraciones complejas** que quiero reutilizar con modificaciones?
- ¿Quiero **evitar recrear** objetos desde cero?

**Ejemplo**: En nuestro proyecto, usamos Prototype para:
1. Configuraciones de modelos (por ejemplo, diferentes conjuntos de hiperparámetros)

## Paso 2: Diseñar la estructura básica

Una vez identificadas las oportunidades, es momento de diseñar tu estructura de carpetas y archivos. No necesitas hacerlo perfecto desde el principio, pero sí tener una estructura básica.

### Estructura recomendada:

```
tu_proyecto/
├── models/                    # Para las clases relacionadas con modelos ML
│   ├── base_model.py          # Clase base abstracta para modelos
│   ├── model_factory.py       # Factory para crear modelos
│   ├── model_types/           # Implementaciones específicas de modelos
│   └── model_config.py        # Configuraciones de modelos (Prototype)
├── visualizations/            # Para las clases de visualizaciones
│   ├── base_visualization.py  # Clase base abstracta para visualizaciones
│   ├── visualization_factory.py  # Factory para visualizaciones
│   └── visualization_types/   # Implementaciones específicas
├── services/                  # Servicios compartidos (Singletons)
│   ├── config_manager.py      # Gestión de configuración
│   └── model_manager.py       # Gestión de modelos
├── static/                    # Archivos estáticos
├── templates/                 # Plantillas HTML
└── app.py                     # Aplicación principal
```

## Paso 3: Implementar un Patrón Factory

Comienza con Factory porque es fundamental para crear los diferentes tipos de objetos que necesitarás.

### 3.1. Crear una clase base abstracta

Primero, define una interfaz común para todos los objetos que creará la factory:

```python
# models/base_model.py
from abc import ABC, abstractmethod

class PredictionModel(ABC):
    """Clase base abstracta para modelos de predicción"""

    @abstractmethod
    def train(self, X, y):
        """Entrena el modelo con datos"""
        pass

    @abstractmethod
    def predict(self, X):
        """Realiza predicciones con el modelo"""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evalúa el rendimiento del modelo"""
        pass
```

### 3.2. Implementar clases concretas

Ahora, crea las implementaciones específicas que heredan de tu clase base:

```python
# models/model_types/linear_regression.py
from sklearn.linear_model import LinearRegression
from ..base_model import PredictionModel

class LinearRegressionModel(PredictionModel):
    def __init__(self):
        self.model = LinearRegression()
        self.feature_importance = None

    def train(self, X, y):
        self.model.fit(X, y)
        self.feature_importance = self.model.coef_

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        # Calcula métricas y devuelve resultados
        return {
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred)
        }
```

### 3.3. Crear la factory

Ahora crea la clase factory que instanciará los objetos correctos:

```python
# models/model_factory.py
from .model_types.linear_regression import LinearRegressionModel
from .model_types.random_forest import RandomForestModel

class ModelFactory:
    """Factory para crear diferentes tipos de modelos de predicción"""

    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Crea un modelo del tipo especificado

        Args:
            model_type: El tipo de modelo ('linear', 'random_forest', etc.)
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            Una instancia de modelo que implementa PredictionModel
        """
        if model_type == "linear":
            return LinearRegressionModel(**kwargs)
        elif model_type == "random_forest":
            return RandomForestModel(**kwargs)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
```

### 3.4. Usar la factory

En tu código principal:

```python
# Ejemplo de uso
from models.model_factory import ModelFactory

# Crear un modelo específico
model = ModelFactory.create_model("linear")

# Usar el modelo sin preocuparte por su implementación específica
model.train(X_train, y_train)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

## Paso 4: Implementar un Patrón Singleton

### 4.1. Crear la clase Singleton

Para implementar un Singleton, necesitas controlar la creación de instancias:

```python
# services/config_manager.py
import json

class ConfigManager:
    """Singleton para gestionar la configuración global de la aplicación"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializar la instancia por primera vez"""
        try:
            with open('config.json', 'r') as file:
                self.config = json.load(file)
        except FileNotFoundError:
            # Configuración predeterminada
            self.config = {
                "model_type": "linear",
                "features": ["RAM", "Weight", "screen_width", "screen_height", "TypeName_Gaming"],
                "test_size": 0.2,
                "random_state": 42
            }

    def get(self, key, default=None):
        """Obtener un valor de configuración"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Establecer un valor de configuración"""
        self.config[key] = value
        self._save_config()

    def _save_config(self):
        """Guardar la configuración en disco"""
        with open('config.json', 'w') as file:
            json.dump(self.config, file, indent=4)
```

### 4.2. Usar el Singleton

En cualquier parte del código:

```python
from services.config_manager import ConfigManager

# Obtener la única instancia
config = ConfigManager()

# Usar la configuración
model_type = config.get("model_type")
features = config.get("features")

# Actualizar configuración
config.set("model_type", "random_forest")
```

## Paso 5: Implementar un Patrón Prototype

### 5.1. Crear la clase base para prototipos

```python
# models/model_config.py
import copy

class ModelConfig:
    """Configuración para modelos de ML que puede ser clonada"""

    def __init__(self, model_type, features, hyperparameters=None):
        self.model_type = model_type
        self.features = features
        self.hyperparameters = hyperparameters or {}

    def clone(self):
        """Crear una copia de esta configuración"""
        return copy.deepcopy(self)
```

### 5.2. Crear un registro de prototipos

```python
# models/model_config_registry.py
from .model_config import ModelConfig

class ModelConfigRegistry:
    """Almacén de configuraciones predefinidas (prototipos)"""

    def __init__(self):
        self.configs = {}
        self._initialize_default_configs()

    def _initialize_default_configs(self):
        """Crear configuraciones predefinidas"""
        # Configuración básica para regresión lineal
        self.configs["basic_linear"] = ModelConfig(
            model_type="linear",
            features=["RAM", "Weight", "screen_width", "screen_height", "TypeName_Gaming"],
            hyperparameters={}
        )

        # Configuración optimizada para Random Forest
        self.configs["optimized_rf"] = ModelConfig(
            model_type="random_forest",
            features=["RAM", "Weight", "screen_width", "screen_height",
                     "TypeName_Gaming", "TypeName_Notebook", "GHz"],
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42
            }
        )

        # Configuración con características reducidas
        self.configs["reduced_features"] = ModelConfig(
            model_type="linear",
            features=["RAM", "Weight", "TypeName_Gaming"],
            hyperparameters={}
        )

    def get(self, config_name):
        """Obtener un clon de una configuración predefinida"""
        config = self.configs.get(config_name)
        if not config:
            raise ValueError(f"Configuración no encontrada: {config_name}")
        return config.clone()

    def add(self, name, config):
        """Añadir una nueva configuración al registro"""
        self.configs[name] = config
```

### 5.3. Usar los prototipos

```python
from models.model_config_registry import ModelConfigRegistry
from models.model_factory import ModelFactory

# Inicializar el registro
config_registry = ModelConfigRegistry()

# Obtener una configuración predefinida
rf_config = config_registry.get("optimized_rf")

# Modificar la configuración
rf_config.hyperparameters["n_estimators"] = 200

# Crear un modelo usando esta configuración
model = ModelFactory.create_model(
    rf_config.model_type,
    **rf_config.hyperparameters
)
```

## Paso 6: Integrar los patrones en la aplicación web

Ahora que tienes los patrones implementados, integra todo en tu aplicación web:

### 6.1. En `app.py` (o main.py)

```python
from flask import Flask, request, render_template
from services.config_manager import ConfigManager
from services.model_manager import ModelManager
from models.model_config_registry import ModelConfigRegistry

app = Flask(__name__)

# Inicializar singletons y registros
config_manager = ConfigManager()
model_manager = ModelManager()
config_registry = ModelConfigRegistry()

@app.route('/')
def index():
    """Página principal"""
    metrics = model_manager.get_model_metrics()
    model_type = config_manager.get("model_type")

    return render_template(
        'index.html',
        metrics=metrics,
        model_type=model_type
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Realizar una predicción"""
    # Obtener datos del formulario
    form_data = request.form

    # Preprocesar datos de entrada
    input_features = preprocess_input(form_data)

    # Realizar predicción usando el ModelManager (Singleton)
    prediction = model_manager.predict(input_features)

    return render_template('prediction.html', prediction=prediction)

@app.route('/change_model/<model_type>')
def change_model(model_type):
    """Cambiar el tipo de modelo usando Factory"""
    # Actualizar configuración (Singleton)
    config_manager.set("model_type", model_type)

    # Reentrenar modelo
    model_manager.train_model()

    # Redirigir a página principal
    return redirect(url_for('index'))

@app.route('/use_config/<config_name>')
def use_config(config_name):
    """Usar una configuración predefinida (Prototype)"""
    # Obtener un clon de la configuración
    config = config_registry.get(config_name)

    # Actualizar configuración global
    config_manager.set("model_type", config.model_type)
    config_manager.set("features", config.features)

    # Reentrenar con la nueva configuración
    model_manager.train_model()

    return redirect(url_for('index'))
```

### 6.2. Actualizar las plantillas HTML

En tus plantillas HTML, añade formas de interactuar con los patrones:

```html
<!-- templates/index.html -->
<div class="model-controls">
    <h3>Configuración del Modelo</h3>
    <div>
        <h4>Seleccionar Modelo</h4>
        <a href="/change_model/linear" class="button">Regresión Lineal</a>
        <a href="/change_model/random_forest" class="button">Random Forest</a>
    </div>

    <div>
        <h4>Configuraciones Predefinidas</h4>
        <a href="/use_config/basic_linear" class="button">Básico (Lineal)</a>
        <a href="/use_config/optimized_rf" class="button">Optimizado (Random Forest)</a>
        <a href="/use_config/reduced_features" class="button">Características Reducidas</a>
    </div>

    <div>
        <h4>Métricas del Modelo</h4>
        <p>Tipo: {{ model_type }}</p>
        <p>MAE: {{ metrics.mae }}</p>
        <p>R²: {{ metrics.r2 }}</p>
    </div>
</div>
```

## Paso 7: Implementar visualizaciones con Factory

Para las visualizaciones, sigue un enfoque similar al de los modelos:

### 7.1. Crear una clase base para visualizaciones

```python
# visualizations/base_visualization.py
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Visualization(ABC):
    """Clase base para todas las visualizaciones"""

    def __init__(self, title="", figsize=(10, 6), **kwargs):
        self.title = title
        self.figsize = figsize

        # Guardar otros parámetros
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def plot(self, data):
        """Crear la visualización con los datos proporcionados"""
        pass

    def save(self, filepath):
        """Guardar la visualización en un archivo"""
        plt.savefig(filepath)
        plt.close()
```

### 7.2. Implementar visualizaciones concretas

```python
# visualizations/visualization_types/scatter_plot.py
import matplotlib.pyplot as plt
from ..base_visualization import Visualization

class ScatterPlot(Visualization):
    """Gráfico de dispersión"""

    def plot(self, data):
        """
        Crear un gráfico de dispersión

        Args:
            data: Diccionario con claves 'x', 'y' y opcionalmente 'add_line'
        """
        plt.figure(figsize=self.figsize)

        # Crear scatter plot
        plt.scatter(data['x'], data['y'], alpha=0.7)

        # Añadir línea de referencia si se solicita
        if data.get('add_line', False):
            min_val = min(min(data['x']), min(data['y']))
            max_val = max(max(data['x']), max(data['y']))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')

        # Configurar etiquetas y título
        plt.xlabel(getattr(self, 'xlabel', 'X'))
        plt.ylabel(getattr(self, 'ylabel', 'Y'))
        plt.title(self.title)
        plt.tight_layout()

        return plt
```

### 7.3. Crear la factory de visualizaciones

```python
# visualizations/visualization_factory.py
from .visualization_types.scatter_plot import ScatterPlot
from .visualization_types.bar_plot import BarPlot
from .visualization_types.histogram_plot import HistogramPlot

class VisualizationFactory:
    """Factory para crear visualizaciones"""

    @staticmethod
    def create_visualization(viz_type, **kwargs):
        """
        Crear una visualización del tipo especificado

        Args:
            viz_type: Tipo de visualización ('scatter', 'bar', 'histogram')
            **kwargs: Parámetros para la visualización

        Returns:
            Una instancia de visualización
        """
        if viz_type == 'scatter':
            return ScatterPlot(**kwargs)
        elif viz_type == 'bar':
            return BarPlot(**kwargs)
        elif viz_type == 'histogram':
            return HistogramPlot(**kwargs)
        else:
            raise ValueError(f"Tipo de visualización desconocido: {viz_type}")
```

### 7.4. Usar las visualizaciones en tu aplicación

```python
from visualizations.visualization_factory import VisualizationFactory

# Crear una visualización
scatter = VisualizationFactory.create_visualization(
    'scatter',
    title='Valores reales vs. predichos',
    xlabel='Real',
    ylabel='Predicho'
)

# Generar la visualización
scatter.plot({
    'x': y_test,
    'y': y_pred,
    'add_line': True
})

# Guardar la visualización
scatter.save('static/prediction_scatter.png')
```

## Resumen y consejos finales

### Secuencia recomendada de implementación

1. **Identifica** dónde aplicar cada patrón en tu código
2. **Comienza** con Factory para los modelos y visualizaciones
3. **Implementa** Singleton para configuración y gestión de modelos
4. **Añade** Prototype para configuraciones reutilizables
5. **Integra** todo en tu aplicación web

### Consejos para principiantes

1. **Comienza con poco**: Implementa un patrón a la vez
2. **Prioriza la Factory**: Es la base para los otros patrones
3. **Escribe pruebas**: Asegúrate de que todo funciona como esperas
4. **No sobre-diseñes**: Utiliza patrones donde aporten valor, no por usarlos
5. **Refactoriza gradualmente**: No tienes que reescribir toda la aplicación de golpe

### Signos de que necesitas patrones

1. **Factory**: Cuando tienes muchos `if/elif/else` para decidir qué crear
2. **Singleton**: Cuando pasas la misma instancia por todo tu código
3. **Prototype**: Cuando copias y pegas configuraciones con pocos cambios

## Conclusión

Implementar patrones de diseño en proyectos de machine learning transforma código exploratorio en software profesional y mantenible. Esta guía te ha mostrado cómo identificar oportunidades para patrones y cómo implementar Factory, Singleton y Prototype en el contexto de un proyecto de machine learning.

Recuerda que los patrones son herramientas, no objetivos. Úsalos cuando te ayuden a resolver problemas específicos de diseño y aporten claridad y estructura a tu código.