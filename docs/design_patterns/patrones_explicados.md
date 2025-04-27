# Patrones de Diseño en el Predictor de Precios de Laptops

## Introducción

Este documento explica la implementación de tres patrones de diseño importantes (Singleton, Factory y Prototype) en nuestra aplicación de predicción de precios de laptops. Está orientado a estudiantes con conocimientos básicos de programación que quieren entender por qué estos patrones son útiles en aplicaciones de machine learning.

## El Problema Original: "Código Espagueti"

En la versión original de la aplicación (que aún pueden ver en `main.old.py`), todo el código estaba concentrado en un solo archivo. Este enfoque, comúnmente llamado "código espagueti", presentaba varios problemas:

1. **Variables globales** que cualquier parte del código podía modificar
2. **Código duplicado** en diferentes funciones
3. **Alto acoplamiento** entre componentes (la predicción, el entrenamiento y la visualización estaban mezclados)
4. **Dificultad para probar** partes individuales del código
5. **Imposibilidad de cambiar modelos** sin modificar múltiples partes del código

Un ejemplo claro es cómo se manejaba el modelo:

```python
# Variables globales para estado
model = None
features = ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']
feature_importances = {}
mae = 0
r2 = 0

# Función que hace demasiadas cosas
def load_data():
    global model, feature_importances, mae, r2

    # Carga datos
    # Limpia datos
    # Entrena modelo
    # Evalúa modelo
    # Genera gráficos
    # ...
```

Este enfoque es problemático porque:
- Es difícil reutilizar partes específicas
- No se pueden cambiar componentes sin afectar a todo el sistema
- No permite experimentar con diferentes algoritmos o configuraciones

## La Solución: Patrones de Diseño

Implementamos tres patrones de diseño fundamentales para resolver estos problemas:

### 1. Patrón Singleton

**¿Qué es?** Es un patrón que garantiza que una clase tenga una única instancia y proporciona un punto de acceso global a ella.

**¿Por qué lo necesitamos?** En machine learning, necesitamos:
- Un único modelo entrenado compartido por toda la aplicación
- Una única configuración centralizada y coherente
- Evitar cargar y entrenar el modelo múltiples veces

**¿Cómo lo implementamos?**

Creamos dos clases Singleton principales:

1. **ConfigManager** (`models/config_manager.py`): Gestiona toda la configuración de la aplicación
   ```python
   class ConfigManager:
       _instance = None

       def __new__(cls):
           if cls._instance is None:
               cls._instance = super(ConfigManager, cls).__new__(cls)
               cls._instance._initialize()
           return cls._instance
   ```

2. **ModelManager** (`models/model_manager.py`): Gestiona el ciclo de vida del modelo ML
   - Carga de datos
   - Entrenamiento
   - Predicción
   - Visualización
   - Métricas

**Beneficios:**
- La configuración está centralizada y protegida
- El modelo se carga y entrena solo una vez
- Cualquier parte de la aplicación puede acceder a la misma instancia

### 2. Patrón Factory

**¿Qué es?** Es un patrón que proporciona una interfaz para crear objetos sin especificar sus clases concretas.

**¿Por qué lo necesitamos?** En machine learning, queremos:
- Cambiar fácilmente entre diferentes algoritmos (Regresión Lineal, Random Forest, etc.)
- Añadir nuevos tipos de modelos sin modificar el código existente
- Ocultar la complejidad de la creación de modelos

**¿Cómo lo implementamos?**

1. **Interfaz base** (`models/prediction_model.py`): Define la API común para todos los modelos
   ```python
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

2. **Implementaciones concretas**:
   - `LinearRegressionModel` en `models/linear_regression_model.py`
   - `RandomForestModel` en `models/random_forest_model.py`

3. **La Factory** (`models/model_factory.py`): Crea diferentes tipos de modelos
   ```python
   class ModelFactory:
       @staticmethod
       def create_model(model_type, **kwargs):
           if model_type == 'linear' or model_type == 'linear_regression':
               return LinearRegressionModel()
           elif model_type == 'random_forest':
               return RandomForestModel(...)
           # Podemos añadir más modelos en el futuro
   ```

**Beneficios:**
- Podemos cambiar de un modelo a otro fácilmente en la interfaz web
- Podemos añadir nuevos tipos de modelos sin modificar el código existente
- El código que usa los modelos no necesita conocer detalles de implementación

### 3. Patrón Prototype

**¿Qué es?** Es un patrón que permite clonar objetos existentes sin depender de sus clases concretas.

**¿Por qué lo necesitamos?** En machine learning, queremos:
- Crear múltiples configuraciones de modelos similares
- Experimentar con diferentes hiperparámetros
- Guardar y cargar configuraciones predefinidas

**¿Cómo lo implementamos?**

1. **ModelConfig** (`models/model_config.py`): Clase prototipo para configuraciones
   ```python
   class ModelConfig:
       def __init__(self, model_type='linear', params=None, features=None):
           # Inicialización con valores por defecto

       def clone(self):
           """Crea una copia profunda de la configuración actual"""
           return copy.deepcopy(self)
   ```

2. **ModelConfigRegistry** (`models/model_config_registry.py`): Almacena prototipos predefinidos
   - Contiene configuraciones prediseñadas que los usuarios pueden aplicar

**Beneficios:**
- Los usuarios pueden experimentar con diferentes configuraciones en la interfaz web
- Podemos predefinir configuraciones optimizadas
- Las configuraciones se pueden clonar y modificar sin afectar a las originales

## Resultado de la Refactorización

La aplicación original pasó de tener todo en un solo archivo a esta estructura organizada:

```
laptop-price-predictor/
├── main.py                    # Código principal, ahora más limpio
├── config.json                # Archivo de configuración externo
├── templates/                 # Plantillas HTML para la interfaz web
│   └── index.html
├── static/                    # Archivos estáticos (gráficos, CSS)
└── models/                    # Módulos de machine learning organizados
    ├── config_manager.py      # Singleton para configuración
    ├── model_manager.py       # Singleton para gestión del modelo
    ├── model_factory.py       # Factory para crear modelos
    ├── prediction_model.py    # Interfaz abstracta para modelos
    ├── linear_regression_model.py  # Implementación concreta
    ├── random_forest_model.py     # Implementación concreta
    ├── model_config.py        # Prototype para configuraciones
    └── model_config_registry.py   # Registro de prototipos
```

## Beneficios Concretos

Esta nueva estructura nos permitió:

1. **Añadir múltiples tipos de modelos** (Regresión Lineal, Random Forest) sin modificar el código existente
2. **Permitir a los usuarios cambiar entre modelos** directamente desde la interfaz web
3. **Experimentar con diferentes configuraciones** sin tocar el código
4. **Centralizar la configuración** para facilitar cambios
5. **Mejorar la mantenibilidad** gracias a la separación de responsabilidades

## Conclusión

Los patrones de diseño no complican el código, ¡lo simplifican! Aunque inicialmente parece que añadimos más archivos y clases, en realidad estamos organizando mejor el código para que sea:

- Más fácil de entender
- Más fácil de mantener
- Más fácil de extender

Esto es especialmente importante en aplicaciones de machine learning, donde necesitamos experimentar con diferentes modelos y configuraciones.

## Para Empezar

Si quieres implementar estos patrones en tu propio proyecto:

1. Identifica las responsabilidades principales (gestión de configuración, entrenamiento, predicción)
2. Separa estas responsabilidades en clases diferentes
3. Usa Singleton para componentes que deben ser únicos
4. Usa Factory para crear diferentes tipos de modelos
5. Usa Prototype para configuraciones flexibles

Recuerda: ¡un buen diseño hace que el código sea más simple, no más complicado!