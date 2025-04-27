# Implementaciones de Patrones de Diseño en el Proyecto Laptop Price Predictor

Este documento describe las implementaciones de los cuatro patrones de diseño aplicados en el proyecto de predicción de precios de laptops. Cada patrón aborda un problema específico y mejora la estructura y flexibilidad del código.

## Índice
1. [Patrón Strategy](#patron-strategy)
2. [Patrón Singleton](#patron-singleton)
3. [Patrón Factory](#patron-factory)
4. [Patrón Prototype](#patron-prototype)

<a id="patron-strategy"></a>
## 1. Patrón Strategy

### Problema Identificado
En la implementación original, la lógica de preprocesamiento de datos estaba dispersa y duplicada en varias partes del código, dificultando su mantenimiento y extensión. Cuando se necesitaba cambiar el preprocesamiento, había que modificar múltiples secciones del código.

### Solución Implementada
Implementamos el patrón Strategy para encapsular diferentes algoritmos de preprocesamiento y permitir su intercambio en tiempo de ejecución. La solución consta de:

1. **Interfaz Abstracta**: `DataPreprocessor` define la interfaz común para todas las estrategias de preprocesamiento.
2. **Estrategias Concretas**: Creamos tres implementaciones diferentes:
   - `StandardPreprocessor`: Implementa el preprocesamiento básico original.
   - `AdvancedPreprocessor`: Añade normalización y manejo de valores atípicos.
   - `MinimalPreprocessor`: Utiliza un conjunto reducido de características para modelos más simples.
3. **Contexto**: La clase `ModelManager` utiliza la estrategia seleccionada y proporciona métodos para cambiarla en tiempo de ejecución.

### Código Clave
```python
class ModelManager:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or StandardPreprocessor()
        self.model = None

    def set_preprocessor(self, preprocessor):
        """Cambiar la estrategia de preprocesamiento en tiempo de ejecución"""
        self.preprocessor = preprocessor

    def train_model(self, data_path):
        # Usar la estrategia de preprocesamiento seleccionada
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_training_data(df)
        # ...
```

### Ventajas de la Implementación
1. **Flexibilidad**: Podemos cambiar el algoritmo de preprocesamiento sin modificar el código cliente.
2. **Eliminación de Duplicación**: La lógica de preprocesamiento está centralizada en las clases de estrategia.
3. **Mantenibilidad**: Las modificaciones en un algoritmo no afectan a los demás.
4. **Extensibilidad**: Es fácil añadir nuevas estrategias implementando la interfaz `DataPreprocessor`.
5. **Experimentación**: Facilita probar diferentes enfoques de preprocesamiento para optimizar el modelo.

### Por Qué Esta Implementación
Elegimos crear tres estrategias diferentes para mostrar la versatilidad del patrón:
1. La estrategia estándar mantiene la funcionalidad original para garantizar la compatibilidad.
2. La estrategia avanzada demuestra cómo se pueden añadir técnicas más sofisticadas.
3. La estrategia minimal ilustra cómo se puede simplificar el preprocesamiento para casos específicos.

Esta estructura permite a los estudiantes entender cómo el mismo conjunto de datos puede ser procesado de diferentes maneras según los requisitos, sin cambiar la interfaz del sistema.

<a id="patron-singleton"></a>
## 2. Patrón Singleton

### Problema Identificado
El código original utilizaba variables globales para almacenar el modelo entrenado y las configuraciones, lo que podía llevar a problemas de consistencia, dificultad en las pruebas y acceso no controlado a estos recursos compartidos.

### Solución Implementada
Implementamos dos clases Singleton:

1. **ModelManager (Singleton)**: Gestiona una única instancia del modelo de predicción.
2. **ConfigManager (Singleton)**: Centraliza la configuración de la aplicación.

### Código Clave
```python
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.features = []
            cls._instance.feature_importances = {}
        return cls._instance

    # Resto de métodos...
```

### Ventajas de la Implementación
1. **Acceso Controlado**: Un único punto de acceso al modelo y la configuración.
2. **Eficiencia**: El modelo se carga y entrena solo una vez.
3. **Consistencia**: Se garantiza que todos los componentes trabajen con la misma instancia.
4. **Organización**: El código queda más limpio al eliminar variables globales.
5. **Testabilidad**: Es más fácil mockear un Singleton para pruebas unitarias.

### Por Qué Esta Implementación
Optamos por una implementación tradicional del Singleton usando un atributo de clase privado `_instance` y sobrescribiendo el método `__new__`. Aunque existen otras opciones como utilizar metaclases, esta implementación es más clara para fines didácticos y suficiente para las necesidades del proyecto.

El patrón se aplicó a dos clases diferentes para demostrar cómo puede usarse en distintos contextos dentro de la misma aplicación, manteniendo la separación de responsabilidades.

<a id="patron-factory"></a>
## 3. Patrón Factory

### Problema Identificado
En la implementación original, el código estaba fuertemente acoplado a un único tipo de modelo (`LinearRegression`), lo que dificultaba probar o implementar diferentes algoritmos de aprendizaje automático.

### Solución Implementada
Creamos un sistema de Factory para la creación de diferentes tipos de modelos:

1. **Interfaz Abstracta**: `PredictionModel` define la interfaz común para todos los modelos.
2. **Clases Concretas**: Implementaciones específicas como `LinearRegressionModel`, `RandomForestModel`, etc.
3. **Factory**: La clase `ModelFactory` que centraliza la creación de los diferentes tipos de modelos.

### Código Clave
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type, **params):
        if model_type == "linear":
            return LinearRegressionModel(**params)
        elif model_type == "random_forest":
            return RandomForestModel(**params)
        elif model_type == "gradient_boosting":
            return GradientBoostingModel(**params)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
```

### Ventajas de la Implementación
1. **Desacoplamiento**: El código cliente no depende de clases concretas de modelos.
2. **Flexibilidad**: Es fácil cambiar entre diferentes algoritmos de predicción.
3. **Mantenibilidad**: La adición de nuevos tipos de modelos no requiere cambios en el código existente.
4. **Encapsulación**: Los detalles de creación están centralizados en la Factory.
5. **Experimentación**: Facilita la comparación entre diferentes algoritmos.

### Por Qué Esta Implementación
Elegimos una implementación de Factory Method con un método estático para mantener la simplicidad. Aunque podríamos haber usado una Abstract Factory para manejar familias de objetos relacionados, el Factory Method es suficiente para ilustrar el patrón y resolver el problema específico.

Implementamos tres tipos diferentes de modelos para mostrar cómo el mismo patrón puede adaptarse a diferentes algoritmos de machine learning, cada uno con sus propios requisitos y características.

<a id="patron-prototype"></a>
## 4. Patrón Prototype

### Problema Identificado
Para experimentar con diferentes configuraciones de modelos y preprocesamiento, el código original requería duplicación significativa, lo que dificultaba la creación y gestión de múltiples configuraciones experimentales.

### Solución Implementada
Desarrollamos un sistema basado en el patrón Prototype para:

1. **Configuraciones Clonables**: Clases `ModelConfig` y `PreprocessingConfig` con método `clone()`.
2. **Protototipos Predefinidos**: Configuraciones base que pueden ser clonadas y modificadas.
3. **Modificaciones No Destructivas**: Métodos que devuelven nuevas instancias modificadas sin alterar el original.

### Código Clave
```python
class ModelConfig:
    def __init__(self, algorithm="linear_regression", hyperparameters=None):
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters or {}

    def clone(self):
        """Crea una copia profunda de esta configuración"""
        return copy.deepcopy(self)

    def set_parameter(self, param_name, param_value):
        """Modifica un hiperparámetro específico y devuelve una nueva instancia"""
        new_config = self.clone()
        new_config.hyperparameters[param_name] = param_value
        return new_config
```

### Ventajas de la Implementación
1. **Reusabilidad**: Las configuraciones existentes sirven como base para nuevas variantes.
2. **Eficiencia**: Evita la recreación completa de objetos complejos.
3. **Flexibilidad**: Facilita la experimentación con múltiples configuraciones.
4. **Mantenibilidad**: Cambios en una configuración base se propagan automáticamente a sus derivados.
5. **Limpieza del Código**: Elimina la duplicación al crear configuraciones similares.

### Por Qué Esta Implementación
Utilizamos el módulo `copy` de Python para implementar la clonación profunda, garantizando que las modificaciones en un clon no afecten al original. Esta implementación es particularmente útil en machine learning, donde la experimentación con diferentes configuraciones es una práctica común.

Implementamos el patrón tanto para configuraciones de modelos como para preprocesamiento para mostrar su versatilidad en diferentes componentes del sistema, permitiendo combinar diferentes configuraciones de forma flexible.

## Conclusión

Los cuatro patrones de diseño implementados trabajan en conjunto para crear un sistema flexible, mantenible y extensible:

1. **Strategy** permite intercambiar algoritmos de preprocesamiento.
2. **Singleton** garantiza acceso coordinado a recursos compartidos.
3. **Factory** facilita la creación de diferentes tipos de modelos.
4. **Prototype** permite experimentar con múltiples configuraciones.

Esta combinación de patrones proporciona una estructura robusta que resuelve los problemas identificados en el código original y facilita futuras extensiones y modificaciones del sistema de predicción de precios de laptops.