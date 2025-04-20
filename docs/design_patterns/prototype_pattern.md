# Ejercicio de Patrón de Diseño: Prototype

## Introducción

El patrón Prototype es un patrón de diseño creacional que permite crear nuevos objetos clonando un objeto existente, conocido como prototipo. Este patrón es útil cuando la creación de un objeto es costosa o compleja y existe un objeto similar que ya ha sido instanciado. En lugar de crear un nuevo objeto desde cero, simplemente clonamos el prototipo y lo modificamos según sea necesario.

## Problema en el Código Actual

En el archivo `main.py`, observamos que no hay manera eficiente de crear múltiples configuraciones de modelos o conjuntos de hiperparámetros para experimentación:

```python
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
```

Y si quisiéramos probar diferentes configuraciones de procesamiento de datos, tendríamos que duplicar mucho código:

```python
def load_data():
    # ...
    # Clean and transform data
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df = pd.get_dummies(df, columns=['TypeName'], dtype='int')
    # ...
```

Esto presenta varios problemas:

1. **Duplicación de código**: Para crear diferentes configuraciones, debemos duplicar código
2. **Inflexibilidad**: Es difícil experimentar con diferentes configuraciones de modelo o preprocesamiento
3. **Dificultad de mantenimiento**: Cambios en una configuración requieren actualizaciones en todos los lugares donde se duplicó el código
4. **Falta de reutilización**: No podemos reutilizar componentes existentes de manera eficiente

## Ejercicio

Tu tarea es refactorizar el código para implementar el patrón Prototype en dos áreas:

1. **Crear un sistema de prototipos para configuraciones de modelos**:
   - Implementa una clase base `ModelConfig` con métodos para clonar y modificar configuraciones
   - Crea diferentes configuraciones prototipo que puedan ser clonadas y ajustadas

2. **Crear un sistema de prototipos para estrategias de preprocesamiento**:
   - Implementa una clase base `PreprocessingConfig` con métodos para clonar y modificar configuraciones de preprocesamiento
   - Crea diferentes configuraciones prototipo para diversos escenarios de preprocesamiento

## Directivas de Implementación

### Para la clase base de configuración de modelos:

1. Crea un nuevo archivo `model_config.py` con una clase que implemente el patrón Prototype:

```python
import copy

class ModelConfig:
    def __init__(self, algorithm="linear_regression", hyperparameters=None):
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters or {}

    def clone(self):
        """Crea una copia profunda de esta configuración"""
        return copy.deepcopy(self)

    def set_parameter(self, param_name, param_value):
        """Modifica un hiperparámetro específico"""
        new_config = self.clone()
        new_config.hyperparameters[param_name] = param_value
        return new_config

    def set_algorithm(self, algorithm):
        """Cambia el algoritmo de aprendizaje"""
        new_config = self.clone()
        new_config.algorithm = algorithm
        return new_config
```

### Para las configuraciones prototipo de modelos:

Crea prototipos predefinidos que puedan ser clonados:

```python
# En model_config.py o un archivo separado

def create_linear_regression_prototype():
    return ModelConfig(
        algorithm="linear_regression",
        hyperparameters={}
    )

def create_random_forest_prototype():
    return ModelConfig(
        algorithm="random_forest",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2
        }
    )

def create_gradient_boosting_prototype():
    return ModelConfig(
        algorithm="gradient_boosting",
        hyperparameters={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    )
```

### Para el sistema de prototipos de preprocesamiento:

1. Crea un nuevo archivo `preprocessing_config.py` con una clase similar:

```python
import copy

class PreprocessingConfig:
    def __init__(self, scaling=None, categorical_encoding=None, feature_selection=None):
        self.scaling = scaling or "none"  # none, standard, minmax
        self.categorical_encoding = categorical_encoding or "one_hot"  # one_hot, label, target
        self.feature_selection = feature_selection or []  # lista de características

    def clone(self):
        """Crea una copia profunda de esta configuración"""
        return copy.deepcopy(self)

    def set_scaling(self, scaling):
        new_config = self.clone()
        new_config.scaling = scaling
        return new_config

    def set_categorical_encoding(self, encoding):
        new_config = self.clone()
        new_config.categorical_encoding = encoding
        return new_config

    def set_feature_selection(self, features):
        new_config = self.clone()
        new_config.feature_selection = features
        return new_config
```

## Cómo Verificar tu Solución

Una implementación correcta del patrón Prototype debe demostrar que:

1. Puedes crear múltiples configuraciones basadas en prototipos sin duplicar código
2. Las modificaciones a una configuración clonada no afectan al prototipo original
3. Puedes crear rápidamente diferentes variaciones de configuraciones

Ejemplo de prueba:

```python
# Crear prototipos básicos
linear_prototype = create_linear_regression_prototype()
rf_prototype = create_random_forest_prototype()

# Clonar y modificar para experimentos
experiment1 = rf_prototype.clone()
experiment1.set_parameter("n_estimators", 200)

experiment2 = rf_prototype.clone()
experiment2.set_parameter("max_depth", 15)

# Verificar que los cambios no afectan al prototipo original
assert rf_prototype.hyperparameters["n_estimators"] == 100
assert experiment1.hyperparameters["n_estimators"] == 200
assert experiment2.hyperparameters["max_depth"] == 15

print("¡Las pruebas del patrón Prototype pasaron con éxito!")
```

## Beneficios Esperados

Al implementar el patrón Prototype:

1. **Reusabilidad**: Puedes reutilizar configuraciones existentes como base para nuevas configuraciones
2. **Flexibilidad**: Facilita la experimentación con diferentes configuraciones de modelos y preprocesamiento
3. **Mantenibilidad**: Evita la duplicación de código
4. **Escalabilidad**: Facilita la adición de nuevos tipos de configuraciones sin modificar código existente
5. **Eficiencia**: Reduce el costo de creación de objetos complejos

## Puntos Adicionales

- El patrón Prototype es especialmente útil en Python gracias al módulo `copy` que facilita la clonación profunda de objetos
- Considera cómo este patrón se complementa con el patrón Factory: la Factory puede utilizar prototipos para crear objetos complejos
- Reflexiona sobre cómo este patrón facilita la implementación de experimentación de modelos de machine learning
- Piensa en otras áreas del código donde el patrón Prototype podría ser aplicado, como configuraciones de visualización o estrategias de validación