# Ejercicio de Patrón de Diseño: Strategy

## Introducción

El patrón Strategy define una familia de algoritmos, encapsula cada uno de ellos y los hace intercambiables. Este patrón permite que el algoritmo varíe independientemente de los clientes que lo utilizan. Es particularmente útil cuando necesitamos seleccionar diferentes comportamientos en tiempo de ejecución.

## Problema en el Código Actual

En el archivo `main.py`, la lógica de preprocesamiento de datos está fuertemente acoplada a la ruta de predicción:

```python
# Dentro de la ruta de predicción
def predict(request, weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram):
    # Preparar la entrada
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])
    # ...
```

Y también en la función de carga de datos:

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
1. **Código duplicado**: La lógica de transformación de datos se repite en varios lugares
2. **Falta de modularidad**: No podemos cambiar fácilmente la estrategia de preprocesamiento
3. **Dificultad de mantenimiento**: Cambios en el preprocesamiento requieren modificar múltiples partes del código
4. **Falta de flexibilidad**: No podemos experimentar con diferentes técnicas de preprocesamiento

## Ejercicio

Tu tarea es refactorizar el código para implementar el patrón Strategy para el preprocesamiento de datos:

1. **Crear una interfaz/clase abstracta para el preprocesamiento**:
   - Implementa una clase base `DataPreprocessor` con métodos comunes
   - Define métodos como `preprocess_training_data()` y `preprocess_input_data()`

2. **Implementar diferentes estrategias de preprocesamiento**:
   - Implementa `StandardPreprocessor` que realiza el preprocesamiento actual
   - Implementa `AdvancedPreprocessor` que añada más transformaciones (ej. normalización, detección de valores atípicos)
   - Opcionalmente, implementa otras estrategias de preprocesamiento

3. **Modificar el código cliente para usar estas estrategias**:
   - Refactoriza `main.py` para usar las estrategias de preprocesamiento
   - Permite cambiar dinámicamente la estrategia utilizada

## Directivas de Implementación

### Para la interfaz de estrategia:

1. Crea un archivo `data_preprocessor.py` con una clase abstracta:

```python
from abc import ABC, abstractmethod

class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess_training_data(self, data):
        """Preprocesa los datos de entrenamiento y devuelve X, y"""
        pass

    @abstractmethod
    def preprocess_input_data(self, input_data):
        """Preprocesa un único punto de datos para predicción"""
        pass

    @abstractmethod
    def get_feature_names(self):
        """Devuelve los nombres de las características después del preprocesamiento"""
        pass
```

### Para las estrategias concretas:

1. Implementa `StandardPreprocessor` en un archivo separado
2. Implementa `AdvancedPreprocessor` en otro archivo
3. Cada estrategia debe implementar los métodos abstractos de la clase base

### Para el contexto:

1. Modifica el código para usar estas estrategias:
   - Crea un contexto (por ejemplo, en `ModelManager`) que utilice la estrategia seleccionada
   - Proporciona un método para cambiar de estrategia en tiempo de ejecución

## Ejemplo de Implementación

La estructura básica podría ser:

```python
# En standard_preprocessor.py
class StandardPreprocessor(DataPreprocessor):
    def preprocess_training_data(self, df):
        df = df.copy()
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df = pd.get_dummies(df, columns=['TypeName'], dtype='int')
        df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(int)
        df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        X = df[self.get_feature_names()]
        y = df['Price_euros']
        return X, y

    def preprocess_input_data(self, input_data):
        # Transformar un solo punto de datos para predicción
        # Aquí asumimos que input_data ya está en el formato correcto
        return input_data

    def get_feature_names(self):
        return ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']
```

```python
# En model_manager.py o contexto similar
class ModelManager:
    def __init__(self):
        self.model = None
        self.preprocessor = StandardPreprocessor()  # Estrategia por defecto

    def set_preprocessor(self, preprocessor):
        """Cambiar la estrategia de preprocesamiento"""
        self.preprocessor = preprocessor

    def train(self, data):
        X, y = self.preprocessor.preprocess_training_data(data)
        # Entrenar modelo con datos preprocesados
        # ...
```

## Cómo Verificar tu Solución

Una implementación correcta del patrón Strategy debe demostrar que:

1. Puedes intercambiar estrategias de preprocesamiento sin modificar el código cliente
2. El código de preprocesamiento está encapsulado en clases separadas
3. Es fácil añadir nuevas estrategias de preprocesamiento

Ejemplo de prueba:

```python
# Crear el contexto con la estrategia estándar
model_manager = ModelManager()
model_manager.train(data)  # Usa StandardPreprocessor

# Cambiar a la estrategia avanzada
advanced_preprocessor = AdvancedPreprocessor()
model_manager.set_preprocessor(advanced_preprocessor)
model_manager.train(data)  # Ahora usa AdvancedPreprocessor
```

## Beneficios Esperados

Al implementar el patrón Strategy:

1. **Eliminación de código duplicado**: La lógica de preprocesamiento está en un solo lugar
2. **Modularidad mejorada**: Fácil de mantener y extender
3. **Flexibilidad**: Capacidad para cambiar estrategias en tiempo de ejecución
4. **Mejor organización**: Separación clara de responsabilidades
5. **Facilidad de prueba**: Cada estrategia puede probarse de forma independiente

## Mejoras Adicionales

- Considera implementar un mecanismo de configuración para cada estrategia
- Explora cómo este patrón se complementa con el patrón Factory (para crear diferentes estrategias)
- Evalúa el rendimiento de diferentes estrategias de preprocesamiento
- Reflexiona sobre cómo este patrón cumple con el principio de Open/Closed (parte de SOLID)