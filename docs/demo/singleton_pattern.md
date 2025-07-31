# Ejercicio de Patrón de Diseño: Singleton

## Introducción

El patrón Singleton es uno de los patrones de diseño más simples y conocidos. Asegura que una clase tenga **una única instancia** y proporciona un punto de acceso global a ella. Esto es útil cuando exactamente un objeto es necesario para coordinar acciones en todo el sistema.

## Problema en el Código Actual

Examina el archivo `main.py`. Observarás que la gestión de configuración y el acceso al modelo ML se realiza mediante variables globales y funciones que acceden a ellas:

```python
# Global variables that hold state
model = None
features = ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']
feature_importances = {}

# ...

# Load and prepare data for the model
def load_data():
    global model, feature_importances

    # Load the dataset
    df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
    # ...
```

Esto presenta varios problemas:
1. Variables globales que pueden ser modificadas desde cualquier parte del código
2. Acceso no controlado al modelo ML
3. Inicialización repetida si se pierde la referencia al modelo
4. Dificulta la implementación de pruebas unitarias
5. No hay encapsulación de la configuración

## Ejercicio

Tu tarea es refactorizar el código para implementar el patrón Singleton en dos áreas clave:

1. **Crear un ModelManager (Singleton)**:
   - Implementa una clase `ModelManager` que gestione la carga, entrenamiento y predicciones del modelo
   - Asegúrate de que solo exista una instancia de esta clase
   - Encapsula las características del modelo, las importancias y el objeto modelo

2. **Crear un ConfigManager (Singleton)**:
   - Implementa una clase `ConfigManager` que gestione la configuración de la aplicación
   - Incluye parámetros como rutas de archivos, configuración del modelo, etc.

## Directivas de Implementación

### Para el ModelManager:

1. Crea un nuevo archivo `model_manager.py` con la clase `ModelManager`
2. Implementa el método `__new__` para controlar la creación de instancias
3. Mueve la lógica de carga y entrenamiento del modelo desde `load_data()` a esta clase
4. Proporciona métodos para acceder al modelo y realizar predicciones

### Para el ConfigManager:

1. Crea un nuevo archivo `config_manager.py` con la clase `ConfigManager`
2. Implementa un atributo de clase privado para almacenar la única instancia
3. Define las configuraciones (ruta del archivo CSV, nombres de características, etc.)
4. Proporciona métodos para acceder a la configuración

## Cómo Verificar tu Solución

Una implementación correcta del patrón Singleton debe demostrar que:

1. Solo existe una instancia de cada clase Singleton
2. El acceso global a la instancia es posible desde cualquier parte del código
3. No es posible crear instancias adicionales directamente

Ejemplo de prueba (puedes añadir esto a un nuevo archivo `test_singleton.py`):

```python
from model_manager import ModelManager
from config_manager import ConfigManager

# Verificar que siempre obtenemos la misma instancia
model_manager1 = ModelManager()
model_manager2 = ModelManager()
assert model_manager1 is model_manager2  # Deben ser el mismo objeto

config1 = ConfigManager()
config2 = ConfigManager()
assert config1 is config2  # Deben ser el mismo objeto

print("¡Las pruebas del patrón Singleton pasaron con éxito!")
```

## Beneficios Esperados

Al implementar el patrón Singleton:

1. **Acceso controlado**: Acceso centralizado y controlado al modelo ML
2. **Eficiencia**: El modelo se carga solo una vez
3. **Mantenibilidad**: Código más limpio y organizado
4. **Extensibilidad**: Más fácil de modificar y extender
5. **Testabilidad**: Facilita las pruebas unitarias

## Puntos Adicionales

- El Singleton puede implementarse con metaclases en Python, aunque la implementación básica suele ser suficiente
- Considera cómo manejar la concurrencia si la aplicación escalara a múltiples hilos
- Reflexiona sobre si un objeto global es realmente necesario o si existen alternativas mejores para tu caso de uso