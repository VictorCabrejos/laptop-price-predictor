# Tarea Grupal: Implementación del Patrón Factory para Visualizaciones

## Introducción

En proyectos de ciencia de datos y aprendizaje automático, las visualizaciones son fundamentales para entender los datos, analizar resultados y comunicar hallazgos. Actualmente, nuestro proyecto de predicción de precios de laptops utiliza visualizaciones creadas directamente en el código, lo que dificulta su reutilización y mantenimiento.

## Problema Actual

En el archivo `main.py`, la creación de visualizaciones está directamente en el código de la función `load_data()`:

```python
# Generate a plot for the model
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Laptop Prices')
plt.savefig('static/prediction_plot.png')

# Create feature importance plot
plt.figure(figsize=(10, 6))
features_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_
})
features_df = features_df.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
```

Esto presenta varios problemas:
1. **Código no reutilizable**: Para crear visualizaciones similares en otra parte, debemos duplicar código
2. **Difícil de mantener**: Cambios en el estilo o formato requieren modificar múltiples secciones
3. **Falta de flexibilidad**: No es fácil añadir nuevos tipos de gráficos o personalizarlos
4. **Mezcla de responsabilidades**: La función `load_data()` tiene demasiadas responsabilidades

## Objetivo de la Tarea

Implementar el patrón Factory para la creación de visualizaciones en el proyecto de predicción de precios de laptops. Esto permitirá crear diferentes tipos de gráficos de manera flexible y reutilizable.

## ¿Qué es el Patrón Factory?

El patrón Factory es un patrón de diseño creacional que proporciona una interfaz para crear objetos sin especificar sus clases concretas. En nuestro caso, queremos una fábrica que cree diferentes tipos de visualizaciones sin que el código cliente necesite conocer los detalles de implementación de cada tipo de gráfico.

## Instrucciones

### 1. Crear una Interfaz para las Visualizaciones

Primero, creen una clase base abstracta que defina la interfaz común para todas las visualizaciones:

```python
# visualizations/base_visualization.py
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization(ABC):
    def __init__(self, title="", figsize=(10, 6)):
        self.title = title
        self.figsize = figsize

    @abstractmethod
    def plot(self, data):
        """
        Método para crear la visualización
        :param data: Datos necesarios para la visualización
        """
        pass

    def save(self, filepath):
        """
        Guarda la figura actual
        :param filepath: Ruta donde guardar la visualización
        """
        plt.savefig(filepath)
        plt.close()
```

### 2. Implementar Visualizaciones Concretas

Implementen al menos tres tipos diferentes de visualizaciones que hereden de la clase base:

1. **ScatterPlot**: Para visualizar predicciones vs valores reales
2. **BarPlot**: Para visualizar importancia de características
3. **HistogramPlot**: Para visualizar distribuciones de datos

Ejemplo de implementación para ScatterPlot:

```python
# visualizations/scatter_plot.py
from visualizations.base_visualization import Visualization
import matplotlib.pyplot as plt
import numpy as np

class ScatterPlot(Visualization):
    def __init__(self, title="Scatter Plot", figsize=(10, 6), xlabel="X", ylabel="Y"):
        super().__init__(title, figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, data):
        """
        Crea un gráfico de dispersión
        :param data: Diccionario con 'x' e 'y' como arrays de datos
        """
        plt.figure(figsize=self.figsize)
        plt.scatter(data['x'], data['y'], alpha=0.7)

        # Agregar línea de referencia
        if 'add_reference_line' in data and data['add_reference_line']:
            min_val = min(min(data['x']), min(data['y']))
            max_val = max(max(data['x']), max(data['y']))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.tight_layout()
        return plt
```

### 3. Crear la Factory de Visualizaciones

Implementen una clase Factory que pueda crear los diferentes tipos de visualizaciones:

```python
# visualizations/visualization_factory.py
from visualizations.scatter_plot import ScatterPlot
from visualizations.bar_plot import BarPlot
from visualizations.histogram_plot import HistogramPlot

class VisualizationFactory:
    @staticmethod
    def create_visualization(viz_type, **kwargs):
        """
        Crea una visualización del tipo especificado
        :param viz_type: Tipo de visualización ('scatter', 'bar', 'histogram', etc)
        :param kwargs: Argumentos adicionales para la visualización específica
        :return: Objeto de visualización
        """
        if viz_type == 'scatter':
            return ScatterPlot(**kwargs)
        elif viz_type == 'bar':
            return BarPlot(**kwargs)
        elif viz_type == 'histogram':
            return HistogramPlot(**kwargs)
        else:
            raise ValueError(f"Tipo de visualización no soportado: {viz_type}")
```

### 4. Refactorizar el Código Principal

Modifiquen la función `load_data()` para utilizar la Factory de visualizaciones:

```python
def load_data():
    # ... código existente ...

    # Evaluamos el modelo
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Usamos la Factory para crear visualizaciones
    factory = VisualizationFactory()

    # Gráfico de predicciones
    scatter = factory.create_visualization(
        'scatter',
        title='Actual vs Predicted Laptop Prices',
        xlabel='Actual',
        ylabel='Predicted'
    )
    scatter_data = {
        'x': y_test,
        'y': y_pred,
        'add_reference_line': True
    }
    scatter.plot(scatter_data)
    scatter.save('static/prediction_plot.png')

    # Gráfico de importancia de características
    bar = factory.create_visualization(
        'bar',
        title='Feature Importance',
        xlabel='Importance',
        ylabel='Feature'
    )
    features_data = {
        'x': model.coef_,
        'y': features,
        'sort': True
    }
    bar.plot(features_data)
    bar.save('static/feature_importance.png')

    print(f"Model trained with MAE: {mae:.2f} and R²: {r2:.2f}")
```

## Estructura de Archivos Sugerida

```
laptop-price-predictor/
├── visualizations/
│   ├── __init__.py
│   ├── base_visualization.py
│   ├── scatter_plot.py
│   ├── bar_plot.py
│   ├── histogram_plot.py
│   └── visualization_factory.py
```

## Criterios de Evaluación

1. **Implementación correcta del patrón Factory** (40%)
   - Interfaz abstracta bien definida
   - Clases concretas que implementan la interfaz
   - Factory que crea los diferentes tipos de visualizaciones

2. **Funcionalidad** (30%)
   - Las visualizaciones deben generar gráficos correctos
   - La aplicación debe funcionar como antes pero usando la nueva estructura

3. **Extensibilidad** (20%)
   - Debe ser fácil añadir nuevos tipos de visualizaciones
   - La solución debe seguir principios SOLID

4. **Documentación y estilo** (10%)
   - Código bien documentado
   - Nombres descriptivos
   - Estructura de clases clara

## Entrega

El trabajo se debe entregar como un Pull Request al repositorio del proyecto incluyendo:

1. Todos los archivos de código necesarios
2. Un archivo README.md explicando:
   - Cómo se implementó el patrón Factory
   - Qué tipos de visualizaciones se implementaron
   - Cómo añadir nuevas visualizaciones al sistema

## ¿Por qué es Útil Este Patrón?

### En Este Proyecto
- **Reutilización**: Pueden crear visualizaciones consistentes en diferentes partes del proyecto
- **Mantenibilidad**: Cambios de estilo o formato se hacen en un solo lugar
- **Extensibilidad**: Fácil agregar nuevos tipos de gráficos sin modificar código existente

### En Proyectos de Tesis y Otros Proyectos de ML
- **Estandarización**: Pueden crear un estándar para todas las visualizaciones en su proyecto
- **Modularidad**: La lógica de visualización está separada de la lógica de negocio
- **Facilidad de uso**: Simplifica la creación de visualizaciones complejas
- **Consistencia**: Todas las visualizaciones siguen el mismo estilo y formato

## Consejos

1. Comiencen creando la estructura básica: la interfaz y una implementación concreta
2. Implementen la Factory después de tener al menos dos visualizaciones concretas
3. Refactoricen el código existente para usar la nueva estructura
4. Añadan más tipos de visualizaciones según sea necesario
5. Piensen en qué parámetros son comunes a todas las visualizaciones y cuáles son específicos