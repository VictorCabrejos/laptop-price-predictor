# Explicación del Proyecto Predictor de Precios de Laptops (Versión Actual)

## Introducción

Esta documentación describe la versión actual del proyecto Predictor de Precios de Laptops, la cual está implementada sin patrones de diseño (código espagueti). El objetivo es proporcionar una comprensión clara del flujo de trabajo de machine learning implementado en la aplicación.

## Estructura General

La aplicación actual es una web simple desarrollada con FastAPI que permite a los usuarios:
- Ingresar características de una laptop
- Obtener una predicción del precio estimado
- Ver información sobre el modelo y sus métricas

Todo el código está acumulado principalmente en un único archivo `main.py`, lo que dificulta su mantenimiento y extensibilidad.

## Flujo de Trabajo de Machine Learning

El proceso de machine learning en la aplicación actual se puede dividir en cuatro partes principales:

### 1. Obtención de Datos (Data Sourcing)

```python
# En la función load_data() del archivo main.py
df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
```

- Los datos se cargan directamente desde un archivo CSV fijo
- No hay mecanismos para actualizar los datos o incorporar nuevas fuentes
- La ruta del archivo está codificada directamente en el código
- No hay manejo de errores si el archivo no existe o está corrupto

### 2. Procesamiento de Datos (Data Wrangling)

```python
# En la función load_data() del archivo main.py
# Clean and transform data
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df = pd.get_dummies(df, columns=['TypeName'], dtype='int')
df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(float)
df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)
```

- La transformación de datos está mezclada con la carga de datos
- El mismo código de preprocesamiento se duplica en diferentes partes de la aplicación
- No hay forma de experimentar con diferentes estrategias de preprocesamiento
- Las transformaciones están codificadas de forma rígida y son difíciles de modificar

### 3. Modelado (Modeling)

```python
# En la función load_data() del archivo main.py
# Feature selection
X = df[features]
y = df['Price_euros']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
```

- Solo se implementa un tipo de modelo (Regresión Lineal)
- No hay forma de experimentar con diferentes algoritmos
- La selección de características está hardcodeada
- El entrenamiento del modelo está mezclado con la carga y transformación de datos
- No hay forma de guardar o cargar modelos entrenados previamente

### 4. Evaluación (Evaluation)

```python
# En la función load_data() del archivo main.py
# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Get feature importances
for i, feature in enumerate(features):
    feature_importances[feature] = float(model.coef_[i])
```

- Las métricas de evaluación están limitadas a MAE y R²
- No hay validación cruzada o técnicas avanzadas de evaluación
- Los resultados se almacenan en variables globales
- No hay forma de comparar diferentes modelos o configuraciones

## Problemas de Diseño Actuales

1. **Acoplamiento Alto**: Todos los componentes están fuertemente acoplados en un solo archivo.
2. **Variables Globales**: El estado se mantiene mediante variables globales que pueden ser modificadas desde cualquier parte del código.
3. **Falta de Modularidad**: Es difícil modificar una parte sin afectar otras.
4. **Duplicación de Código**: El mismo código se repite en varios lugares.
5. **Falta de Extensibilidad**: Agregar nuevas características requiere modificar código existente.
6. **Dificultad para Pruebas**: Es casi imposible hacer pruebas unitarias.

## Interacción Usuario-Aplicación

1. El usuario ingresa a la página principal (`/`)
2. Completa el formulario con características de una laptop
3. Al enviar el formulario, se hace una petición POST a la ruta `/predict`
4. La función `predict()` en main.py:
   - Verifica si el modelo está cargado, si no, llama a `load_data()`
   - Prepara los datos de entrada
   - Realiza la predicción usando el modelo global
   - Devuelve la predicción en la template `prediction.html`

## Visualización del Flujo

```
Usuario → Formulario → POST /predict → Procesamiento de entrada → Modelo → Predicción → Visualización
```

## Conclusión

La versión actual del proyecto funciona, pero presenta muchos problemas de diseño que dificultan su mantenimiento, extensibilidad y prueba. Los ejercicios de patrones de diseño proporcionados tienen como objetivo refactorizar esta aplicación para resolver estos problemas mientras se mantiene la misma funcionalidad.