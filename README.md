# Laptop Price Predictor

Este proyecto implementa una aplicación web de predicción de precios de laptops utilizando aprendizaje automático. Sirve como caso de estudio para aprender sobre patrones de diseño de software en un contexto de aplicación de machine learning.

## 📋 Descripción del Proyecto

La aplicación permite a los usuarios:
- Ingresar características de una laptop (peso, RAM, tipo, resolución de pantalla, etc.)
- Obtener una predicción del precio en euros
- Explorar las métricas del modelo y la importancia de las características

Esta aplicación ha sido intencionalmente desarrollada como "código espagueti" para que los estudiantes puedan aplicar patrones de diseño para mejorarla.

## 🛠️ Tecnologías Utilizadas

- **Backend**: FastAPI, Python, scikit-learn
- **Frontend**: HTML, Tailwind CSS, Chart.js
- **Machine Learning**: Regresión Lineal, análisis de características

## 🚀 Configuración e Instalación

1. Clonar el repositorio:
   ```
   git clone https://github.com/tu-usuario/laptop-price-predictor.git
   cd laptop-price-predictor
   ```

2. Crear un entorno virtual e instalar dependencias:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:
   ```
   uvicorn main:app --reload
   ```

4. Acceder a la aplicación en el navegador:
   ```
   http://127.0.0.1:8000/
   ```

## 📊 Dataset

El dataset `laptop_price.csv` contiene información sobre una variedad de laptops, incluyendo:
- Marca y producto
- Tipo de laptop (Gaming, Notebook, etc.)
- Especificaciones técnicas (RAM, peso, resolución de pantalla, CPU)
- Precio en euros

## 📝 Estructura del Proyecto

```
laptop-price-predictor/
├── docs/
│   └── design_patterns/             # Documentación de patrones de diseño
│       ├── singleton_pattern.md     # Ejercicio del patrón Singleton
│       ├── factory_pattern.md       # Ejercicio del patrón Factory
│       ├── prototype_pattern.md     # Ejercicio del patrón Prototype
│       └── solutions/               # Soluciones a los ejercicios
├── templates/                       # Plantillas HTML para la interfaz
│   ├── index.html                   # Página principal con formulario
│   ├── prediction.html              # Página de resultados de predicción
│   └── model_info.html              # Página de información del modelo
├── static/                          # Archivos estáticos (generados dinámicamente)
├── data_science_notebook.ipynb      # Notebook con el análisis original
├── laptop_price.csv                 # Dataset de precios de laptops
├── main.py                          # Código principal de la aplicación
└── requirements.txt                 # Dependencias del proyecto
```

## 🎯 Ejercicios de Patrones de Diseño

Este proyecto incluye ejercicios para implementar los siguientes patrones de diseño:

### 1. Patrón Singleton
Implementar un gestor de configuración y un gestor de modelo que garantice una única instancia.
[Ver ejercicio](docs/design_patterns/singleton_pattern.md)

### 2. Patrón Factory
Crear una fábrica de modelos que permita utilizar diferentes algoritmos de predicción.
[Ver ejercicio](docs/design_patterns/factory_pattern.md)

### 3. Patrón Prototype
Implementar un sistema de prototipos para configuraciones de modelos y estrategias de preprocesamiento.
[Ver ejercicio](docs/design_patterns/prototype_pattern.md)

## 📈 Próximos Pasos

Después de completar los ejercicios básicos, considera estas mejoras:

1. Implementar más algoritmos de predicción (SVR, Gradient Boosting)
2. Añadir validación cruzada para evaluar modelos
3. Implementar guardado/carga de modelos desde disco
4. Crear una API REST separada del frontend
5. Añadir visualizaciones adicionales de los datos

## 📜 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Ver el archivo LICENSE para más detalles.

---

Desarrollado para el curso de Diseño de Software - URP 2025