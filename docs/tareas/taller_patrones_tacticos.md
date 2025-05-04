# Taller Grupal: Implementación de un Patrón Táctico

## Introducción

En este taller práctico, aplicaremos **uno** de los patrones de diseño tácticos estudiados (Adapter, Decorator o Facade) para mejorar un componente del proyecto Laptop Price Predictor. Este ejercicio permitirá comprender cómo los patrones de diseño pueden transformar un código monolítico en una solución más mantenible y extensible.

## Problema Actual

Actualmente, la funcionalidad para exportar las predicciones a diferentes formatos está implementada de manera rudimentaria y no estructurada. El código "spaguetti" que se muestra a continuación está directamente en el archivo `main.py` y mezcla diferentes responsabilidades:

```python
# Función para exportar predicción a diferentes formatos
@app.get("/export/{format}")
async def export_prediction(request: Request, format: str, price: float):
    if not price:
        return {"error": "No hay predicción para exportar"}

    if format == "json":
        # Exportar a JSON
        content = {
            "prediction": {
                "price": price,
                "currency": "EUR",
                "timestamp": datetime.now().isoformat(),
                "model": "LinearRegression"
            }
        }

        # Código para guardar el JSON en un archivo
        with open("static/prediction.json", "w") as f:
            json.dump(content, f, indent=2)

        return FileResponse(
            path="static/prediction.json",
            filename="laptop_prediction.json",
            media_type="application/json"
        )

    elif format == "csv":
        # Exportar a CSV
        import csv

        csv_data = [
            ["price", "currency", "timestamp", "model"],
            [str(price), "EUR", datetime.now().isoformat(), "LinearRegression"]
        ]

        with open("static/prediction.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        return FileResponse(
            path="static/prediction.csv",
            filename="laptop_prediction.csv",
            media_type="text/csv"
        )

    elif format == "pdf":
        # Exportar a PDF
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        pdf_path = "static/prediction.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "Laptop Price Prediction")
        c.drawString(100, 730, f"Price: {price} EUR")
        c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(100, 690, "Model: LinearRegression")
        c.save()

        return FileResponse(
            path=pdf_path,
            filename="laptop_prediction.pdf",
            media_type="application/pdf"
        )

    else:
        return {"error": f"Formato '{format}' no soportado"}
```

Este código tiene varios problemas:

1. Mezcla múltiples responsabilidades en una sola función
2. Es difícil extender para soportar nuevos formatos
3. No hay abstracción entre la lógica de exportación y la API web
4. Repite lógica de construcción de datos en cada formato
5. Es difícil de testear y mantener

## Objetivo del Taller

**Refactorizar este componente aplicando UN patrón de diseño táctico** a su elección entre:

1. **Patrón Adapter**
2. **Patrón Decorator**
3. **Patrón Facade**

Cada equipo debe **elegir uno solo** de estos patrones para implementar. La elección debe justificarse en el informe final.

## Instrucciones

### 1. Elegir un Patrón

Analicen el código actual y decidan qué patrón es más adecuado para resolver los problemas identificados. Justifiquen su elección.

### 2. Diseñar la Solución

Antes de comenzar a programar:
- Dibujen un diagrama de clases de la solución
- Identifiquen las interfaces y clases necesarias
- Decidan cómo se relacionarán los componentes

### 3. Implementar el Patrón Elegido

Dependiendo del patrón elegido, enfóquense en los siguientes aspectos:

#### Si eligen el Patrón Adapter:
- Definan una interfaz común `ExporterInterface`
- Creen adaptadores para los diferentes formatos
- Implementen un mecanismo para seleccionar el adaptador adecuado

#### Si eligen el Patrón Decorator:
- Identifiquen la funcionalidad base
- Definan una interfaz `Exporter` común
- Implementen decoradores para añadir funcionalidades como:
  * Validación de datos
  * Registro de eventos (logging)
  * Compresión de archivos

#### Si eligen el Patrón Facade:
- Identifiquen los subsistemas involucrados
- Creen una fachada que simplifique la interacción con estos subsistemas
- Encapsulen la lógica compleja

### 4. Modificar el Endpoint Existente

Modifiquen el endpoint `/export/{format}` para que utilice el patrón implementado.

## Estructura Sugerida

Aquí hay una estructura sugerida de carpetas y archivos, pero pueden adaptarla según sus necesidades:

```
laptop-price-predictor/
├── exporters/
│   ├── __init__.py
│   └── [archivos según el patrón elegido]
```

## Entrega

El trabajo debe entregarse en una semana con los siguientes elementos:

1. Código fuente de la implementación
2. Un informe breve (máximo 3 páginas) que incluya:
   - Justificación del patrón elegido
   - Diagrama de clases de la solución
   - Explicación de cómo se aplicó el patrón
   - Reflexión sobre las ventajas de la nueva implementación
   - Dificultades encontradas y cómo se resolvieron

## Criterios de Evaluación

1. **Comprensión del patrón** (30%)
   - Aplicación correcta del patrón
   - Justificación adecuada de la elección

2. **Implementación** (40%)
   - Código funcional
   - Separación de responsabilidades
   - Seguimiento de buenas prácticas

3. **Extensibilidad** (20%)
   - Facilidad para añadir nuevos formatos o funcionalidades
   - Adherencia a principios SOLID

4. **Informe y documentación** (10%)
   - Claridad en la explicación
   - Calidad del diagrama de clases

## Recursos Adicionales

- Ejemplos de implementación de patrones en `docs/design_patterns/`
- Documentación oficial de FastAPI para `FileResponse`
- Capítulos sobre patrones de diseño del libro "Design Patterns: Elements of Reusable Object-Oriented Software"

## Reflexión Final

Este ejercicio les permitirá ver de primera mano cómo un patrón de diseño puede transformar código problemático en una solución elegante. Al finalizar, reflexionen sobre:

- ¿Qué problemas específicos resolvió el patrón elegido?
- ¿Sería más fácil ahora añadir un nuevo formato de exportación?
- ¿Cómo mejoró la mantenibilidad del código?
- ¿Qué pasaría si hubieran elegido otro de los patrones?