# Implementación de Patrones Tácticos para el Exportador de Predicciones

Este proyecto demuestra la implementación de tres patrones de diseño tácticos (Adapter, Decorator y Facade) para refactorizar un componente de exportación de predicciones en el proyecto Laptop Price Predictor.

## Estructura del Proyecto

```
exporters/
├── __init__.py
├── adapter/                  # Patrón Adapter
│   ├── __init__.py
│   ├── exporter_interface.py   # Interfaz común para todos los exportadores
│   ├── json_exporter.py        # Exportador de JSON
│   ├── csv_exporter.py         # Exportador de CSV
│   ├── pdf_exporter.py         # Exportador de PDF
│   └── exporter_factory.py     # Factory para crear exportadores
├── decorator/                # Patrón Decorator
│   ├── __init__.py
│   ├── base_exporter.py        # Interfaz base para exportadores
│   ├── exporter_decorator.py   # Clase base para los decoradores
│   ├── validation_decorator.py # Decorador para validación
│   ├── logging_decorator.py    # Decorador para logging
│   ├── metadata_decorator.py   # Decorador para metadatos
│   └── concrete_exporters.py   # Implementaciones concretas de exportadores
└── facade/                   # Patrón Facade
    ├── __init__.py
    ├── export_facade.py        # Fachada para el subsistema de exportación
    └── export_utils.py         # Utilidades para exportación

fastapi_integration.py        # Integración con FastAPI
demo.py                       # Script de demostración
```

## Explicación de los Patrones

### Patrón Adapter

El patrón Adapter proporciona una interfaz común para diferentes formatos de exportación.

- **Problema que resuelve**: Incompatibilidad entre las distintas bibliotecas y formatos de exportación.
- **Solución implementada**: Una interfaz común `ExporterInterface` que todos los exportadores implementan.
- **Ventajas**: Facilidad para añadir nuevos formatos sin modificar el código cliente.

### Patrón Decorator

El patrón Decorator permite añadir funcionalidades adicionales a los exportadores de manera dinámica.

- **Problema que resuelve**: Necesidad de añadir comportamientos como validación, logging o metadatos sin modificar las clases base.
- **Solución implementada**: Una jerarquía de decoradores que envuelven los exportadores base.
- **Ventajas**: Las funcionalidades pueden combinarse de manera flexible según las necesidades.

### Patrón Facade

El patrón Facade proporciona una interfaz simplificada al subsistema completo de exportación.

- **Problema que resuelve**: La complejidad de gestionar diferentes patrones, formatos y configuraciones.
- **Solución implementada**: Una clase `ExportFacade` que coordina todo el sistema.
- **Ventajas**: El código cliente solo necesita interactuar con la fachada, sin conocer los detalles internos.

## Cómo Ejecutar

1. Asegúrate de tener instaladas las dependencias:
   - fastapi
   - reportlab
   - python-multipart

2. Para probar la demostración:
   ```
   python demo.py
   ```

3. Para integrar con FastAPI:
   ```python
   from fastapi_integration import app
   import uvicorn

   if __name__ == "__main__":
       uvicorn.run("main:app", reload=True)
   ```

## Conclusiones

- El **Patrón Adapter** es ideal para manejar diferentes formatos de salida con una interfaz común.
- El **Patrón Decorator** es excelente para añadir funcionalidades sin modificar el código existente.
- El **Patrón Facade** simplifica el uso del sistema al ocultar su complejidad interna.

La combinación de estos tres patrones produce un sistema modular, extensible y fácil de usar.
