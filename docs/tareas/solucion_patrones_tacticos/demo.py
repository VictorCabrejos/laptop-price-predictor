"""
Script principal que demuestra el uso de los diferentes patrones de diseño.
"""
import os
import json
import time
from datetime import datetime
from fastapi import Request

# Importación de implementaciones de cada patrón
# Patrón Adapter
from exporters.adapter.exporter_factory import ExporterFactory

# Patrón Decorator
from exporters.decorator.concrete_exporters import JsonExporterBase, CsvExporterBase, PdfExporterBase
from exporters.decorator.validation_decorator import ValidationDecorator
from exporters.decorator.logging_decorator import LoggingDecorator
from exporters.decorator.metadata_decorator import MetadataDecorator

# Patrón Facade
from exporters.facade.export_facade import ExportFacade
from exporters.facade.export_utils import ExportUtils

# Creación de un objeto Request falso para pruebas
class MockRequest:
    class Client:
        host = "127.0.0.1"

    def __init__(self):
        self.client = self.Client()
        self.headers = {"user-agent": "Python Mock Request"}

# Datos de ejemplo para la exportación
prediction_data = {
    "price": 1299.99,
    "currency": "EUR",
    "model": "RandomForest"
}

def demo_adapter_pattern():
    """Demostración del patrón Adapter."""
    print("\n\n===== DEMOSTRACIÓN DEL PATRÓN ADAPTER =====")
    print("El patrón Adapter permite adaptar diferentes formatos de salida a una interfaz común.")

    # Crear directorio para pruebas
    os.makedirs("static", exist_ok=True)

    # Crear una petición simulada
    request = MockRequest()

    # Demostrar el uso del Adapter para diferentes formatos
    formats = ["json", "csv", "pdf"]

    for format in formats:
        print(f"\nExportando a formato: {format}")
        try:
            exporter = ExporterFactory.get_exporter(format)
            response = exporter.export(prediction_data, request)
            print(f"✓ Exportación exitosa: {response.path}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")

    # Intentar con un formato no compatible
    print("\nIntentando exportar a formato no soportado (xml):")
    try:
        exporter = ExporterFactory.get_exporter("xml")
        response = exporter.export(prediction_data, request)
        print(f"Exportación exitosa: {response.path}")
    except Exception as e:
        print(f"✗ Error esperado: {str(e)}")

    print("\nExtensión del patrón Adapter:")
    print("Para añadir un nuevo formato, solo necesitas:")
    print("1. Crear una nueva clase que implemente ExporterInterface")
    print("2. Registrarla en la ExporterFactory")
    print("Sin modificar ningún otro código existente.")

def demo_decorator_pattern():
    """Demostración del patrón Decorator."""
    print("\n\n===== DEMOSTRACIÓN DEL PATRÓN DECORATOR =====")
    print("El patrón Decorator permite añadir funcionalidades de manera dinámica sin modificar las clases base.")

    # Crear directorio para pruebas
    os.makedirs("static", exist_ok=True)

    # Crear una petición simulada
    request = MockRequest()

    print("\n1) Exportación básica sin decoradores:")
    exporter = JsonExporterBase()
    response = exporter.export(prediction_data, request)
    print(f"✓ Exportación exitosa: {response.path}")

    print("\n2) Añadiendo validación con decorador:")
    validated_exporter = ValidationDecorator(JsonExporterBase())

    # Datos válidos
    print("- Probando con datos válidos:")
    response = validated_exporter.export(prediction_data, request)
    print(f"✓ Exportación exitosa: {response.path}")

    # Datos inválidos
    print("- Probando con datos inválidos (precio negativo):")
    invalid_data = prediction_data.copy()
    invalid_data["price"] = -100
    try:
        validated_exporter.export(invalid_data, request)
        print("Exportación exitosa (no debería ocurrir)")
    except Exception as e:
        print(f"✓ Error esperado: {str(e)}")

    print("\n3) Añadiendo logging y metadatos:")
    decorated_exporter = LoggingDecorator(
        MetadataDecorator(
            ValidationDecorator(
                JsonExporterBase()
            )
        )
    )

    print("- Exportando con la cadena completa de decoradores:")
    response = decorated_exporter.export(prediction_data, request)
    print(f"✓ Exportación exitosa: {response.path}")

    print("\nVentajas del patrón Decorator:")
    print("- Puedes añadir o quitar funcionalidades sin cambiar las clases base")
    print("- Las funcionalidades están desacopladas")
    print("- Puedes combinarlos en cualquier orden según tus necesidades")

def demo_facade_pattern():
    """Demostración del patrón Facade."""
    print("\n\n===== DEMOSTRACIÓN DEL PATRÓN FACADE =====")
    print("El patrón Facade proporciona una interfaz simplificada a un subsistema complejo.")

    # Crear directorio para pruebas
    os.makedirs("static", exist_ok=True)

    # Crear una petición simulada
    request = MockRequest()

    print("\n1) Exportación usando la fachada con Adapter:")
    response = ExportFacade.export_prediction(
        format="json",
        price=1299.99,
        request=request,
        model="RandomForest",
        use_adapter_pattern=True
    )
    print(f"✓ Exportación exitosa: {response.path}")

    print("\n2) Exportación usando la fachada con Decorator:")
    response = ExportFacade.export_prediction(
        format="pdf",
        price=1299.99,
        request=request,
        model="RandomForest",
        use_adapter_pattern=False,
        add_validation=True,
        add_logging=True,
        add_metadata=True
    )
    print(f"✓ Exportación exitosa: {response.path}")

    print("\n3) Usando utilidades a través de la fachada:")
    formats = ExportFacade.get_supported_formats()
    print(f"Formatos soportados: {', '.join(formats.keys())}")

    # Limpiar archivos viejos
    ExportUtils.clean_old_exports()

    print("\nVentajas del patrón Facade:")
    print("- Interfaz simplificada para el cliente")
    print("- Oculta la complejidad de los subsistemas")
    print("- Reduce el acoplamiento entre el cliente y los subsistemas")
    print("- Centraliza la lógica relacionada")

if __name__ == "__main__":
    # Mostrar encabezado
    print("========================================================")
    print("  DEMOSTRACIÓN DE PATRONES DE DISEÑO PARA EXPORTACIÓN")
    print("========================================================")
    print("Este script demuestra los tres patrones de diseño")
    print("implementados para el sistema de exportación:")
    print("- Adapter: Para manejar diferentes formatos")
    print("- Decorator: Para añadir funcionalidades dinámicamente")
    print("- Facade: Para simplificar el uso del sistema completo")

    # Ejecutar demostraciones
    demo_adapter_pattern()
    demo_decorator_pattern()
    demo_facade_pattern()

    # Mostrar conclusión
    print("\n\n========================================================")
    print("                   CONCLUSIÓN")
    print("========================================================")
    print("Cada patrón resuelve un problema específico:")
    print("- ADAPTER: Compatibilidad entre interfaces incompatibles")
    print("- DECORATOR: Adición dinámica de comportamientos")
    print("- FACADE: Simplificación de sistemas complejos")
    print("\nLa combinación de los tres patrones crea un sistema")
    print("extensible, modular y fácil de usar.")
    print("========================================================")
