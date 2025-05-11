"""
Decorator base para los exportadores de predicciones.
"""
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import FileResponse

from exporters.decorator.base_exporter import BaseExporter

class ExporterDecorator(BaseExporter):
    """
    Clase base para todos los decoradores de exportadores.
    """

    def __init__(self, wrapped_exporter: BaseExporter):
        """
        Inicializa el decorador con un exportador.

        Args:
            wrapped_exporter: El exportador al que se le añadirá funcionalidad
        """
        self._wrapped_exporter = wrapped_exporter

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Delega la exportación al exportador decorado.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo generado
        """
        return self._wrapped_exporter.export(data, request)
