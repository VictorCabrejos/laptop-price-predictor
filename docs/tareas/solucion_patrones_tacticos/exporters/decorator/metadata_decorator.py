"""
Decorator para añadir metadatos al exportador.
"""
from typing import Dict, Any
from datetime import datetime
import platform
import socket

from fastapi import Request
from fastapi.responses import FileResponse

from exporters.decorator.exporter_decorator import ExporterDecorator
from exporters.decorator.base_exporter import BaseExporter

class MetadataDecorator(ExporterDecorator):
    """
    Decorator que añade metadatos adicionales a los datos de exportación.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Añade metadatos adicionales a los datos antes de exportarlos.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo generado
        """
        # Crear una copia de los datos para no modificar el original
        enriched_data = data.copy()

        # Añadir metadatos adicionales
        enriched_data.update({
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "ip_address": request.client.host if request.client else "unknown"
        })

        # Continuar con la exportación usando los datos enriquecidos
        return super().export(enriched_data, request)
