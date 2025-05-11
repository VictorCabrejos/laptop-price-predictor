"""
Base interface for exporters with the decorator pattern.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from fastapi import Request
from fastapi.responses import FileResponse

class BaseExporter(ABC):
    """
    Interfaz base para todos los exportadores y decoradores.
    """

    @abstractmethod
    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Exporta los datos de predicción.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo generado
        """
        pass
