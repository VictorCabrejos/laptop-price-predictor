"""
Interfaz para los exportadores de predicciones.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from fastapi import Request
from fastapi.responses import FileResponse

class ExporterInterface(ABC):
    """
    Interfaz común para todos los exportadores de predicciones.
    """

    @abstractmethod
    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Exporta los datos de predicción al formato específico.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo generado
        """
        pass
