"""
Decorator para añadir validación de datos al exportador.
"""
from typing import Dict, Any
import logging

from fastapi import Request, HTTPException
from fastapi.responses import FileResponse

from exporters.decorator.exporter_decorator import ExporterDecorator
from exporters.decorator.base_exporter import BaseExporter

logger = logging.getLogger(__name__)

class ValidationDecorator(ExporterDecorator):
    """
    Decorator que añade validación de datos antes de la exportación.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Valida los datos antes de exportarlos.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo generado

        Raises:
            HTTPException: Si los datos no pasan la validación
        """
        # Validar precio
        price = data.get("price")
        if price is None:
            error_msg = "No hay precio para exportar"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        if price <= 0:
            error_msg = f"El precio debe ser positivo, recibido: {price}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Validar modelo
        model = data.get("model")
        if not model or not isinstance(model, str):
            error_msg = "El nombre del modelo es requerido"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Continuar con la exportación si la validación pasa
        logger.info("Datos validados correctamente")
        return super().export(data, request)
