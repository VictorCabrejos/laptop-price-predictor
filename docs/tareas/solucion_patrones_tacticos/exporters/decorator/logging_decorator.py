"""
Decorator para añadir logging al exportador.
"""
import logging
import time
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import FileResponse

from exporters.decorator.exporter_decorator import ExporterDecorator
from exporters.decorator.base_exporter import BaseExporter

# Configuración del logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LoggingDecorator(ExporterDecorator):
    """
    Decorator que añade logging antes y después de la exportación.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Registra información antes y después de la exportación.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo generado
        """
        start_time = time.time()

        # Log antes de la exportación
        client = request.client.host if request.client else "unknown"
        logger.info(f"Iniciando exportación para cliente {client}")
        logger.info(f"Datos a exportar: precio={data.get('price')}, modelo={data.get('model')}")

        try:
            # Realizar la exportación
            response = super().export(data, request)

            # Log después de la exportación exitosa
            elapsed_time = time.time() - start_time
            logger.info(f"Exportación completada en {elapsed_time:.3f} segundos")

            return response

        except Exception as e:
            # Log en caso de error
            elapsed_time = time.time() - start_time
            logger.error(f"Error durante la exportación: {str(e)}")
            logger.error(f"La exportación falló después de {elapsed_time:.3f} segundos")
            raise
