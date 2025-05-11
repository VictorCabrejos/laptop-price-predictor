"""
Fachada para el sistema de exportación de predicciones.
"""
import logging
from typing import Dict, Any, Optional

from fastapi import Request, HTTPException
from fastapi.responses import FileResponse

# Importaciones para el patrón Adapter
from exporters.adapter.exporter_factory import ExporterFactory

# Importaciones para el patrón Decorator
from exporters.decorator.base_exporter import BaseExporter
from exporters.decorator.concrete_exporters import JsonExporterBase, CsvExporterBase, PdfExporterBase
from exporters.decorator.validation_decorator import ValidationDecorator
from exporters.decorator.logging_decorator import LoggingDecorator
from exporters.decorator.metadata_decorator import MetadataDecorator

logger = logging.getLogger(__name__)

class ExportFacade:
    """
    Fachada que proporciona una interfaz simplificada para exportar predicciones.
    """

    @staticmethod
    def export_prediction(
        format: str,
        price: float,
        request: Request,
        model: str = "LinearRegression",
        currency: str = "EUR",
        use_adapter_pattern: bool = True,
        add_validation: bool = True,
        add_logging: bool = True,
        add_metadata: bool = True
    ) -> FileResponse:
        """
        Exporta una predicción al formato especificado.

        Args:
            format: Formato de exportación (json, csv, pdf)
            price: Precio predicho
            request: Objeto Request de FastAPI
            model: Nombre del modelo utilizado
            currency: Moneda del precio
            use_adapter_pattern: Si se debe usar el patrón Adapter en lugar del Decorator
            add_validation: Si se debe añadir validación (solo para Decorator)
            add_logging: Si se debe añadir logging (solo para Decorator)
            add_metadata: Si se debe añadir metadatos (solo para Decorator)

        Returns:
            FileResponse con el archivo generado

        Raises:
            HTTPException: Si el formato no es soportado o hay un error
        """
        try:
            # Datos básicos de la predicción
            data = {
                "price": price,
                "model": model,
                "currency": currency
            }

            # Usar el patrón Adapter o Decorator según la configuración
            if use_adapter_pattern:
                # Implementación usando el patrón Adapter
                exporter = ExporterFactory.get_exporter(format)
                return exporter.export(data, request)
            else:
                # Implementación usando el patrón Decorator
                # Seleccionar el exportador base según el formato
                if format.lower() == "json":
                    exporter: BaseExporter = JsonExporterBase()
                elif format.lower() == "csv":
                    exporter = CsvExporterBase()
                elif format.lower() == "pdf":
                    exporter = PdfExporterBase()
                else:
                    raise ValueError(f"Formato '{format}' no soportado")

                # Aplicar decoradores según la configuración
                if add_validation:
                    exporter = ValidationDecorator(exporter)

                if add_logging:
                    exporter = LoggingDecorator(exporter)

                if add_metadata:
                    exporter = MetadataDecorator(exporter)

                # Exportar usando la cadena de decoradores
                return exporter.export(data, request)

        except ValueError as e:
            # Capturar errores de formato no soportado
            logger.error(f"Error de formato: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            # Capturar otros errores
            logger.error(f"Error durante la exportación: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error durante la exportación: {str(e)}")

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        """
        Devuelve los formatos de exportación soportados con sus descripciones.

        Returns:
            Diccionario con formatos y descripciones
        """
        return {
            "json": "JavaScript Object Notation",
            "csv": "Comma-Separated Values",
            "pdf": "Portable Document Format"
        }
