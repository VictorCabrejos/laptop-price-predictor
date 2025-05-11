"""
Factory para crear exportadores según el formato solicitado.
"""
from typing import Dict, Type

from exporters.adapter.exporter_interface import ExporterInterface
from exporters.adapter.json_exporter import JsonExporter
from exporters.adapter.csv_exporter import CsvExporter
from exporters.adapter.pdf_exporter import PdfExporter

class ExporterFactory:
    """
    Factory para crear adaptadores de exportación según el formato.
    """

    # Registro de exportadores disponibles
    _exporters: Dict[str, Type[ExporterInterface]] = {
        "json": JsonExporter,
        "csv": CsvExporter,
        "pdf": PdfExporter,
    }

    @classmethod
    def get_exporter(cls, format: str) -> ExporterInterface:
        """
        Obtiene una instancia del exportador para el formato especificado.

        Args:
            format: Formato de exportación (json, csv, pdf)

        Returns:
            Instancia del exportador correspondiente

        Raises:
            ValueError: Si el formato no está soportado
        """
        exporter_class = cls._exporters.get(format.lower())

        if not exporter_class:
            supported_formats = ", ".join(cls._exporters.keys())
            raise ValueError(f"Formato '{format}' no soportado. Formatos disponibles: {supported_formats}")

        return exporter_class()

    @classmethod
    def register_exporter(cls, format: str, exporter_class: Type[ExporterInterface]) -> None:
        """
        Registra un nuevo exportador.

        Args:
            format: Formato de exportación
            exporter_class: Clase del exportador que implementa ExporterInterface
        """
        cls._exporters[format.lower()] = exporter_class
