"""
Exportador de predicciones en formato CSV.
"""
import csv
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import FileResponse

from exporters.adapter.exporter_interface import ExporterInterface

class CsvExporter(ExporterInterface):
    """
    Adaptador para exportar predicciones en formato CSV.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Exporta los datos de predicción a formato CSV.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo CSV generado
        """
        # Asegurar que existe el directorio static
        os.makedirs("static", exist_ok=True)

        # Crear datos para CSV
        csv_data = [
            ["price", "currency", "timestamp", "model"],
            [
                str(data.get("price", 0)),
                data.get("currency", "EUR"),
                datetime.now().isoformat(),
                data.get("model", "LinearRegression")
            ]
        ]

        # Guardar en archivo
        file_path = "static/prediction.csv"
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        # Devolver respuesta
        return FileResponse(
            path=file_path,
            filename="laptop_prediction.csv",
            media_type="text/csv"
        )
