"""
Exportador de predicciones en formato JSON.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import FileResponse

from exporters.adapter.exporter_interface import ExporterInterface

class JsonExporter(ExporterInterface):
    """
    Adaptador para exportar predicciones en formato JSON.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Exporta los datos de predicción a formato JSON.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo JSON generado
        """
        # Asegurar que existe el directorio static
        os.makedirs("static", exist_ok=True)

        # Crear contenido JSON
        content = {
            "prediction": {
                "price": data.get("price", 0),
                "currency": data.get("currency", "EUR"),
                "timestamp": datetime.now().isoformat(),
                "model": data.get("model", "LinearRegression")
            }
        }

        # Guardar en archivo
        file_path = "static/prediction.json"
        with open(file_path, "w") as f:
            json.dump(content, f, indent=2)

        # Devolver respuesta
        return FileResponse(
            path=file_path,
            filename="laptop_prediction.json",
            media_type="application/json"
        )
