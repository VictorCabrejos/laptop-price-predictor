"""
Exportador de predicciones en formato PDF.
"""
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from exporters.adapter.exporter_interface import ExporterInterface

class PdfExporter(ExporterInterface):
    """
    Adaptador para exportar predicciones en formato PDF.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """
        Exporta los datos de predicción a formato PDF.

        Args:
            data: Diccionario con los datos de la predicción
            request: Objeto Request de FastAPI

        Returns:
            FileResponse: Respuesta con el archivo PDF generado
        """
        # Asegurar que existe el directorio static
        os.makedirs("static", exist_ok=True)

        # Crear archivo PDF
        file_path = "static/prediction.pdf"
        c = canvas.Canvas(file_path, pagesize=letter)

        # Añadir contenido al PDF
        c.drawString(100, 750, "Laptop Price Prediction")
        c.drawString(100, 730, f"Price: {data.get('price', 0)} {data.get('currency', 'EUR')}")
        c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(100, 690, f"Model: {data.get('model', 'LinearRegression')}")

        # Guardar el PDF
        c.save()

        # Devolver respuesta
        return FileResponse(
            path=file_path,
            filename="laptop_prediction.pdf",
            media_type="application/pdf"
        )
