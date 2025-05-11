"""
Implementación concreta de los exportadores base para el patrón decorator.
"""
import json
import csv
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import Request
from fastapi.responses import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from exporters.decorator.base_exporter import BaseExporter

class JsonExporterBase(BaseExporter):
    """
    Implementación básica del exportador JSON para usar con decoradores.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """Exporta los datos a formato JSON."""
        # Asegurar que existe el directorio static
        os.makedirs("static", exist_ok=True)

        # Crear contenido JSON
        content = {
            "prediction": {
                "price": data.get("price", 0),
                "currency": data.get("currency", "EUR"),
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "model": data.get("model", "LinearRegression")
            }
        }

        # Añadir metadatos si existen
        if "user_agent" in data:
            content["metadata"] = {
                key: data[key] for key in ["user_agent", "ip_address", "hostname", "platform"]
                if key in data
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

class CsvExporterBase(BaseExporter):
    """
    Implementación básica del exportador CSV para usar con decoradores.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """Exporta los datos a formato CSV."""
        # Asegurar que existe el directorio static
        os.makedirs("static", exist_ok=True)

        # Preparar datos para CSV
        headers = ["price", "currency", "timestamp", "model"]
        row = [
            str(data.get("price", 0)),
            data.get("currency", "EUR"),
            data.get("timestamp", datetime.now().isoformat()),
            data.get("model", "LinearRegression")
        ]

        # Añadir metadatos si existen
        if "user_agent" in data:
            headers.extend(["user_agent", "ip_address", "hostname", "platform"])
            row.extend([
                data.get("user_agent", ""),
                data.get("ip_address", ""),
                data.get("hostname", ""),
                data.get("platform", "")
            ])

        # Guardar en archivo
        file_path = "static/prediction.csv"
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

        # Devolver respuesta
        return FileResponse(
            path=file_path,
            filename="laptop_prediction.csv",
            media_type="text/csv"
        )

class PdfExporterBase(BaseExporter):
    """
    Implementación básica del exportador PDF para usar con decoradores.
    """

    def export(self, data: Dict[str, Any], request: Request) -> FileResponse:
        """Exporta los datos a formato PDF."""
        # Asegurar que existe el directorio static
        os.makedirs("static", exist_ok=True)

        # Crear archivo PDF
        file_path = "static/prediction.pdf"
        c = canvas.Canvas(file_path, pagesize=letter)

        # Añadir contenido al PDF
        c.drawString(100, 750, "Laptop Price Prediction")
        c.drawString(100, 730, f"Price: {data.get('price', 0)} {data.get('currency', 'EUR')}")
        c.drawString(100, 710, f"Date: {data.get('timestamp', datetime.now().isoformat())}")
        c.drawString(100, 690, f"Model: {data.get('model', 'LinearRegression')}")

        # Añadir metadatos si existen
        y_pos = 670
        if "user_agent" in data:
            c.drawString(100, y_pos, "--- Metadata ---")
            y_pos -= 20
            c.drawString(100, y_pos, f"User Agent: {data.get('user_agent', '')}")
            y_pos -= 20
            c.drawString(100, y_pos, f"IP Address: {data.get('ip_address', '')}")
            y_pos -= 20
            c.drawString(100, y_pos, f"Hostname: {data.get('hostname', '')}")
            y_pos -= 20
            c.drawString(100, y_pos, f"Platform: {data.get('platform', '')}")

        # Guardar el PDF
        c.save()

        # Devolver respuesta
        return FileResponse(
            path=file_path,
            filename="laptop_prediction.pdf",
            media_type="application/pdf"
        )
