"""
Módulo de utilidades para el subsistema de exportación de predicciones.
"""
import os
import shutil
import glob
from typing import Dict, Any, List, Tuple
from datetime import datetime

class ExportUtils:
    """
    Clase utilitaria para operaciones relacionadas con exportación.
    """

    @staticmethod
    def ensure_export_directory(directory: str = "static") -> str:
        """
        Asegura que el directorio de exportación existe.

        Args:
            directory: Ruta del directorio a crear

        Returns:
            La ruta del directorio
        """
        os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def clean_old_exports(
        directory: str = "static",
        patterns: List[str] = ["prediction.*"],
        max_age_hours: int = 24
    ) -> int:
        """
        Elimina archivos de exportación antiguos.

        Args:
            directory: Directorio donde buscar
            patterns: Patrones de nombre de archivo
            max_age_hours: Edad máxima en horas

        Returns:
            Número de archivos eliminados
        """
        if not os.path.exists(directory):
            return 0

        now = datetime.now()
        deleted_count = 0

        for pattern in patterns:
            for file_path in glob.glob(os.path.join(directory, pattern)):
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_hours = (now - file_time).total_seconds() / 3600

                    if age_hours >= max_age_hours:
                        os.remove(file_path)
                        deleted_count += 1

        return deleted_count

    @staticmethod
    def generate_filename(
        base_name: str,
        format: str,
        timestamp: bool = True,
        random_suffix: bool = False
    ) -> str:
        """
        Genera un nombre de archivo para la exportación.

        Args:
            base_name: Nombre base del archivo
            format: Formato/extensión del archivo
            timestamp: Si se debe incluir marca de tiempo
            random_suffix: Si se debe incluir sufijo aleatorio

        Returns:
            Nombre de archivo generado
        """
        filename = base_name

        # Añadir timestamp si se solicita
        if timestamp:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{time_str}"

        # Añadir sufijo aleatorio si se solicita
        if random_suffix:
            import random
            suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
            filename = f"{filename}_{suffix}"

        # Añadir extensión
        if not format.startswith('.'):
            format = f".{format}"
        filename = f"{filename}{format}"

        return filename
