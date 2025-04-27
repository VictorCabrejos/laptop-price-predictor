"""
Clase base abstracta para visualizaciones.
Implementa un diseño común de interfaz para todas las visualizaciones.
"""
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization(ABC):
    """
    Clase base abstracta para todas las visualizaciones.
    Define la interfaz común que todas las visualizaciones concretas deben implementar.
    """

    def __init__(self, title="", figsize=(10, 6), color='blue', **kwargs):
        """
        Inicializa una visualización base

        Args:
            title (str): Título de la visualización
            figsize (tuple): Tamaño de la figura (ancho, alto) en pulgadas
            color (str): Color principal para la visualización
            **kwargs: Argumentos adicionales específicos para cada tipo de visualización
        """
        self.title = title
        self.figsize = figsize
        self.color = color

        # Guardar argumentos adicionales como atributos
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def plot(self, data):
        """
        Método abstracto para crear la visualización

        Args:
            data (dict): Diccionario con los datos necesarios para la visualización

        Returns:
            matplotlib.pyplot: Objeto plt con el gráfico creado
        """
        pass

    def save(self, filepath):
        """
        Guarda la figura actual en un archivo

        Args:
            filepath (str): Ruta donde guardar la visualización
        """
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()