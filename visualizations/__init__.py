"""
Package for visualization components implementing the Factory Pattern.

Este paquete contiene las clases necesarias para implementar el patrón Factory
para la creación de diferentes tipos de visualizaciones de datos.
"""

# Visualization package that implements the Factory Pattern
from .visualization_factory import VisualizationFactory
from .visualization import Visualization
from .scatter_plot import ScatterPlot
from .bar_plot import BarPlot
from .histogram_plot import HistogramPlot

__all__ = [
    'VisualizationFactory',
    'Visualization',
    'ScatterPlot',
    'BarPlot',
    'HistogramPlot'
]