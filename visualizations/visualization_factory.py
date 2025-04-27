"""
Factory class for creating visualization objects.
Implements the Factory Pattern for visualization generation.
"""
from typing import Any, Dict

class VisualizationFactory:
    """
    Factory class for creating different types of visualizations.
    Implements the Factory Pattern to encapsulate the creation logic.
    """

    @staticmethod
    def create_visualization(viz_type: str, **kwargs) -> Any:
        """
        Create and return a visualization object of the specified type.

        Args:
            viz_type (str): Type of visualization to create ('scatter', 'bar', 'histogram', etc.)
            **kwargs: Additional parameters to pass to the visualization constructor

        Returns:
            Visualization: An instance of a concrete Visualization class

        Raises:
            ValueError: If the requested visualization type is not supported
        """
        # Import here to avoid circular imports
        from .scatter_plot import ScatterPlot
        from .bar_plot import BarPlot
        from .histogram_plot import HistogramPlot

        if viz_type.lower() == "scatter":
            return ScatterPlot(**kwargs)
        elif viz_type.lower() == "bar":
            return BarPlot(**kwargs)
        elif viz_type.lower() == "histogram":
            return HistogramPlot(**kwargs)
        else:
            raise ValueError(f"Visualization type '{viz_type}' is not supported")