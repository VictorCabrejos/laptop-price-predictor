"""
HistogramPlot visualization class that extends the base Visualization class.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .base_visualization import Visualization


class HistogramPlot(Visualization):
    """
    Concrete implementation of Visualization for generating histogram plots.
    """

    def plot(self, data):
        """
        Create a histogram plot

        Args:
            data (dict): Dictionary with plot data containing:
                - values: Series/list of values to plot
                - bins: Number of bins for histogram
                - kde: Whether to show density curve

        Returns:
            matplotlib.pyplot: The created plot
        """
        # Setup the figure
        plt.figure(figsize=self.figsize)

        # Create the histogram with seaborn
        sns.histplot(
            x=data['values'],
            bins=data.get('bins', 20),
            kde=data.get('kde', True),
            color=self.color,
            edgecolor='black',
            alpha=0.7
        )

        # Calculate statistics for annotations
        mean_val = data['values'].mean()
        median_val = data['values'].median()

        # Add vertical lines for mean and median
        plt.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
        plt.axvline(median_val, color='#3498db', linestyle='-', linewidth=2, label=f'Mediana: {median_val:.2f}')

        # Add title and labels
        plt.title(self.title, fontsize=16)
        plt.xlabel(getattr(self, 'xlabel', ''), fontsize=12)
        plt.ylabel(getattr(self, 'ylabel', 'Frecuencia'), fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Improve layout
        plt.tight_layout()

        return plt