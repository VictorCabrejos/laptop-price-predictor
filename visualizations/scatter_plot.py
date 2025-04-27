"""
ScatterPlot visualization class that extends the base Visualization class.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .base_visualization import Visualization


class ScatterPlot(Visualization):
    """
    Concrete implementation of Visualization for generating scatter plots.
    """

    def plot(self, data):
        """
        Create a scatter plot

        Args:
            data (dict): Dictionary with plot data containing:
                - x: Values for x-axis
                - y: Values for y-axis

        Returns:
            matplotlib.pyplot: The created plot
        """
        # Setup the figure
        plt.figure(figsize=self.figsize)

        # Create the scatter plot with regression line
        sns.regplot(
            x=data['x'],
            y=data['y'],
            scatter_kws={'alpha': 0.6, 'color': self.color},
            line_kws={'color': getattr(self, 'line_color', '#e74c3c')}
        )

        # Add title and labels
        plt.title(self.title, fontsize=16)
        plt.xlabel(getattr(self, 'xlabel', ''), fontsize=12)
        plt.ylabel(getattr(self, 'ylabel', ''), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Calculate correlation
        correlation = np.corrcoef(data['x'], data['y'])[0, 1]

        # Add annotation for correlation
        plt.annotate(
            f'Correlación: {correlation:.4f}',
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=12
        )

        # Improve layout
        plt.tight_layout()

        return plt