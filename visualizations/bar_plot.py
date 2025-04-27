"""
BarPlot visualization class that extends the base Visualization class.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .base_visualization import Visualization


class BarPlot(Visualization):
    """
    Concrete implementation of Visualization for generating bar plots.
    """

    def plot(self, data):
        """
        Create a bar plot

        Args:
            data (dict): Dictionary with plot data containing:
                - x: Values for bars (heights)
                - y: Labels for bars
                - sort: Whether to sort the data

        Returns:
            matplotlib.pyplot: The created plot
        """
        # Setup the figure
        plt.figure(figsize=self.figsize)

        # Determine orientation
        horizontal = getattr(self, 'horizontal', False)

        # Data preparation
        x = data.get('x', [])
        y = data.get('y', [])

        # Sort if requested
        if data.get('sort', False):
            # Create sorted indices
            sort_idx = np.argsort(x)
            if not horizontal:  # For vertical bars, sort in descending order
                sort_idx = sort_idx[::-1]

            # Apply sorting
            x = np.array(x)[sort_idx]
            y = np.array(y)[sort_idx]

        # Create the bar plot with proper orientation
        if horizontal:
            ax = sns.barplot(x=x, y=y, color=self.color, orient='h')
        else:
            ax = sns.barplot(x=y, y=x, color=self.color)

        # Add title and labels
        plt.title(self.title, fontsize=16)
        plt.xlabel(getattr(self, 'xlabel', ''), fontsize=12)
        plt.ylabel(getattr(self, 'ylabel', ''), fontsize=12)

        # Add value labels on the bars
        for i, p in enumerate(ax.patches):
            if horizontal:
                width = p.get_width()
                ax.annotate(f'{width:.2f}',
                           (width, p.get_y() + p.get_height()/2),
                           ha='left', va='center',
                           fontsize=10, fontweight='bold',
                           xytext=(5, 0), textcoords='offset points')
            else:
                height = p.get_height()
                ax.annotate(f'{height:.2f}',
                           (p.get_x() + p.get_width()/2, height),
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           xytext=(0, 5), textcoords='offset points')

        # Improve layout
        plt.tight_layout()

        return plt