"""
Base abstract class for all visualizations used in the Factory Pattern implementation.
"""
from abc import ABC, abstractmethod
import pandas as pd
import os


class Visualization(ABC):
    """
    Abstract base class for all visualization types.
    Implements the base functionality and defines the interface for concrete visualizations.
    """

    def __init__(self, data: pd.DataFrame, output_path: str):
        """
        Initialize the visualization with data and output path.

        Args:
            data (pd.DataFrame): The dataset to visualize
            output_path (str): Path where the visualization will be saved
        """
        self.data = data
        self.output_path = output_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    @abstractmethod
    def generate(self) -> str:
        """
        Generate the visualization and save it to the output path.
        This is an abstract method that must be implemented by concrete classes.

        Returns:
            str: Path to the generated visualization file
        """
        pass

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess data before visualization.
        This method can be overridden by concrete classes if needed.

        Returns:
            pd.DataFrame: Preprocessed data
        """
        return self.data.copy()