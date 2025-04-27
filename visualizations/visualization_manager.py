"""
Clase para gestionar la generación y almacenamiento de visualizaciones.
Utiliza el patrón Factory para crear diferentes tipos de visualizaciones.
"""
import os
import pandas as pd
import numpy as np
from .visualization_factory import VisualizationFactory

class VisualizationManager:
    """
    Clase para gestionar la generación y almacenamiento de visualizaciones
    utilizando el patrón Factory para crear diferentes tipos de gráficos.
    """

    def __init__(self, data_path="laptop_price.csv", output_dir="static/visualizations"):
        """
        Inicializa el gestor de visualizaciones

        Args:
            data_path (str): Ruta al archivo CSV con los datos
            output_dir (str): Directorio donde se guardarán las visualizaciones
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.factory = VisualizationFactory()
        self.df = None

        # Asegurar que el directorio de salida existe
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """
        Carga los datos desde el archivo CSV y realiza preprocesamiento básico
        """
        self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1')

        # Preprocesamiento básico
        self.df['Weight'] = self.df['Weight'].str.replace('kg', '').astype(float)
        self.df['Ram'] = self.df['Ram'].str.replace('GB', '').astype(int)
        self.df = pd.get_dummies(self.df, columns=['TypeName'], dtype='int')

        # Extraer resolución
        self.df[['screen_width', 'screen_height']] = self.df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(int)

        # Extraer GHz del procesador
        self.df['GHz'] = self.df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

        return self.df

    def generate_all_visualizations(self):
        """
        Genera todas las visualizaciones disponibles
        """
        if self.df is None:
            self.load_data()

        # Lista de visualizaciones a generar
        visualizations = [
            self.generate_price_distribution,
            self.generate_feature_correlation,
            self.generate_ram_distribution,
            self.generate_ram_vs_price,
            self.generate_price_by_type
        ]

        # Generar cada visualización
        for viz_func in visualizations:
            viz_func()

        return True

    def generate_price_distribution(self):
        """
        Genera un histograma de la distribución de precios
        """
        histogram = self.factory.create_visualization(
            'histogram',
            title='Distribución de Precios de Laptops',
            xlabel='Precio (Euros)',
            color='skyblue',
            bins=30,
            kde=True
        )

        data = {
            'values': self.df['Price_euros'],
            'bins': 30,
            'kde': True
        }

        histogram.plot(data)
        histogram.save(os.path.join(self.output_dir, 'price_distribution.png'))

    def generate_feature_correlation(self):
        """
        Genera un gráfico de barras con la correlación entre características y precio
        """
        # Calcular correlaciones con el precio
        numeric_cols = ['Weight', 'Ram', 'screen_width', 'screen_height', 'GHz',
                        'TypeName_Gaming', 'TypeName_Notebook']
        correlations = self.df[numeric_cols].corrwith(self.df['Price_euros']).abs().sort_values(ascending=False)

        bar_plot = self.factory.create_visualization(
            'bar',
            title='Correlación de Características con el Precio',
            xlabel='Correlación',
            ylabel='Característica',
            color='cornflowerblue',
            horizontal=True
        )

        data = {
            'x': correlations.values,
            'y': correlations.index,
            'sort': True
        }

        bar_plot.plot(data)
        bar_plot.save(os.path.join(self.output_dir, 'feature_correlation.png'))

    def generate_ram_distribution(self):
        """
        Genera un histograma de la distribución de RAM
        """
        histogram = self.factory.create_visualization(
            'histogram',
            title='Distribución de RAM en Laptops',
            xlabel='RAM (GB)',
            color='lightgreen',
            bins=15,
            kde=False
        )

        data = {
            'values': self.df['Ram'],
            'bins': 15,
            'kde': False
        }

        histogram.plot(data)
        histogram.save(os.path.join(self.output_dir, 'ram_distribution.png'))

    def generate_ram_vs_price(self):
        """
        Genera un gráfico de dispersión entre RAM y precio
        """
        scatter = self.factory.create_visualization(
            'scatter',
            title='Relación entre RAM y Precio',
            xlabel='RAM (GB)',
            ylabel='Precio (Euros)'
        )

        data = {
            'x': self.df['Ram'],
            'y': self.df['Price_euros']
        }

        scatter.plot(data)
        scatter.save(os.path.join(self.output_dir, 'ram_vs_price.png'))

    def generate_price_by_type(self):
        """
        Genera un gráfico de barras comparando precios por tipo de laptop
        """
        # Calcular precio medio por tipo
        price_by_type = []
        labels = []

        if 'TypeName_Gaming' in self.df.columns:
            gaming_price = self.df[self.df['TypeName_Gaming'] == 1]['Price_euros'].mean()
            price_by_type.append(gaming_price)
            labels.append('Gaming')

        if 'TypeName_Notebook' in self.df.columns:
            notebook_price = self.df[self.df['TypeName_Notebook'] == 1]['Price_euros'].mean()
            price_by_type.append(notebook_price)
            labels.append('Notebook')

        bar_plot = self.factory.create_visualization(
            'bar',
            title='Precio Promedio por Tipo de Laptop',
            xlabel='Tipo de Laptop',
            ylabel='Precio Promedio (Euros)',
            color='salmon',
            horizontal=False
        )

        data = {
            'x': price_by_type,
            'y': labels
        }

        bar_plot.plot(data)
        bar_plot.save(os.path.join(self.output_dir, 'price_by_type.png'))