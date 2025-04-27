# Laptop Price Predictor

A sophisticated machine learning application designed to predict laptop prices based on specifications. This project demonstrates the transition from a prototype data science notebook to a well-structured, maintainable web application using software design patterns.

![Laptop Price Predictor](https://img.shields.io/badge/Project-ML%20Application-blue)
![Design Patterns](https://img.shields.io/badge/Patterns-Factory%20%7C%20Prototype%20%7C%20Singleton-success)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

## 📋 Project Description

The Laptop Price Predictor allows users to:
- Enter laptop specifications (weight, RAM, type, screen resolution, etc.)
- Get an accurate price prediction in euros
- Switch between multiple ML models (Linear Regression, Random Forest)
- Apply different model configurations
- View detailed model metrics and feature importance
- Explore data visualizations with interactive features
- Compare model performance

This application demonstrates how proper software design can transform a data science prototype into a robust, scalable application with enhanced functionality and maintainability.

## 🚀 Project Evolution

### Stage 1: Design Pattern Implementation ✅
- **Factory Pattern**: Implemented for visualizations and model creation, enabling easy extension with new visualization types and ML algorithms
- **Prototype Pattern**: Applied for model configurations, allowing users to select and customize predefined configurations
- **Singleton Pattern**: Used for configuration and model managers to ensure consistent access to resources
- **Enhanced UI**: Interactive dashboard with model comparison and configurable visualizations
- [Read the Technical Architecture Overview](docs/architecture/technical-overview.md) for in-depth details

### Stage 2: Model Expansion & Advanced Features 🔲
- Additional ML models (Gradient Boosting, SVR)
- Automated hyperparameter optimization
- Feature engineering options
- More advanced visualizations

### Stage 3: Enterprise Features 🔲
- Authentication system
- AI-generated analysis reports
- RESTful API endpoints
- Persistent storage with database integration
- Deployment configurations for cloud platforms

## 💡 Business Value

### For Companies
- Accurately estimate laptop prices for procurement decisions
- Understand which specifications most impact pricing
- Optimize laptop configurations for cost-effectiveness
- Make data-driven decisions based on market trends

### For Educational Purposes
- Learn software design patterns in a real-world ML context
- Understand the transition from data science prototype to production application
- Explore feature importance and model comparison techniques
- Study visualization best practices

## 🛠️ Technology Stack

- **Backend**: Python 3.11, FastAPI, scikit-learn
- **Frontend**: HTML, TailwindCSS, Chart.js
- **Machine Learning**: Linear Regression, Random Forest Regression
- **Visualization**: Matplotlib, Seaborn
- **Design Patterns**: Factory, Prototype, Singleton

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.9+ installed
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/laptop-price-predictor.git
   cd laptop-price-predictor
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

## 📊 Features

### Price Prediction
Enter laptop specifications through an intuitive form and receive an instant price estimate based on the selected model.

### Model Selection and Configuration
- Switch between Linear Regression and Random Forest models
- Choose from predefined configurations:
  - Basic Linear Regression
  - Optimized Random Forest
  - Reduced Feature Set

### Model Information
- Detailed performance metrics (MAE, R²)
- Feature importance visualization
- Coefficient interpretation for Linear Regression
- Model comparison functionality

### Data Visualizations
- RAM vs Price scatter plots with regression line
- Price distribution histograms
- Gaming vs Non-Gaming laptop price comparison
- Interactive visualization with lightbox feature for detailed viewing

## 📝 Project Structure

```
laptop-price-predictor/
├── docs/                            # Documentation
│   ├── architecture/                # Technical architecture documents
│   └── design_patterns/             # Design pattern tutorials (in Spanish)
├── models/                          # Model implementation with design patterns
│   ├── prediction_model.py          # Abstract model interface
│   ├── linear_regression_model.py   # Linear regression implementation
│   ├── random_forest_model.py       # Random forest implementation
│   ├── model_factory.py             # Factory pattern implementation
│   ├── model_config.py              # Configuration objects (Prototype)
│   └── model_config_registry.py     # Registry for configurations
├── visualizations/                  # Visualization components with Factory pattern
│   ├── base_visualization.py        # Abstract visualization class
│   ├── scatter_plot.py              # Scatter plot implementation
│   ├── bar_plot.py                  # Bar plot implementation
│   ├── histogram_plot.py            # Histogram plot implementation
│   ├── visualization_factory.py     # Factory for creating visualizations
│   └── visualization_manager.py     # Manager for generating visualizations
├── preprocessors/                   # Data preprocessing components
│   └── data_preprocessor.py         # Preprocessor for input data
├── static/                          # Static files and generated visualizations
├── templates/                       # HTML templates for web interface
│   ├── index.html                   # Home page with prediction form
│   ├── model_info.html              # Model details page
│   └── visualizations.html          # Visualizations dashboard
├── main.py                          # Application entry point
├── laptop_price.csv                 # Dataset of laptop prices
└── requirements.txt                 # Project dependencies
```

## 🔍 Dataset

The `laptop_price.csv` dataset contains information about various laptops including:
- Brand and product name
- Type (Gaming, Notebook, Ultrabook, etc.)
- Screen resolution and size
- CPU specifications
- RAM and storage capacity
- GPU information
- Weight
- Price in euros

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to your branch: `git push origin feature-name`
5. Open a pull request

## 📚 Learning Resources

For students learning about design patterns:
- Design pattern tutorials (in Spanish) are available in the `docs/design_patterns/` directory
- Check out the implementation of each pattern in the codebase
- Read the [Technical Architecture Overview](docs/architecture/technical-overview.md) for design decisions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Developed for Software Design Course - URP 2025