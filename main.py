import numpy as np
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Importaciones de nuestros módulos refactorizados con patrones de diseño
from models.config_manager import ConfigManager
from models.model_manager import ModelManager
from models.model_factory import ModelFactory
from models.model_config import ModelConfig
from models.model_config_registry import ModelConfigRegistry

# Importar nuestro nuevo módulo de visualizaciones con patrón Factory
from visualizations.visualization_manager import VisualizationManager

# Initialize FastAPI app
app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instancias singleton de nuestros managers
config_manager = ConfigManager()
model_manager = ModelManager()

# Instancia del registro de configuraciones (para el patrón Prototype)
config_registry = ModelConfigRegistry()

# Instancia del gestor de visualizaciones (que usa el patrón Factory)
visualization_manager = VisualizationManager(data_path="laptop_price.csv", output_dir="static/visualizations")

# Variable para almacenar las comparaciones de modelos
model_comparison_data = {
    'linear': {
        'name': 'Regresión Lineal',
        'mae': None,
        'r2': None,
        'features': None,
        'active': False
    },
    'random_forest': {
        'name': 'Random Forest',
        'mae': None,
        'r2': None,
        'features': None,
        'active': False
    }
}

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Get model metrics - ModelManager se encarga de entrenar el modelo si es necesario
    metrics = model_manager.get_model_metrics()

    # Get current model type
    current_model_type = config_manager.get('model_type')
    model_name = model_comparison_data[current_model_type]['name'] if current_model_type in model_comparison_data else "Regresión Lineal"

    # Update active model in comparison data
    update_active_model(current_model_type)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "mae": f"{metrics['mae']:.2f}",
        "r2": f"{metrics['r2']:.4f}",
        "model_name": model_name
    })

@app.get("/model_info", response_class=HTMLResponse)
async def model_info(request: Request):
    # Get feature importances - ModelManager se encarga de entrenar el modelo si es necesario
    feature_importances = model_manager.get_feature_importances()

    # Get current model type
    current_model_type = config_manager.get('model_type')
    model_name = model_comparison_data[current_model_type]['name'] if current_model_type in model_comparison_data else "Regresión Lineal"
    features_count = len(config_manager.get('features'))

    # Prepare feature importance data for the template
    feature_data = []
    for feature, importance in feature_importances.items():
        feature_data.append({
            "feature": feature,
            "importance": f"{importance:.4f}"
        })

    metrics = model_manager.get_model_metrics()

    return templates.TemplateResponse("model_info.html", {
        "request": request,
        "mae": f"{metrics['mae']:.2f}",
        "r2": f"{metrics['r2']:.4f}",
        "feature_importances": feature_data,
        "model_name": model_name,
        "features_count": features_count
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    weight: float = Form(...),
    is_gaming: int = Form(...),
    is_notebook: int = Form(...),
    screen_width: int = Form(...),
    screen_height: int = Form(...),
    ghz: float = Form(...),
    ram: int = Form(...)
):
    # Prepare input for prediction
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])

    # Make prediction using ModelManager (que gestiona el modelo Singleton)
    prediction = model_manager.predict(input_data)

    # Find similar laptops
    similar_laptops = model_manager.find_similar_laptops(prediction)

    # Return prediction page
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "price": f"{prediction:,.2f}",
        "weight": weight,
        "ram": ram,
        "is_gaming": "Sí" if is_gaming == 1 else "No",
        "is_notebook": "Sí" if is_notebook == 1 else "No",
        "screen_resolution": f"{screen_width}x{screen_height}",
        "ghz": ghz,
        "similar_laptops": similar_laptops
    })

@app.get("/retrain")
async def retrain():
    # Force retraining of the model
    model_manager.train_model()
    metrics = model_manager.get_model_metrics()

    # Update comparison data
    update_model_metrics(config_manager.get('model_type'), metrics, config_manager.get('features'))

    return {"message": "Model retrained successfully", "mae": metrics['mae'], "r2": metrics['r2']}

@app.get("/change_model/{model_type}")
async def change_model(model_type: str):
    """
    Cambia el tipo de modelo utilizado en el sistema.
    Utiliza el patrón Factory para crear una nueva instancia del modelo.
    """
    # Actualizar el tipo de modelo en la configuración
    config_manager.set('model_type', model_type)

    # Forzar reentrenamiento con el nuevo tipo de modelo
    model_manager.train_model()

    metrics = model_manager.get_model_metrics()

    # Update comparison data
    update_model_metrics(model_type, metrics, config_manager.get('features'))
    update_active_model(model_type)

    return {
        "message": f"Model changed to {model_type} successfully",
        "mae": metrics['mae'],
        "r2": metrics['r2']
    }

@app.get("/model_configs")
async def list_model_configs():
    """
    Lista todas las configuraciones de modelo disponibles en el registro.
    Demuestra el uso del patrón Prototype.
    """
    return config_registry.list_prototypes()

@app.get("/use_model_config/{config_name}")
async def use_model_config(config_name: str):
    """
    Usa una configuración de modelo predefinida del registro.
    Demuestra el uso del patrón Prototype.
    """
    # Obtiene un clon de la configuración requerida
    config = config_registry.get(config_name)
    if not config:
        return {"error": f"Configuration {config_name} not found"}

    # Actualiza la configuración del sistema
    config_manager.set('model_type', config.model_type)
    config_manager.set('features', config.features)

    # Guarda la configuración y fuerza reentrenamiento
    config_manager.save_config()
    model_manager.train_model()

    metrics = model_manager.get_model_metrics()

    # Update comparison data
    update_model_metrics(config.model_type, metrics, config.features)
    update_active_model(config.model_type)

    return {
        "message": f"Model configuration changed to {config_name}",
        "model_type": config.model_type,
        "features": len(config.features),
        "mae": metrics['mae'],
        "r2": metrics['r2']
    }

@app.get("/model_comparison")
async def model_comparison():
    """
    Devuelve datos de comparación de diferentes modelos.
    """
    # Ensure we have data for the current active model
    current_model_type = config_manager.get('model_type')
    if current_model_type not in model_comparison_data or not model_comparison_data[current_model_type]['mae']:
        metrics = model_manager.get_model_metrics()
        features = config_manager.get('features')
        update_model_metrics(current_model_type, metrics, features)

    # Mark current model as active
    update_active_model(current_model_type)

    return {
        "models": list(model_comparison_data.values())
    }

@app.get("/visualizations", response_class=HTMLResponse)
async def visualizations_dashboard(request: Request):
    """
    Muestra el dashboard de visualizaciones creadas con el patrón Factory.
    """
    # Generar las visualizaciones usando nuestro gestor (que utiliza el Factory Pattern)
    visualization_manager.generate_all_visualizations()

    # Renderizar el template con las visualizaciones
    return templates.TemplateResponse("visualizations.html", {
        "request": request
    })

def update_model_metrics(model_type, metrics, features):
    """
    Actualiza los datos de comparación para un modelo específico.
    """
    if model_type in model_comparison_data:
        model_comparison_data[model_type]['mae'] = f"{metrics['mae']:.2f}"
        model_comparison_data[model_type]['r2'] = f"{metrics['r2']:.4f}"
        model_comparison_data[model_type]['features'] = len(features)

def update_active_model(active_model_type):
    """
    Marca el modelo activo en los datos de comparación.
    """
    for model_type in model_comparison_data:
        model_comparison_data[model_type]['active'] = (model_type == active_model_type)

# Run the application
if __name__ == "__main__":
    # Ensure model is trained before serving
    model_manager.train_model()
    # Initialize comparison data with current model
    metrics = model_manager.get_model_metrics()
    update_model_metrics(config_manager.get('model_type'), metrics, config_manager.get('features'))
    update_active_model(config_manager.get('model_type'))

    # Generate visualizations on startup
    visualization_manager.generate_all_visualizations()

    uvicorn.run(app, host="0.0.0.0", port=8000)