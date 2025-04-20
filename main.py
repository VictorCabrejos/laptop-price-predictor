import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json
import uvicorn
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables that hold state
model = None
features = ['Weight', 'TypeName_Gaming', 'TypeName_Notebook', 'screen_width', 'screen_height', 'GHz', 'Ram']
feature_importances = {}
mae = 0
r2 = 0

# Load and prepare data for the model
def load_data():
    global model, feature_importances, mae, r2

    # Load the dataset
    df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')

    # Clean and transform data
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df = pd.get_dummies(df, columns=['TypeName'], dtype='int')

    # Extract screen resolution
    df[['screen_width', 'screen_height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(float)

    # Extract CPU speed
    df['GHz'] = df['Cpu'].str.split().str[-1].str.replace('GHz', '').astype(float)

    # Feature selection
    X = df[features]
    y = df['Price_euros']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Get feature importances
    for i, feature in enumerate(features):
        feature_importances[feature] = float(model.coef_[i])

    # Generate a plot for the model
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Laptop Prices')
    plt.savefig('static/prediction_plot.png')

    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    features_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.coef_
    })
    features_df = features_df.sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')

    print(f"Model trained with MAE: {mae:.2f} and R²: {r2:.2f}")

# Find similar laptops based on price
def find_similar_laptops(price, count=3):
    df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')
    df['price_diff'] = abs(df['Price_euros'] - price)
    similar = df.sort_values('price_diff').head(count)
    result = []
    for _, row in similar.iterrows():
        result.append({
            'company': row['Company'],
            'product': row['Product'],
            'price': f"{row['Price_euros']:,.2f}"
        })
    return result

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # If model is not loaded, load it now
    if model is None:
        load_data()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "mae": f"{mae:.2f}",
        "r2": f"{r2:.4f}"
    })

@app.get("/model_info", response_class=HTMLResponse)
async def model_info(request: Request):
    if model is None:
        load_data()

    # Prepare feature importance data for the template
    feature_data = []
    for feature, importance in feature_importances.items():
        feature_data.append({
            "feature": feature,
            "importance": f"{importance:.4f}"
        })

    return templates.TemplateResponse("model_info.html", {
        "request": request,
        "mae": f"{mae:.2f}",
        "r2": f"{r2:.4f}",
        "feature_importances": feature_data
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
    if model is None:
        load_data()

    # Prepare input for prediction
    input_data = np.array([[weight, is_gaming, is_notebook, screen_width, screen_height, ghz, ram]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Find similar laptops
    similar_laptops = find_similar_laptops(prediction)

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
    load_data()
    return {"message": "Model retrained successfully", "mae": mae, "r2": r2}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)