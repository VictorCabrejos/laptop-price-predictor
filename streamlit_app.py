"""
Laptop Price Predictor - Streamlit Dashboard
===========================================

A simple web application to predict laptop prices based on specifications.
Built for educational purposes and classroom demonstrations.

Author: MLOps Portfolio Project
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 2px solid #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the laptop dataset"""
    try:
        # Load data with multiple path and encoding fallbacks
        import os

        # Try different path approaches
        csv_paths_to_try = [
            "laptop_price.csv",
            "./laptop_price.csv",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "laptop_price.csv"
            ),
            "/c/Users/victor-pc/Desktop/MLOps-Portfolio/active-projects/laptop-price-predictor/laptop_price.csv",
            "C:/Users/victor-pc/Desktop/MLOps-Portfolio/active-projects/laptop-price-predictor/laptop_price.csv",
            "C:\\Users\\victor-pc\\Desktop\\MLOps-Portfolio\\active-projects\\laptop-price-predictor\\laptop_price.csv",
            os.path.join(
                "C:",
                "Users",
                "victor-pc",
                "Desktop",
                "MLOps-Portfolio",
                "active-projects",
                "laptop-price-predictor",
                "laptop_price.csv",
            ),
        ]

        df = None
        for csv_path in csv_paths_to_try:
            try:
                if os.path.exists(csv_path):
                    # Try different encodings for CSV files with special characters
                    encodings_to_try = [
                        "utf-8",
                        "latin-1",
                        "iso-8859-1",
                        "cp1252",
                        "utf-8-sig",
                    ]

                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception:
                            continue

                    if df is not None:
                        break

            except Exception:
                continue

        if df is None:
            raise FileNotFoundError("Could not find or load laptop_price.csv")

        # Clean and preprocess
        df_clean = df.copy()

        # Clean Weight column
        df_clean["Weight"] = df_clean["Weight"].str.replace("kg", "").astype(float)

        # Clean RAM column
        df_clean["Ram"] = df_clean["Ram"].str.replace("GB", "").astype(int)

        # Extract screen resolution
        resolution_pattern = r"(\d{3,4})x(\d{3,4})"
        resolution_extracted = df_clean["ScreenResolution"].str.extract(
            resolution_pattern
        )
        df_clean["screen_width"] = pd.to_numeric(
            resolution_extracted[0], errors="coerce"
        )
        df_clean["screen_height"] = pd.to_numeric(
            resolution_extracted[1], errors="coerce"
        )

        # Extract CPU speed
        cpu_pattern = r"(\d+\.?\d*)GHz"
        cpu_speed = df_clean["Cpu"].str.extract(cpu_pattern)
        df_clean["cpu_speed"] = pd.to_numeric(cpu_speed[0], errors="coerce")

        # Handle missing values
        df_clean = df_clean.dropna(
            subset=["screen_width", "screen_height", "cpu_speed"]
        )

        # Encode categorical variables
        le_company = LabelEncoder()
        le_typename = LabelEncoder()
        le_cpu = LabelEncoder()
        le_gpu = LabelEncoder()
        le_opsys = LabelEncoder()

        df_clean["Company_encoded"] = le_company.fit_transform(df_clean["Company"])
        df_clean["TypeName_encoded"] = le_typename.fit_transform(df_clean["TypeName"])
        df_clean["Cpu_encoded"] = le_cpu.fit_transform(df_clean["Cpu"])
        df_clean["Gpu_encoded"] = le_gpu.fit_transform(df_clean["Gpu"])
        df_clean["OpSys_encoded"] = le_opsys.fit_transform(df_clean["OpSys"])

        # Select features for modeling
        feature_columns = [
            "Inches",
            "Ram",
            "Weight",
            "screen_width",
            "screen_height",
            "cpu_speed",
            "Company_encoded",
            "TypeName_encoded",
            "Cpu_encoded",
            "Gpu_encoded",
            "OpSys_encoded",
        ]

        X = df_clean[feature_columns]
        y = df_clean["Price_euros"]

        return (
            df_clean,
            X,
            y,
            {
                "company": le_company,
                "typename": le_typename,
                "cpu": le_cpu,
                "gpu": le_gpu,
                "opsys": le_opsys,
            },
        )

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None


@st.cache_resource
def train_models(X, y):
    """Train and return both Linear Regression and Random Forest models"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Calculate metrics
    lr_metrics = {
        "mae": mean_absolute_error(y_test, lr_pred),
        "mse": mean_squared_error(y_test, lr_pred),
        "r2": r2_score(y_test, lr_pred),
    }

    rf_metrics = {
        "mae": mean_absolute_error(y_test, rf_pred),
        "mse": mean_squared_error(y_test, rf_pred),
        "r2": r2_score(y_test, rf_pred),
    }

    return lr_model, rf_model, lr_metrics, rf_metrics, X_test, y_test


def main():
    # Main header
    st.markdown(
        '<h1 class="main-header">💻 Laptop Price Predictor Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading and preprocessing dataset..."):
        df, X, y, encoders = load_and_preprocess_data()

    if df is None:
        st.error(
            "Failed to load dataset. Please check if 'laptop_price.csv' exists in the project directory."
        )
        return

    # Train models
    with st.spinner("Training models..."):
        lr_model, rf_model, lr_metrics, rf_metrics, X_test, y_test = train_models(X, y)

    # Sidebar for user inputs
    st.sidebar.header("🔧 Laptop Specifications")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose Prediction Model:",
        ["Random Forest", "Linear Regression"],
        help="Select the machine learning model for price prediction",
    )

    # Input fields
    inches = st.sidebar.slider("Screen Size (inches)", 10.0, 18.0, 15.6, 0.1)
    ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
    weight = st.sidebar.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

    # Company selection
    companies = sorted(df["Company"].unique())
    company = st.sidebar.selectbox("Brand", companies)

    # Type selection
    types = sorted(df["TypeName"].unique())
    typename = st.sidebar.selectbox("Laptop Type", types)

    # CPU selection
    cpus = sorted(df["Cpu"].unique())
    cpu = st.sidebar.selectbox("CPU", cpus[:10])  # Show first 10 for simplicity

    # GPU selection
    gpus = sorted(df["Gpu"].unique())
    gpu = st.sidebar.selectbox("GPU", gpus[:10])  # Show first 10 for simplicity

    # OS selection
    os_options = sorted(df["OpSys"].unique())
    opsys = st.sidebar.selectbox("Operating System", os_options)

    # Screen resolution
    screen_width = st.sidebar.number_input("Screen Width (pixels)", 1000, 4000, 1920)
    screen_height = st.sidebar.number_input("Screen Height (pixels)", 600, 3000, 1080)

    # CPU speed
    cpu_speed = st.sidebar.slider("CPU Speed (GHz)", 1.0, 4.0, 2.5, 0.1)

    # Create prediction button
    predict_button = st.sidebar.button("🔮 Predict Price", type="primary")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Dataset overview
        st.subheader("📊 Dataset Overview")

        # Basic statistics
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

        with col_stats1:
            st.metric("Total Laptops", len(df))

        with col_stats2:
            st.metric("Average Price", f"€{df['Price_euros'].mean():.0f}")

        with col_stats3:
            st.metric(
                "Price Range",
                f"€{df['Price_euros'].min():.0f} - €{df['Price_euros'].max():.0f}",
            )

        with col_stats4:
            st.metric("Brands", df["Company"].nunique())

        # Price distribution
        st.subheader("💰 Price Distribution")
        fig_hist = px.histogram(
            df,
            x="Price_euros",
            nbins=30,
            title="Distribution of Laptop Prices",
            labels={"Price_euros": "Price (€)", "count": "Number of Laptops"},
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Price by brand
        st.subheader("🏷️ Average Price by Brand")
        brand_prices = (
            df.groupby("Company")["Price_euros"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        fig_brand = px.bar(
            x=brand_prices.index,
            y=brand_prices.values,
            title="Top 10 Brands by Average Price",
            labels={"x": "Brand", "y": "Average Price (€)"},
        )
        st.plotly_chart(fig_brand, use_container_width=True)

    with col2:
        # Model performance metrics
        st.subheader("📈 Model Performance")

        # Display metrics for both models
        st.markdown("**Linear Regression**")
        st.markdown(
            f"""
        <div class="metric-card">
            <strong>MAE:</strong> €{lr_metrics['mae']:.2f}<br>
            <strong>R² Score:</strong> {lr_metrics['r2']:.3f}<br>
            <strong>RMSE:</strong> €{np.sqrt(lr_metrics['mse']):.2f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("**Random Forest**")
        st.markdown(
            f"""
        <div class="metric-card">
            <strong>MAE:</strong> €{rf_metrics['mae']:.2f}<br>
            <strong>R² Score:</strong> {rf_metrics['r2']:.3f}<br>
            <strong>RMSE:</strong> €{np.sqrt(rf_metrics['mse']):.2f}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Feature importance for Random Forest
        if selected_model == "Random Forest":
            st.subheader("🔍 Feature Importance")

            feature_names = [
                "Screen Size",
                "RAM",
                "Weight",
                "Screen Width",
                "Screen Height",
                "CPU Speed",
                "Brand",
                "Type",
                "CPU",
                "GPU",
                "OS",
            ]

            importances = rf_model.feature_importances_
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values("Importance", ascending=True)

            fig_importance = px.bar(
                importance_df.tail(8),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Features",
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)

    # Prediction section
    if predict_button:
        st.markdown("---")
        st.subheader("🔮 Price Prediction Result")

        try:
            # Prepare input data
            input_data = pd.DataFrame(
                {
                    "Inches": [inches],
                    "Ram": [ram],
                    "Weight": [weight],
                    "screen_width": [screen_width],
                    "screen_height": [screen_height],
                    "cpu_speed": [cpu_speed],
                    "Company_encoded": [encoders["company"].transform([company])[0]],
                    "TypeName_encoded": [encoders["typename"].transform([typename])[0]],
                    "Cpu_encoded": [encoders["cpu"].transform([cpu])[0]],
                    "Gpu_encoded": [encoders["gpu"].transform([gpu])[0]],
                    "OpSys_encoded": [encoders["opsys"].transform([opsys])[0]],
                }
            )

            # Make prediction
            if selected_model == "Random Forest":
                predicted_price = rf_model.predict(input_data)[0]
                model_name = "Random Forest"
                accuracy = rf_metrics["r2"]
            else:
                predicted_price = lr_model.predict(input_data)[0]
                model_name = "Linear Regression"
                accuracy = lr_metrics["r2"]

            # Display result
            col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])

            with col_pred2:
                st.markdown(
                    f"""
                <div class="prediction-result">
                    <h2>Predicted Price</h2>
                    <h1 style="color: #1f77b4; font-size: 3rem;">€{predicted_price:.2f}</h1>
                    <p><strong>Model:</strong> {model_name}</p>
                    <p><strong>Model Accuracy (R²):</strong> {accuracy:.3f}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Show similar laptops
            st.subheader("🔍 Similar Laptops in Dataset")

            # Find similar laptops based on specs
            similar_condition = (
                (df["Ram"] == ram)
                & (df["Company"] == company)
                & (abs(df["Inches"] - inches) <= 1.0)
            )

            similar_laptops = df[similar_condition][
                ["Company", "Product", "TypeName", "Ram", "Inches", "Price_euros"]
            ].head(5)

            if not similar_laptops.empty:
                st.dataframe(similar_laptops, use_container_width=True)
            else:
                st.info(
                    "No similar laptops found in the dataset with these exact specifications."
                )

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        💻 <strong>Laptop Price Predictor</strong> | Built with Streamlit & Scikit-learn |
        Educational Demo for MLOps Portfolio
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
