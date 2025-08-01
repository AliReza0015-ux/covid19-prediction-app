# app.py
import streamlit as st
import pandas as pd
from model_rf_xgb_ridge import load_and_prepare_data, train_models, predict_all
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="COVID-19 Prediction", layout="wide")
st.title("COVID-19 Case Prediction using Ridge, Random Forest & XGBoost")

st.markdown("Upload your dataset or use the default one to see predictions from three models.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    filepath = uploaded_file
    st.success("File uploaded successfully!")
else:
    filepath = "FinalDataSet.csv"
    st.info("Using default dataset: FinalDataSet.csv")

# Load data and train models
X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)
models = train_models(X_train, y_train)
predictions = predict_all(models, X_test)

# Display metrics
st.header("Model Evaluation Metrics")
metrics = {}
for model_name, y_pred in predictions.items():
    metrics[model_name] = {
        "R² Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False)
    }

metrics_df = pd.DataFrame(metrics).T
st.dataframe(metrics_df.style.format("{:.2f}"))

# Visualize predictions
st.header("Actual vs Predicted")
for model_name, y_pred in predictions.items():
    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    }).reset_index(drop=True)

    st.subheader(f"{model_name}")
    st.line_chart(result_df)
