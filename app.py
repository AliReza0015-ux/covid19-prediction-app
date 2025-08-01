
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_rf_xgb_ridge import (
    load_and_prepare_data,
    evaluate_model,
    get_rf_model,
    get_xgb_model
)

st.set_page_config(page_title="COVID-19 Case Predictor", layout="wide")
st.title("COVID-19 Case Prediction (Random Forest & XGBoost)")

# Upload or use default
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    filepath = uploaded_file
    st.success("Custom dataset uploaded.")
else:
    filepath = "FinalDataSet.csv"
    st.info("Using default dataset: FinalDataSet.csv")

# Load and prepare
X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

# Model choice
model_choice = st.radio("Choose Model", ["Random Forest", "XGBoost"])
if model_choice == "Random Forest":
    model = get_rf_model()
else:
    model = get_xgb_model()

# Train and evaluate
model.fit(X_train, y_train)
results = evaluate_model(model, X_test, y_test, name=model_choice)

# Metrics
st.subheader(f"{model_choice} Results")
st.metric("R²", f"{results['R² Score']:.3f}")
st.metric("MAE", f"{results['MAE']:.2f}")
st.metric("RMSE", f"{results['RMSE']:.2f}")

# Plot
st.subheader("Predicted vs Actual")
df_plot = pd.DataFrame({
    "Actual": results["Actual"].values,
    "Predicted": results["Predicted"]
}).reset_index(drop=True)

fig, ax = plt.subplots()
ax.plot(df_plot["Actual"], label="Actual", marker='o')
ax.plot(df_plot["Predicted"], label="Predicted", marker='x')
ax.set_xlabel("Index")
ax.set_ylabel("COVID-19 Cases (May 13)")
ax.legend()
st.pyplot(fig)

# Download
csv = df_plot.to_csv(index=False).encode()
st.download_button("Download Prediction CSV", csv, "predictions.csv", "text/csv")
