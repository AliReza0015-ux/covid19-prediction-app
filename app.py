
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_rf_xgb_ridge import (
    load_and_prepare_data,
    evaluate_model,
    get_ridge_model,
    get_rf_model,
    get_xgb_model
)

st.set_page_config(page_title="COVID-19 Prediction App", layout="wide")
st.title("COVID-19 Case Prediction (Ridge, RF, XGBoost)")

# Upload or load default dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    filepath = uploaded_file
    st.success("Custom dataset uploaded.")
else:
    filepath = "FinalDataSet.csv"
    st.info("Using default dataset: FinalDataSet.csv")

# Load and split data
X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

# Model selection
model_type = st.selectbox("Choose a model to run:", ["Ridge Regression", "Random Forest", "XGBoost"])

if model_type == "Ridge Regression":
    model = get_ridge_model()
elif model_type == "Random Forest":
    model = get_rf_model()
else:
    model = get_xgb_model()

# Train model
model.fit(X_train, y_train)

# Evaluate
results = evaluate_model(model, X_test, y_test, name=model_type)
st.subheader(f"{model_type} Results")
st.write(f"**R² Score**: {results['R²']:.3f}")
st.write(f"**MAE**: {results['MAE']:.2f}")
st.write(f"**RMSE**: {results['RMSE']:.2f}")

# Plot predictions vs actual
st.subheader("Predicted vs Actual Cases (May 2020)")
plot_df = pd.DataFrame({
    "Actual": results["Actual"].values,
    "Predicted": results["Predicted"]
}).reset_index(drop=True)

fig, ax = plt.subplots()
ax.plot(plot_df["Actual"], label="Actual", marker='o')
ax.plot(plot_df["Predicted"], label="Predicted", marker='x')
ax.set_xlabel("Test Sample")
ax.set_ylabel("Cases")
ax.legend()
st.pyplot(fig)

# Export for Power BI
csv_download = plot_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Prediction Results (CSV)", csv_download, "predictions.csv", "text/csv")

