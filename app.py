
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_rf_xgb_ridge import (
    load_and_prepare_data,
    evaluate_model,
    get_rf_model,
    get_xgb_model,
    get_ridge_model
)

st.set_page_config(page_title="COVID-19 Prediction App", layout="wide")
st.title("COVID-19 Case Prediction App (Ridge PCA, Random Forest, XGBoost)")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    filepath = uploaded_file
    st.success("Custom dataset uploaded.")
else:
    filepath = "FinalDataSet.csv"
    st.info("Using default dataset: FinalDataSet.csv")

# Load and split
X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)

# Model selection
model_choice = st.radio("Select a model to run", ["Ridge PCA", "Random Forest", "XGBoost"])

if model_choice == "Ridge PCA":
    model = get_ridge_model()
elif model_choice == "Random Forest":
    model = get_rf_model()
else:
    model = get_xgb_model()

# Train and predict
model.fit(X_train, y_train)
results = evaluate_model(model, X_test, y_test, name=model_choice)

# Show metrics
st.subheader(f"{model_choice} Results")
st.metric("R² Score", f"{results['R² Score']:.3f}")
st.metric("MAE", f"{results['MAE']:.2f}")
st.metric("RMSE", f"{results['RMSE']:.2f}")

# Plot actual vs predicted
st.subheader("Predicted vs Actual")
df_plot = pd.DataFrame({
    "Actual": results["Actual"].values,
    "Predicted": results["Predicted"]
}).reset_index(drop=True)

fig, ax = plt.subplots()
ax.plot(df_plot["Actual"], label="Actual", linestyle='--', marker='o')
ax.plot(df_plot["Predicted"], label="Predicted", linestyle='-', marker='x')
ax.set_title("Predicted vs Actual COVID-19 Cases")
ax.set_xlabel("Index")
ax.set_ylabel("Cases (May 13)")
ax.legend()
st.pyplot(fig)

# Export CSV
csv_data = df_plot.to_csv(index=False).encode()
st.download_button("Download Predictions", csv_data, "predictions.csv", "text/csv")
