
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df = df.fillna(df.median(numeric_only=True))
    df_model = df.drop(columns=["Health_Region_ID", "Health_Region_name"])
    X = df_model.drop(columns=["cases_may13"])
    y = df_model["cases_may13"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    return {
        "Model": name,
        "R²": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "Actual": y_test,
        "Predicted": preds
    }

def get_ridge_model():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("ridge", Ridge(alpha=1.0))
    ])
    return pipeline

def get_rf_model():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return pipeline

def get_xgb_model():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(n_estimators=100, max_depth=4, random_state=42))
    ])
    return pipeline
