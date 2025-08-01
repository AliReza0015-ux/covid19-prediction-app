
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df = df.drop(columns=['Health_Region_ID', 'Health_Region_name'])
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop(columns=['cases_may13'])
    y = df['cases_may13']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    models = {}

    # Ridge
    ridge_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
    ridge_pipeline.fit(X_train, y_train)
    models['Ridge'] = ridge_pipeline

    # Random Forest
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    models['Random Forest'] = rf_pipeline

    # XGBoost
    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, random_state=42))
    ])
    xgb_pipeline.fit(X_train, y_train)
    models['XGBoost'] = xgb_pipeline

    return models

def predict_all(models, X_test):
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    return predictions
