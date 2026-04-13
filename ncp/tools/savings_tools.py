import os
import json
import joblib
import pandas as pd
from utils.finance_utils import load_quartiles, EXPENSE_COLS

QUARTILES, SRC = load_quartiles()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "savings_model.joblib")

saved = joblib.load(MODEL_PATH)

pipeline = saved["pipeline"]
feature_cols = saved["feature_cols"]
target_cols = saved["target_cols"]
categorical_features = saved.get("categorical_features", [])


async def forecast_savings(json_path: str, desired_savings_percentage: float):

    # Load JSON
    with open(json_path) as f:
        data = json.load(f)

    X = pd.DataFrame([data])

    # Fill missing columns correctly
    for col in feature_cols:
        if col not in X.columns:
            X[col] = "Unknown" if col in categorical_features else 0

    # Ensure correct order
    X = X[feature_cols]

    # ---- TYPE CASTING ----
    X["Income"] = pd.to_numeric(X["Income"], errors="coerce").fillna(0)
    X["Desired_Savings_Percentage"] = desired_savings_percentage

    # ---- HANDLE EXPENSES ----
    for col in EXPENSE_COLS:
        if col not in X.columns:
            X[col] = 0

    X["Total_Expenses"] = X[EXPENSE_COLS].sum(axis=1)
    X["Disposable_Income"] = X["Income"] - X["Total_Expenses"]
    X["Desired_Savings"] = (X["Income"] * desired_savings_percentage) / 100

    # ---- PREDICTION ----
    preds = pipeline.predict(X)

    preds_df = pd.DataFrame(preds, columns=target_cols)

    return {
        "derived_values": {
            "Total_Expenses": float(X["Total_Expenses"].iloc[0]),
            "Disposable_Income": float(X["Disposable_Income"].iloc[0]),
            "Desired_Savings": float(X["Desired_Savings"].iloc[0])
        },
        "predicted_savings": preds_df.iloc[0].to_dict()
    }