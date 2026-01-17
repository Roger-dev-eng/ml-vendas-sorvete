from pathlib import Path
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_train():
    return pd.read_csv(PROC_DIR / "train_tratado.csv")


def metrics(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    r2 = r2_score(y, preds)
    return rmse, mae, r2


def train(test_size=0.2, random_state=42):
    df = load_train()
    X = df[["temperatura"]]
    y = df["vendas"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    candidates = {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = XGBRegressor(
            n_estimators=200, random_state=42, verbosity=0
        )
    except Exception:
        pass

    best_model = None
    best_name = None
    best_rmse = np.inf
    results = {}

    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        rmse, mae, r2 = metrics(model, X_val, y_val)
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    joblib.dump(best_model, MODELS_DIR / "melhor_modelo.pkl")

    summary = {
        "best_model": best_name,
        "best_rmse": best_rmse,
        "results": results,
        "test_size": test_size,
        "random_state": random_state,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return True


if __name__ == "__main__":
    train()
    print("train finalizado.")
