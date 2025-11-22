from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_train():
    return pd.read_csv(PROC_DIR / "train_tratado.csv")

def metrics(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    r2 = r2_score(y, preds)
    return rmse, mae, r2

def train():
    df = load_train()
    X = df[["temperatura"]].values
    y = df["vendas"].values

    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    except:
        pass

    best_model = None
    best_rmse = np.inf

    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        rmse, mae, r2 = metrics(model, X_val, y_val)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    joblib.dump(best_model, MODELS_DIR / "melhor_modelo.pkl")
    return True

if __name__ == "__main__":
    train()
    print("train finalizado.")