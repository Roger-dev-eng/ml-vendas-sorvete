from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

def load_model():
    return joblib.load(MODELS_DIR / "melhor_modelo.pkl")

def load_scaler():
    return joblib.load(PROC_DIR / "scaler.joblib")

def predict_single(input_dict, model=None):
    if model is None:
        model = load_model()
    scaler = load_scaler()
    df = pd.DataFrame([[float(input_dict["temperatura"])]], columns=["temperatura"])
    Xs = scaler.transform(df)
    pred = model.predict(Xs)
    print("Predição única executada com sucesso.")
    return float(pred[0])
    


def predict_batch(path, model=None):
    if model is None:
        model = load_model()
    scaler = load_scaler()
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    if "temperatura" not in df.columns:
        raise ValueError("CSV precisa de coluna temperatura")
    Xs = scaler.transform(df[["temperatura"]].astype(float))
    df["prediction"] = model.predict(Xs)
    print("Predição em lote executada com sucesso.")
    return df
    

