from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def load_model():
    return joblib.load(MODELS_DIR / "melhor_modelo.pkl")


def predict_single(input_dict, model=None):
    if model is None:
        model = load_model()
    df = pd.DataFrame([[float(input_dict["temperatura"])]], columns=["temperatura"])
    pred = model.predict(df)
    print("Predição única executada com sucesso.")
    return float(pred[0])


def predict_batch(path, model=None):
    if model is None:
        model = load_model()
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    if "temperatura" not in df.columns:
        raise ValueError("CSV precisa de coluna temperatura")
    df["prediction"] = model.predict(df[["temperatura"]].astype(float))
    print("Predição em lote executada com sucesso.")
    return df
