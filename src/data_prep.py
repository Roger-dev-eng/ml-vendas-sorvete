from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_raw():
    return pd.read_csv(RAW_DIR / "ice_cream_sales.csv")

def prepare():
    df = load_raw()
    df = df.rename(columns=lambda c: c.strip())
    temp_col = "Temperatura (°C)"
    target_col = "Sorvetes Vendidos"
    df = df[[temp_col, target_col]].dropna()
    df = df.rename(columns={temp_col: "temperatura", target_col: "vendas"})
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce")
    df["vendas"] = pd.to_numeric(df["vendas"], errors="coerce")
    df = df.dropna()

    X = df[["temperatura"]]
    y = df["vendas"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_s, columns=["temperatura"])
    train_df["vendas"] = y_train.values

    test_df = pd.DataFrame(X_test_s, columns=["temperatura"])
    test_df["vendas"] = y_test.values

    train_df.to_csv(PROC_DIR / "train_tratado.csv", index=False)
    test_df.to_csv(PROC_DIR / "test_tratado.csv", index=False)
    joblib.dump(scaler, PROC_DIR / "scaler.joblib")
    print("Processo de preparação concluído com sucesso.")

if __name__ == "__main__":
    prepare()
    print("data_prep finalizado.")
