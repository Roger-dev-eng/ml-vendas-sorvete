from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_test(test_file="test_tratado.csv"):
    path = PROC_DIR / test_file
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {path}")
    return pd.read_csv(path)

def load_model(model_file="melhor_modelo.pkl"):
    path = MODELS_DIR / model_file
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    return joblib.load(path)

def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # RMSE manual
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}

def shap_analysis(model, X, max_display=10):
    try:
        import shap
        explainer = None
        if hasattr(model, "feature_importances_") or model.__class__.__name__.lower().find("xgb") >= 0 or model.__class__.__name__.lower().find("forest") >= 0:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            background = X.sample(n=min(50, len(X)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X.sample(n=min(200, len(X))))
            X = X.sample(n=min(200, len(X)))
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X, show=False, max_display=max_display)
        out = OUTPUT_DIR / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"SHAP salvo em: {out}")
    except Exception as e:
        print("SHAP não pôde ser executado:", e)

def run_evaluation(model_file="melhor_modelo.pkl", test_file="test_tratado.csv"):
    df_test = load_test(test_file)
    target = df_test.columns[-1]
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    model = load_model(model_file)
    preds = model.predict(X_test)
    metrics = print_metrics(y_test, preds)

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted")
    scatter_path = OUTPUT_DIR / "true_vs_pred.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Scatter salvo em: {scatter_path}")

    shap_analysis(model, X_test)

    return metrics

if __name__ == "__main__":
    run_evaluation()
