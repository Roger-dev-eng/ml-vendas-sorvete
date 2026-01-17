from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def _extract_estimator(model, X):
    if hasattr(model, "steps"):
        try:
            pre = model[:-1]
            estimator = model[-1]
            if hasattr(pre, "transform"):
                X_transformed = pre.transform(X)
            else:
                X_transformed = X
            return estimator, X_transformed
        except Exception:
            return model, X
    return model, X


def shap_analysis(model, X, max_display=10, sample_size=200):
    try:
        import shap
    except Exception as e:
        print("SHAP indisponível:", e)
        return

    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)
    estimator, X_used = _extract_estimator(model, X_sample)

    model_name = estimator.__class__.__name__.lower()
    is_tree = (
        hasattr(estimator, "feature_importances_")
        or "xgb" in model_name
        or "forest" in model_name
    )
    is_linear = "linear" in model_name

    if is_tree:
        explainer = shap.TreeExplainer(estimator)
    elif is_linear:
        explainer = shap.LinearExplainer(estimator, X_used)
    else:
        print("SHAP ignorado: modelo sem suporte barato.")
        return

    shap_values = explainer.shap_values(X_used)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_used, show=False, max_display=max_display)
    out = OUTPUT_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"SHAP salvo em: {out}")


def run_evaluation(model_file="melhor_modelo.pkl", test_file="test_tratado.csv", enable_shap=False):
    df_test = load_test(test_file)
    target = df_test.columns[-1]
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    model = load_model(model_file)
    preds = model.predict(X_test)
    metrics = print_metrics(y_test, preds)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted")
    scatter_path = OUTPUT_DIR / "true_vs_pred.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Scatter salvo em: {scatter_path}")

    if enable_shap:
        shap_analysis(model, X_test)

    return metrics


if __name__ == "__main__":
    run_evaluation()
