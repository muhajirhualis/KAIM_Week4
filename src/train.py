
import pathlib  # Add this import at the top
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
import joblib
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# SETUP LOGGING
# ============================================================
# This helps us see what's happening when the script runs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    # Get directory of THIS script (src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root, then to data/
    data_path = os.path.join(script_dir, "..", "data",
                             "processed", "modeling_dataset.csv")

    # Normalize path (handles ../ correctly)
    data_path = os.path.normpath(data_path)

    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}\n")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["CustomerId", "is_high_risk"], errors="ignore")
    y = df["is_high_risk"]
    return X, y


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_model(model, filepath="models/credit_model.pkl"):
    """
    Save the trained model to a file.
    The API (src/api/main.py) will load this file to make predictions.
    """
    # Ensure models/ directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        joblib.dump(model, filepath)
        logger.info(f" Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f" Failed to save model: {e}")
        raise


def train_and_log(model, model_name, X_train, X_test, y_train, y_test, tuned=False):

    with mlflow.start_run(run_name=f"{model_name}_{'tuned' if tuned else 'baseline'}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)

        # Log
        mlflow.log_params(model.get_params() if not tuned else {"tuned": True})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(
            f" {model_name} ({'Tuned' if tuned else 'Baseline'}): AUC = {metrics['roc_auc']:.4f}")

        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"{model_name} ({'Tuned' if tuned else 'Baseline'}): {metrics_str}")

        # Cross-Validation:
        cv_auc = cross_val_score(
            model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        metrics["cv_auc_mean"] = cv_auc.mean()
        metrics["cv_auc_std"] = cv_auc.std()

        if hasattr(model, 'coef_'):
            coef_df = pd.DataFrame({
                'feature': X_train.columns,
                'coef': model.coef_[0]
            }).sort_values('coef', key=abs, ascending=False)
            print("\nTop 5 Features (by |coefficient|):")
            print(coef_df.head())

        return metrics


def main():
    # 1. Get the Absolute Path correctly
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    # 2. Define the paths
    mlruns_dir = os.path.join(PROJECT_ROOT, "mlruns")
    db_path = os.path.join(PROJECT_ROOT, "mlflow.db")

    # 3. CONVERT TO WINDOWS-FRIENDLY URI
    # This converts "D:\path\to\mlruns" to "file:///D:/path/to/mlruns"
    # which MLflow requires on Windows.
    mlruns_uri = pathlib.Path(mlruns_dir).as_uri()
    db_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"

    # 4. Set Tracking URI
    mlflow.set_tracking_uri(db_uri)

    # 5. Handle Experiment Creation with the fixed URI
    experiment_name = "credit_risk_final"
    try:
        # Pass the URI, not the raw string path
        mlflow.create_experiment(experiment_name, artifact_location=mlruns_uri)
    except:
        mlflow.set_experiment(experiment_name)

    # Load & split
    X, y = load_data()

    # Identify non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print("\n Non-numeric columns found:", non_numeric)
        print("Dropping non-numeric columns for modeling.")
        X = X.select_dtypes(include=[np.number])  # Keep only numeric

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # After train_test_split, ensure X_train is DataFrame
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    print(f" Loaded {len(X)} samples | High-risk: {y.mean():.1%}")
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    # Store results for comparison
    results = {}

    # 1. Baseline Models
    # ==========================
    print("\n--- BASELINE MODELS ---")
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    rf_base = RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=5)

    lr_metrics = train_and_log(
        lr_base, "LogisticRegression", X_train, X_test, y_train, y_test, tuned=False)
    rf_metrics = train_and_log(
        rf_base, "RandomForest", X_train, X_test, y_train, y_test, tuned=False)

    results["lr_baseline"] = {"model": lr_base, "auc": lr_metrics["roc_auc"]}
    results["rf_baseline"] = {"model": rf_base, "auc": rf_metrics["roc_auc"]}

    # 2. Tuned Models (Random Search)
    # ==========================
    print("\n--- TUNED MODELS (RandomizedSearchCV) ---")

    # Logistic Regression
    lr_tuned = LogisticRegression(random_state=42, max_iter=1000)
    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        # safer default (l1 + liblinear = deprecated combo)
        'penalty': ['l2'],
        # lbfgs supports l2, better for most cases
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    }
    lr_search = RandomizedSearchCV(
        lr_tuned, lr_params, n_iter=8, cv=3, scoring='roc_auc', random_state=42)
    lr_search.fit(X_train, y_train)
    lr_tuned_metrics = train_and_log(
        lr_search.best_estimator_, "LogisticRegression", X_train, X_test, y_train, y_test, tuned=True)
    results["lr_tuned"] = {
        "model": lr_search.best_estimator_, "auc": lr_tuned_metrics["roc_auc"]}

    # Random Forest
    rf_tuned = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [
        5, 10, 15], 'min_samples_split': [2, 5, 10]}
    rf_search = RandomizedSearchCV(
        rf_tuned, rf_params, n_iter=12, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf_tuned_metrics = train_and_log(
        rf_search.best_estimator_, "RandomForest", X_train, X_test, y_train, y_test, tuned=True)
    results["rf_tuned"] = {
        "model": rf_search.best_estimator_, "auc": rf_tuned_metrics["roc_auc"]}

    # 3. Identify & Save BEST Model
    # ==========================
    best_name = max(results.keys(), key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    best_auc = results[best_name]["auc"]

    print(f"\n BEST MODEL: {best_name} (AUC = {best_auc:.4f})")

    # Save best model
    best_model_path = os.path.join(
        PROJECT_ROOT, "models", "credit_risk_best.pkl")
    save_model(best_model, best_model_path)
    print(" Best model saved to: models/credit_risk_best.pkl")

    # Log best model to MLflow Registry
    try:
        with mlflow.start_run(run_name="Best_Model"):
            mlflow.log_params({"model_type": best_name, "auc": best_auc})
            signature = mlflow.models.infer_signature(
                X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                registered_model_name="CreditRisk_Best"
            )
        print("Best model also logged to MLflow Registry.")
    except Exception as e:
        print(f"MLflow registration skipped: {e}")

    print("\n Done. Run `mlflow ui` to compare all runs.")


if __name__ == "__main__":
    main()
