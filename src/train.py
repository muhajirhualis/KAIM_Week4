

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data():
    df = pd.read_csv("../data/processed/modeling_dataset.csv")
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
    
        
        print(f" {model_name} ({'Tuned' if tuned else 'Baseline'}): AUC = {metrics['roc_auc']:.4f}")
        
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"{model_name} ({'Tuned' if tuned else 'Baseline'}): {metrics_str}")  
              
        # Cross-Validation:
        cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        metrics["cv_auc_mean"] = cv_auc.mean()
        metrics["cv_auc_std"] = cv_auc.std()
        
        return metrics

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit_risk_comparison")

    # Load & split
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f" Loaded {len(X)} samples | High-risk: {y.mean():.1%}")
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

   
    # 1. Baseline Models
    # ==========================
    print("\n---.BASELINE MODELS---")
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    rf_base = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    
    train_and_log(lr_base, "LogisticRegression", X_train, X_test, y_train, y_test, tuned=False)
    train_and_log(rf_base, "RandomForest", X_train, X_test, y_train, y_test, tuned=False)
    
   
    # 2. Tuned Models (Random Search)
    # ==========================
    print("\n---TUNED MODELS (RandomizedSearchCV)---")
    
    # Logistic Regression tuning
    lr_tuned = LogisticRegression(random_state=42, max_iter=1000)
    lr_params = {
        'C': [0.01, 0.1, 1, 10], 
        'penalty': ['l1', 'l2'], 
        'solver': ['liblinear']
        }
    lr_search = RandomizedSearchCV(lr_tuned, lr_params, n_iter=8, cv=3, scoring='roc_auc', random_state=42)
    
    lr_search.fit(X_train, y_train)
    train_and_log(lr_search.best_estimator_, "LogisticRegression", X_train, X_test, y_train, y_test, tuned=True)

    # Random Forest tuning
    rf_tuned = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(rf_tuned, rf_params, n_iter=12, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    train_and_log(rf_search.best_estimator_, "RandomForest", X_train, X_test, y_train, y_test, tuned=True)

    print("\n Done. Run `mlflow ui` to compare baseline vs. tuned.")

if __name__ == "__main__":
    main()