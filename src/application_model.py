import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "credito_aplicacao_clientes_final.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, low_memory=False)
    return df


def build_preprocessor(df, target_col="inadimplente"):
    # Drop obvious id-like columns if present
    drop_cols = [c for c in df.columns if c.lower().startswith("id_") or c.lower() == "id"]
    features = [c for c in df.columns if c not in drop_cols + [target_col]]

    X = df[features].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Simple pipelines
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    return preprocessor, features


def train_and_evaluate(X, y, preprocessor, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    results = {}

    # Baseline: Logistic Regression
    lr_pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))])
    lr_pipe.fit(X_train, y_train)
    y_pred_proba = lr_pipe.predict_proba(X_test)[:, 1]
    y_pred = lr_pipe.predict(X_test)

    results["logistic_regression"] = {
        "model": lr_pipe,
        "metrics": compute_metrics(y_test, y_pred, y_pred_proba),
    }

    # Random Forest
    rf_pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_jobs=-1, random_state=random_state))])
    rf_pipe.fit(X_train, y_train)
    y_pred_proba = rf_pipe.predict_proba(X_test)[:, 1]
    y_pred = rf_pipe.predict(X_test)
    results["random_forest"] = {
        "model": rf_pipe,
        "metrics": compute_metrics(y_test, y_pred, y_pred_proba),
    }

    # XGBoost if available
    try:
        from xgboost import XGBClassifier

        xgb_pipe = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))])
        xgb_pipe.fit(X_train, y_train)
        y_pred_proba = xgb_pipe.predict_proba(X_test)[:, 1]
        y_pred = xgb_pipe.predict(X_test)
        results["xgboost"] = {
            "model": xgb_pipe,
            "metrics": compute_metrics(y_test, y_pred, y_pred_proba),
        }
    except Exception:
        warnings.warn("XGBoost not available; skipping XGBoost training.")

    # Pick best by ROC-AUC
    best_name, best_score = None, -1
    for name, info in results.items():
        auc = info["metrics"].get("roc_auc", 0)
        if auc > best_score:
            best_score = auc
            best_name = name

    # Hyperparameter tuning on best model (only RF or XGBoost for speed)
    tuned = None
    if best_name == "random_forest":
        tuned = tune_random_forest(X_train, y_train, preprocessor, random_state)
    elif best_name == "xgboost":
        tuned = tune_xgboost(X_train, y_train, preprocessor, random_state)

    if tuned is not None:
        y_pred_proba = tuned.predict_proba(X_test)[:, 1]
        y_pred = tuned.predict(X_test)
        results["tuned_" + best_name] = {
            "model": tuned,
            "metrics": compute_metrics(y_test, y_pred, y_pred_proba),
        }
        final_model = tuned
        final_name = "tuned_" + best_name
    else:
        final_model = results[best_name]["model"]
        final_name = best_name

    # Save final model and metrics
    model_path = MODELS_DIR / f"application_best_model_{final_name}.joblib"
    joblib.dump(final_model, model_path)

    metrics_out = {n: r["metrics"] for n, r in results.items()}
    metrics_path = MODELS_DIR / "application_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    return final_name, model_path, metrics_path, results


def compute_metrics(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def tune_random_forest(X, y, preprocessor, random_state=42):
    rf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    pipe = Pipeline([("pre", preprocessor), ("clf", rf)])

    param_dist = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 10, 20, 40],
        "clf__max_features": ["auto", "sqrt"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=6, cv=cv, n_jobs=-1, random_state=random_state)
    search.fit(X, y)
    return search.best_estimator_


def tune_xgboost(X, y, preprocessor, random_state=42):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    pipe = Pipeline([("pre", preprocessor), ("clf", xgb)])

    param_dist = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 6, 10],
        "clf__learning_rate": [0.01, 0.1, 0.2],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=6, cv=cv, n_jobs=-1, random_state=random_state)
    search.fit(X, y)
    return search.best_estimator_


def main():
    df = load_data()

    if "inadimplente" not in df.columns:
        raise RuntimeError("Target column 'inadimplente' not found in dataset")

    preprocessor, features = build_preprocessor(df, target_col="inadimplente")

    X = df[features]
    y = df["inadimplente"]

    final_name, model_path, metrics_path, results = train_and_evaluate(X, y, preprocessor)

    print("Final model:", final_name)
    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
