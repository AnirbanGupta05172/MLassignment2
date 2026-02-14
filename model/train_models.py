"""
Train 6 classification models on ONE dataset (UCI Breast Cancer - via sklearn),
compute required metrics, save all trained models, and export test CSV for Streamlit upload.

Run:
    python -m model.train_models
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "model" / "artifacts"
DATA_DIR = ROOT / "data"

DATASET_META = {
    "name": "Breast Cancer Wisconsin (Diagnostic)",
    "source": "UCI (also available via sklearn.datasets.load_breast_cancer)",
    "n_samples_min_req": 500,
    "n_features_min_req": 12,
    "target_col": "target",
    "positive_label": 1,
    "negative_label": 0,
    "notes": "Binary classification: malignant vs benign (sklearn encodes as 0/1).",
}

def _build_models(random_state: int = 42) -> Dict[str, object]:
    """
    Returns 6 models as required by the assignment.
    """
    models: Dict[str, object] = {}

    # 1) Logistic Regression (scaled)
    models["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]
    )

    # 2) Decision Tree
    models["decision_tree"] = DecisionTreeClassifier(random_state=random_state)

    # 3) KNN (scaled)
    models["knn"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]
    )

    # 4) Naive Bayes (Gaussian)
    models["naive_bayes"] = GaussianNB()

    # 5) Random Forest (Ensemble)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    # 6) XGBoost (Ensemble)
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed. Install it with `pip install xgboost` "
            "or ensure it is present in requirements.txt."
        )

    models["xgboost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )

    return models


MODEL_SPECS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes (Gaussian)",
    "random_forest": "Random Forest (Ensemble)",
    "xgboost": "XGBoost (Ensemble)",
}


def _get_dataset() -> Tuple[pd.DataFrame, pd.Series, list]:
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    feature_names = list(X.columns)
    return X, y, feature_names


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = np.nan
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)

    return {
        "Accuracy": float(acc),
        "AUC": float(auc) if np.isfinite(auc) else np.nan,
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "MCC": float(mcc),
    }


def train_and_save(random_state: int = 42) -> pd.DataFrame:
    """
    Train all models, evaluate on held-out test split, save artifacts, and export test CSV.
    Returns metrics dataframe.
    """
    ART_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = _get_dataset()

    # store in meta for Streamlit
    global DATASET_META
    DATASET_META = {**DATASET_META, "feature_names": feature_names, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}

    if X.shape[0] < DATASET_META["n_samples_min_req"] or X.shape[1] < DATASET_META["n_features_min_req"]:
        raise ValueError("Dataset does not meet min size requirements.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    models = _build_models(random_state=random_state)

    rows = []
    for key, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        metrics = _metric_row(y_test.to_numpy(), y_pred, y_proba)
        row = {"ML Model Name": MODEL_SPECS[key], **metrics}
        rows.append(row)

        # Save model
        joblib.dump(model, ART_DIR / f"{key}.joblib")

    # Save metrics
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(ART_DIR / "metrics_summary.csv", index=False)

    # Export test data with labels for Streamlit upload
    test_df = X_test.copy()
    test_df[DATASET_META["target_col"]] = y_test.values
    test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

    # Save dataset meta for app
    joblib.dump(DATASET_META, ART_DIR / "dataset_meta.joblib")

    # Also generate / update README metrics table
    try:
        from model.make_readme import update_readme_with_metrics
        update_readme_with_metrics(metrics_df)
    except Exception:
        # don't fail training if README update fails
        pass

    return metrics_df


def ensure_trained_artifacts(force: bool = False) -> None:
    """
    Ensure artifacts exist. If missing (or force), train and save.
    """
    has_models = ART_DIR.exists() and any(ART_DIR.glob("*.joblib"))
    has_metrics = (ART_DIR / "metrics_summary.csv").exists()
    if force or (not has_models) or (not has_metrics):
        train_and_save()


if __name__ == "__main__":
    df = train_and_save()
    print("\n=== Metrics Summary (held-out test split) ===")
    print(df.to_string(index=False))
    print(f"\nSaved models to: {ART_DIR}")
    print(f"Exported Streamlit upload file to: {DATA_DIR / 'test_data.csv'}")
