import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import joblib

from model.train_models import ensure_trained_artifacts, DATASET_META, MODEL_SPECS

ART_DIR = Path(__file__).parent / "model" / "artifacts"

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

st.title("Machine Learning Assignment 2 — Classification Model Demo")
st.caption("Upload test CSV → choose a model → get predictions + evaluation (if labels provided).")

# Sidebar controls
st.sidebar.header("Controls")
retrain = st.sidebar.button("Re-train / Refresh models (uses built-in dataset)")
selected_model = st.sidebar.selectbox("Select model", list(MODEL_SPECS.keys()))

# Make sure artifacts exist BEFORE we try to read feature_names
if retrain:
    with st.spinner("Training models and saving artifacts..."):
        ensure_trained_artifacts(force=True)
    st.success("Models re-trained and artifacts refreshed.")

if not ART_DIR.exists() or not any(ART_DIR.glob("*.joblib")):
    with st.spinner("First run: training models and saving artifacts..."):
        ensure_trained_artifacts(force=False)

# Load dataset metadata saved during training (contains feature_names)
meta_path = ART_DIR / "dataset_meta.joblib"
if meta_path.exists():
    try:
        DATASET_META = joblib.load(meta_path)
    except Exception:
        pass

feature_names = DATASET_META.get("feature_names", [])

with st.expander("Dataset & expected columns", expanded=False):
    st.write("This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset (UCI).")
    st.write("Expected feature columns:")
    if feature_names:
        st.code(", ".join(feature_names))
    else:
        st.warning("Feature names not available yet. Click 'Re-train / Refresh models' in the sidebar.")
    st.write("Optional label column:")
    st.code(DATASET_META.get("target_col", "target"))


# Make sure artifacts exist
if retrain:
    with st.spinner("Training models and saving artifacts..."):
        ensure_trained_artifacts(force=True)
    st.success("Models re-trained and artifacts refreshed.")

if not ART_DIR.exists() or not any(ART_DIR.glob("*.joblib")):
    with st.spinner("First run: training models and saving artifacts..."):
        ensure_trained_artifacts(force=False)

# Load metrics summary (computed during training)
metrics_path = ART_DIR / "metrics_summary.csv"
if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)
else:
    metrics_df = pd.DataFrame()

# Layout
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader("1) Upload test data (CSV)")
    uploaded = st.file_uploader("Upload CSV (recommended: test split exported from training script)", type=["csv"])
    st.info("Tip: run `python -m model.train_models` once to generate a ready-to-upload `data/test_data.csv`.")

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read CSV. Please upload a valid CSV file.")
            st.stop()

        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)

    st.subheader("2) Predict & evaluate")
    if df is None:
        st.warning("Upload a CSV to proceed.")
    else:
        missing = [c for c in DATASET_META["feature_names"] if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        X = df[DATASET_META["feature_names"]].copy()
        y_true = df[DATASET_META["target_col"]] if DATASET_META["target_col"] in df.columns else None

        model_path = ART_DIR / f"{selected_model}.joblib"
        model = joblib.load(model_path)

        # predict
        y_pred = model.predict(X)

        # proba for AUC if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            # scale to [0,1] for a pseudo-probability (monotonic); ok for AUC
            scores = model.decision_function(X)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        out = df.copy()
        out["prediction"] = y_pred
        if y_proba is not None:
            out["probability_1"] = y_proba

        st.write("Predictions (downloadable):")
        st.dataframe(out.head(20), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            data=csv_bytes,
            file_name=f"predictions_{selected_model}.csv",
            mime="text/csv",
        )

        if y_true is not None:
            # Metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan

            st.markdown("### Evaluation metrics (on uploaded data)")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Accuracy", f"{acc:.4f}")
            mcol1.metric("AUC", f"{auc:.4f}" if np.isfinite(auc) else "N/A")
            mcol2.metric("Precision", f"{prec:.4f}")
            mcol2.metric("Recall", f"{rec:.4f}")
            mcol3.metric("F1", f"{f1:.4f}")
            mcol3.metric("MCC", f"{mcc:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            st.markdown("### Confusion matrix")
            st.write(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))

            st.markdown("### Classification report")
            st.code(classification_report(y_true, y_pred, digits=4))
        else:
            st.warning("No label column found, so metrics are not computed. Add a 'target' column to evaluate.")

with col2:
    st.subheader("Model comparison (held-out test split)")
    if metrics_df.empty:
        st.warning("metrics_summary.csv not found yet. Click 'Re-train / Refresh models'.")
    else:
        show_df = metrics_df.copy()
        show_df = show_df.sort_values("F1", ascending=False)
        st.dataframe(show_df, use_container_width=True)

    st.subheader("Notes")
    st.write(
        """
- Models are trained on the built-in UCI Breast Cancer dataset (569 rows, 30 features).
- The app expects the **same feature columns** as the dataset; upload only *test data* as per assignment guidance.
- Use the sidebar button to retrain if you change code or want fresh artifacts.
"""
    )
