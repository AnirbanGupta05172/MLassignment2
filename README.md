# ML Assignment 2 - Multi-Model Classification + Streamlit App

**GitHub repo:** https://github.com/AnirbanGupta05172/MLassignment2  
**Live Streamlit app:** https://anirbangupta05172-mfg8aymdxx6dgfgnftvuxa.streamlit.app/

This project implements the end-to-end workflow required in **Machine Learning - Assignment 2**:
- Train **6 classification models** on one dataset
- Evaluate each model with the required metrics
- Build a **Streamlit** app (CSV upload + model selection + metrics + report/CM)
- Deploy on Streamlit Community Cloud and provide the live link

---

## a) Problem statement

Build a classification system to predict whether a breast tumor is **malignant** or **benign** using numeric features extracted from digitized FNA images.

---

## b) Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) (WDBC)  
**Source:** UCI Machine Learning Repository (public)  
**Access method:** `sklearn.datasets.load_breast_cancer`

**Why it satisfies the assignment constraints**
- Instances: **569** (>= 500)
- Features: **30** (>= 12)
- Type: **Binary classification**

**Target variable**
- `target`: 0/1 labels from the dataset loader (benign vs malignant).

---

## c) Models used and evaluation metrics

### Models implemented (6)

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Evaluation metrics (for every model)

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.982456 | 0.995370 | 0.986111 | 0.986111 | 0.986111 | 0.962302 |
| Decision Tree | 0.912281 | 0.915675 | 0.955882 | 0.902778 | 0.928571 | 0.817412 |
| K-Nearest Neighbors | 0.973684 | 0.988426 | 0.960000 | 1.000000 | 0.979592 | 0.944155 |
| Naive Bayes (Gaussian) | 0.938596 | 0.987765 | 0.945205 | 0.958333 | 0.951724 | 0.867553 |
| Random Forest (Ensemble) | 0.947368 | 0.993717 | 0.958333 | 0.958333 | 0.958333 | 0.886905 |
| XGBoost (Ensemble) | 0.956140 | 0.995040 | 0.946667 | 0.986111 | 0.965986 | 0.905824 |


---

## Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the best overall performance with the highest accuracy (0.9825), excellent AUC (0.9954) and the strongest MCC (0.9623), indicating very reliable classification. |
| Decision Tree | Shows the weakest generalization (accuracy 0.9123, AUC 0.9157, lowest MCC 0.8174), suggesting higher variance/overfitting compared to other models. |
| K-Nearest Neighbors | Delivers near-top results (accuracy 0.9737, AUC 0.9884) and perfect recall (1.0000), meaning it avoids false negatives on this test split. |
| Naive Bayes (Gaussian) | Provides a solid baseline with high AUC (0.9878) and balanced precision/recall (~0.95), though overall accuracy (0.9386) is below the top models. |
| Random Forest (Ensemble) | Produces robust, balanced predictions (precision=recall=F1=0.9583) with very high AUC (0.9937), but accuracy (0.9474) is mid-range among models. |
| XGBoost (Ensemble) | Performs strongly with near-best AUC (0.9950) and high recall (0.9861), but slightly lower precision (0.9467) indicates a few more false positives than Logistic Regression. |


---

## Project structure

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- model/
|   |-- train_models.py
|   `-- artifacts/          # generated after training
`-- data/
    `-- test_data.csv       # generated after training (for Streamlit upload)
```

---

## Run locally (BITS Virtual Lab / local)

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### 2) Train and generate artifacts

```bash
python -m model.train_models
```

Creates:
- `model/artifacts/*.joblib` (saved models)
- `model/artifacts/metrics_summary.csv` (metrics table)
- `data/test_data.csv` (file to upload in Streamlit)

### 3) Run the Streamlit app

```bash
streamlit run app.py
```

---

## How to use the Streamlit app

1. Click **"Re-train / Refresh models"** (uses built-in dataset) if required.
2. Upload the CSV file: `data/test_data.csv`
3. Choose the model from the dropdown.
4. View evaluation metrics + classification report (or confusion matrix).
5. Download predictions as CSV.

---

## Deployment (Streamlit Community Cloud)

1. Streamlit Cloud -> New App
2. Select GitHub repository + branch (`main`)
3. Select `app.py`
4. Deploy and verify the app opens and runs

---
