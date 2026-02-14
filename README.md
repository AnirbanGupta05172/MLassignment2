# ML Assignment 2 — Classification + Streamlit Deployment

## a) Problem statement
Implement multiple **classification** models on a single dataset, evaluate them using required metrics, and build a **Streamlit** web application that allows:
- Uploading test CSV data
- Selecting a model from a dropdown
- Displaying evaluation metrics
- Showing a confusion matrix / classification report

## b) Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI (also available via `sklearn.datasets.load_breast_cancer`)  
**Problem type:** Binary classification  
**Minimum requirements satisfied:** 569 instances (>= 500), 30 features (>= 12)

**Target column name used in this project:** `target`  
- `0` = malignant (as encoded by sklearn)
- `1` = benign

## c) Models used (all on the same dataset)
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN) Classifier  
4. Naive Bayes Classifier (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison table (metrics on held-out test split)
> This table is auto-filled when you run training: `python -m model.train_models`

<!-- METRICS_TABLE_START -->

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| K-Nearest Neighbors | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes (Gaussian) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest (Ensemble) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost (Ensemble) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

<!-- METRICS_TABLE_END -->

### Observations (write 1–3 lines each after you run)
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | *(Fill after running)* |
| Decision Tree | *(Fill after running)* |
| kNN | *(Fill after running)* |
| Naive Bayes | *(Fill after running)* |
| Random Forest (Ensemble) | *(Fill after running)* |
| XGBoost (Ensemble) | *(Fill after running)* |

---

## How to run on BITS Virtual Lab (required)
1. Open terminal in BITS Virtual Lab.
2. Upload / clone this project folder.
3. Create and activate a venv (optional but recommended):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
4. Install requirements:
   - `pip install -r requirements.txt`
5. Train models + generate metrics table + export test CSV:
   - `python -m model.train_models`
6. Start Streamlit app:
   - `streamlit run app.py`
7. Take **ONE screenshot** showing successful execution on BITS Virtual Lab (terminal + app running).

## Streamlit app usage
- Upload `data/test_data.csv` (generated in step 5)
- Choose a model
- App shows predictions + required metrics + confusion matrix + classification report

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repo
2. Go to Streamlit Cloud → New App
3. Select repo + branch + `app.py`
4. Deploy

**Common deployment fix:** If the app fails, check that `requirements.txt` includes every dependency.

---

## Repository structure
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- .streamlit/config.toml
│-- model/
│   │-- train_models.py
│   │-- make_readme.py
│   │-- artifacts/   (created after training)
│-- data/
│   │-- test_data.csv (created after training)
```

## Academic integrity note
You **must** run the code yourself, use your own GitHub account, and maintain your own commit history.
Customize at least one thing (dataset choice, model hyperparameters, UI text, or additional plots) before submitting.
