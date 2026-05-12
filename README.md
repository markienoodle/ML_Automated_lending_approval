# ML_Automated_Lending_Approval

**Glass-Box Lending: Leveraging the Explainability of CART for Automated Loan Approval Among Unbanked Populations**

A Machine Learning Project
Department of Computer Science, College of Information and Computing Sciences
University of Santo Tomas

In Partial Fulfilment of the Requirements for the Degree Bachelor of Science in Computer Science

By:
- Balading, Mark Lawrence R.
- Manalo, Ira Sophia T.
- Quicho, Carlos M.
- Tacata, Jeydin S.

May 2026

---

## Data Source

The dataset used is the **Home Credit Default Risk** dataset, publicly available on Kaggle:
https://www.kaggle.com/c/home-credit-default-risk

You will need a Kaggle account to download it. The files needed for this project are:
- `application_train.csv` (main applicant records)
- `bureau.csv` (credit bureau enquiry records, used for population filtering)

The original dataset files are not included in this repository because of their size. All other project files (scripts, outputs) are in the Google Drive linked below:
https://drive.google.com/drive/u/1/folders/1wpaZ7b2dUQtGGspGI1dGrOnOBFtxUlG3

---

## Setup Instructions

### Requirements

- Python 3.9 or higher
- The following Python libraries:

```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

You can verify that all libraries import correctly by running:

```bash
python try.py
```

### File Structure

Place the downloaded dataset files in the same directory as the scripts before running:

```
project/
├── application_train.csv       # from Kaggle
├── bureau.csv                  # from Kaggle
├── app_minus_bureau.py         # Step 1: population filtering
├── smote.py                    # Step 2: preprocessing + SMOTE
├── cart.py                     # Step 3: model training and evaluation
└── try.py                      # optional: library check
```

---

## Reproducibility Guide

Run the scripts in order:

### Step 1 — Filter the unbanked population

```bash
python app_minus_bureau.py
```

This removes all applicants who appear in `bureau.csv` from `application_train.csv`, keeping only those with no credit bureau records. The output is `application_minus_bureau.csv`.

### Step 2 — Preprocess and apply SMOTE

```bash
python smote.py
```

This encodes categorical columns, fills missing values, splits the data 80/20, and applies SMOTE to the training set to balance the class distribution. The output is `application_minus_bureau_smote.csv`, which contains only the SMOTE-balanced training data combined back into a single file.

### Step 3 — Train and evaluate the CART model

```bash
python cart.py
```

This loads the SMOTE dataset, runs a GridSearchCV to find the best hyperparameters, trains the final CART model, and produces all evaluation outputs (plots, metrics, decision rules, saved model) in the `cart_outputs/` folder.

### Expected outputs in `cart_outputs/`

| File | Description |
|---|---|
| `01_confusion_matrix.png` | Confusion matrix on test set |
| `02_roc_curve.png` | ROC curve (AUC = 0.9273) |
| `03_precision_recall_curve.png` | Precision-Recall curve |
| `04_feature_importance_gini.png` | Top 25 features by Gini importance |
| `05_feature_importance_permutation.png` | Top 25 features by permutation importance |
| `06_learning_curve.png` | Training vs. validation ROC-AUC across sample sizes |
| `07_decision_tree_FULL.png` | Full rendered decision tree |
| `decision_rules_FULL.txt` | Complete decision rules as text |
| `feature_importance_gini.csv` | Gini importance scores for all features |
| `feature_importance_permutation.csv` | Permutation importance scores |
| `test_predictions.csv` | Test set predictions and probabilities |
| `cart_final_model.pkl` | Saved model (joblib format) |

### Notes on reproducibility

- All random operations use `random_state=42` for reproducibility.
- The GridSearchCV uses `n_jobs=1` (single-core) to avoid multiprocessing issues on Windows with Python 3.13+. This makes the tuning step slow (1,440 fits). Expected runtime on a standard laptop is 10 to 30 minutes depending on hardware.
- Results may differ slightly if you use a different Python or scikit-learn version due to changes in internal defaults. The results in the paper were produced with scikit-learn 1.x.
