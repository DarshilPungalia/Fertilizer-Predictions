# **Fertilizer Prediction - Stacked Ensemble Modeling Pipeline**

This project tackles a multi-class classification task: predicting the optimal fertilizer for a given crop-soil condition, using a rich ensemble learning strategy. The approach integrates gradient boosting methods, deep neural networks, and advanced ensembling techniques, all within a robust preprocessing and evaluation framework.

---

## 🏆 Highlights & Accomplishments

- Three-level ensemble architecture implemented:

  - Level 1: XGBoost classifiers with K-Fold ensembling

  - Level 2: LightGBM models trained on meta-features from XGBoost

  - Level 3: Neural Network meta-model trained on LightGBM predictions

- End-to-end modular pipeline for:

  - Data ingestion and transformation using ColumnTransformer

  - Hyperparameter optimization with Optuna for both XGBoost and Neural Network models

  - Evaluation using MAP@3, suitable for top-3 classification problems

- Stacked generalization to combine model strengths and minimize overfitting

- Submission-ready predictions with top-3 ranked fertilizer recommendations

- Clean and reusable modules: `ingestion.py`, `eval.py`, `xgb_pipeline.py`, `neural_pipeline.py`, `ensemble_pipeline.py`

---

## 🧠 Methodology

- ### Data Handling

  - Preprocessed via MinMaxScaler for numeric columns and OneHotEncoder for categorical columns.

  - Label encoding used for target classes.

  - Consistent preprocessing applied to both training and test datasets.

  - Modeling Pipelines

- ### XGBoost Pipeline

  - Hyperparameter tuning with Optuna (10-fold Stratified CV)

  - Optimized on MAP@3

  - Final model trained on full data

- ### Neural Network Pipeline

  - TensorFlow-based feedforward network

  - Architecture dynamically optimized (layers, dropout, activation functions, etc.)

  - Trained using Stratified K-Fold and ReduceLROnPlateau

- ### Ensemble Pipeline

  - Level 1: K-Fold ensemble of XGBoost classifiers

  - Level 2: LightGBM ensemble trained on out-of-fold (OOF) predictions of XGBoost

  - Level 3: Meta Neural Network trained on OOF predictions of LightGBM

  - Final predictions generated via softmax and top-3 label ranking

---

## 📁 Project Structure

`

    ├── data/
    │   ├── train.csv
    │   └── test.csv
    ├── ingestion.py          # Data preprocessing pipeline
    ├── eval.py               # Custom MAP@3 and top-3 prediction utilities
    ├── xgb_pipeline.py       # XGBoost modeling with Optuna tuning
    ├── neural_pipeline.py    # Neural Network training and tuning
    ├── ensemble_pipeline.py  # Stacked ensemble combining XGB + LGBM + DNN
    ├── eda.ipynb             # Exploratory Data Analysis
    └── submissions/          # Saved CSV files from various pipelines

`

---

## 📈 Performance Metrics

- MAP@3 scores evaluated throughout cross-validation and final ensemble stage.

- Level-wise model output analysis (distribution, OOF stats) included for debugging.

---

## 🙌 Acknowledgements

This project demonstrates:

- Practical use of ensemble learning in structured tabular problems.

- Integration of PyTorch, XGBoost, LightGBM, TensorFlow, and Optuna in a cohesive workflow.

- Reproducible ML pipeline design with extensibility in mind.
