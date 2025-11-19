#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cardiovascular Disease Prediction Model (XGBoost, Random Forest, Gradient Boosting)

This script:
- Loads and cleans the dataset
- Performs correlation analysis
- Splits the dataset
- Runs RandomizedSearchCV for model tuning
- Evaluates multiple models
- Saves the best model (XGBoost)
"""

# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)

import joblib
import warnings
warnings.filterwarnings("ignore")


# ==============================
# 2. FUNCTION: Load Data
# ==============================
def load_data(path):
    print(f"\nLoading data from: {path}")
    df = pd.read_csv(path)
    print("Data successfully loaded.")
    print(df.head())
    return df


# ==============================
# 3. FUNCTION: Plot Correlation Matrix
# ==============================
def plot_correlation(df):
    print("\nPlotting correlation matrix...")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr.mask(mask), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


# ==============================
# 4. FUNCTION: Split Data
# ==============================
def split_data(df):
    X = df.drop(columns=['patientid', 'target'], axis=1)
    y = df["target"]

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ==============================
# 5. FUNCTION: Hyperparameter Search
# ==============================
def run_random_search(model, param_grid, X_train, y_train, model_name):
    print(f"\nRunning RandomizedSearchCV for {model_name}...")

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    rs.fit(X_train, y_train)
    print(f"Best parameters for {model_name}:")
    print(rs.best_params_)
    return rs


# ==============================
# 6. FUNCTION: Evaluate Model
# ==============================
def evaluate_model(model, X_test, y_test, name):
    print(f"\nEvaluating {name}...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"{name} Performance:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"ROC-AUC:    {roc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    return {
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": roc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    }


# ==============================
# 7. MAIN PIPELINE
# ==============================
def main():

    # --- Load Dataset ---
    df = load_data("/home/arhcellius-anami/Documents/mtb/archive/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv")

    # --- Correlation Plot ---
    plot_correlation(df)

    # --- Train-Test Split ---
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Hyperparameter Grids ---
    xgb_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1],
        "colsample_bytree": [0.7, 0.9, 1],
    }

    rf_params = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    gb_params = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.9, 1],
        "max_features": ["sqrt", "log2"],
    }

    # --- Run Random Searches ---
    xgb_rs = run_random_search(xgb.XGBClassifier(eval_metric="logloss"), xgb_params, X_train, y_train, "XGBoost")
    rf_rs = run_random_search(RandomForestClassifier(), rf_params, X_train, y_train, "RandomForest")
    gb_rs = run_random_search(GradientBoostingClassifier(), gb_params, X_train, y_train, "GradientBoosting")

    # --- Evaluate Models ---
    results = []
    results.append(evaluate_model(xgb_rs.best_estimator_, X_test, y_test, "XGBoost"))
    results.append(evaluate_model(rf_rs.best_estimator_, X_test, y_test, "RandomForest"))
    results.append(evaluate_model(gb_rs.best_estimator_, X_test, y_test, "GradientBoosting"))

    # --- Results Summary ---
    res_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("\n=== Final Model Comparison ===")
    print(res_df)

    # --- Save Best Model (XGBoost) ---
    best_model = xgb_rs.best_estimator_
    joblib.dump(best_model, "heart_disease_model.pkl")
    print("\nBest model saved as heart_disease_model.pkl")


# ==============================
# 8. ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
