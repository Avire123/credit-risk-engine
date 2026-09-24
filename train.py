import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def generate_synthetic_credit_data(num_samples: int = 10000, random_seed: int = 42):
    """Generates realistic synthetic credit underwriting data."""
    np.random.seed(random_seed)

    age = np.random.randint(21, 70, size=num_samples)
    annual_income = np.random.lognormal(mean=10.8, sigma=0.6, size=num_samples)
    loan_amount = np.random.uniform(2000, 50000, size=num_samples)
    loan_term_months = np.random.choice([24, 36, 48, 60], size=num_samples)
    credit_score = np.random.randint(300, 850, size=num_samples)
    existing_debt = np.random.uniform(500, 40000, size=num_samples)
    derogatory_marks = np.random.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03], size=num_samples)
    employment_length_years = np.random.randint(0, 25, size=num_samples)

    # Derived Features
    dti_ratio = (existing_debt + (loan_amount / loan_term_months)) / (annual_income / 12)
    loan_to_income = loan_amount / annual_income

    # Simulate Default Probability Logit
    logit = (
        -0.008 * (credit_score - 600)
        + 1.8 * dti_ratio
        + 1.2 * loan_to_income
        + 0.5 * derogatory_marks
        - 0.05 * employment_length_years
        - 1.5
    )
    
    prob_default = 1 / (1 + np.exp(-logit))
    default_flag = (np.random.rand(num_samples) < prob_default).astype(int)

    return pd.DataFrame({
        "age": age,
        "annual_income": np.round(annual_income, 2),
        "loan_amount": np.round(loan_amount, 2),
        "loan_term_months": loan_term_months,
        "credit_score": credit_score,
        "existing_debt": np.round(existing_debt, 2),
        "derogatory_marks": derogatory_marks,
        "employment_length_years": employment_length_years,
        "dti_ratio": np.round(dti_ratio, 4),
        "loan_to_income": np.round(loan_to_income, 4),
        "default": default_flag
    })

def train_and_save_model():
    print("Generating synthetic credit portfolio dataset...")
    data = generate_synthetic_credit_data()
    data.to_csv("data/historical_credit_data.csv", index=False)

    X = data.drop(columns=["default"])
    y = data["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_cols = list(X.columns)

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)]
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print("Training XGBoost Credit Risk Model...")
    pipeline.fit(X_train, y_train)

    # Evaluation Metrics
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    # Compute Curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

    # Save metrics to JSON for Streamlit
    metrics_payload = {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "calibration": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    }

    with open("data/model_metrics.json", "w") as f:
        json.dump(metrics_payload, f)

    print("\n--- Model Evaluation Summary ---")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier Score: {brier:.4f}")

    joblib.dump({"pipeline": pipeline, "feature_names": feature_cols}, "models/credit_model.joblib")
    print("Model artifact & metrics JSON saved successfully!\n")

if __name__ == "__main__":
    train_and_save_model()