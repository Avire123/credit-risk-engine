# 💳 AI Credit Risk Engine

An end-to-end credit underwriting and portfolio analytics project built with Python, FastAPI, Streamlit, XGBoost, and DuckDB. This system simulates a real-world lending workflow by training a default-risk model, exposing it through an API, and visualizing portfolio health in a dashboard.

## 🌟 What this project does

- Trains a credit-risk model using synthetic lending data
- Predicts borrower default probability and underwriting decision
- Exposes model inference through a FastAPI service
- Stores prediction audit logs in DuckDB
- Visualizes portfolio metrics and model validation plots in a Streamlit dashboard
- Uses SHAP values to explain key risk drivers for each application

## 🧠 Business use case

The project is designed to mimic an AI-powered lending decision support system. It helps assess whether a borrower is likely to default and supports risk-based decisions such as:

- approving a loan
- requesting manual review
- denying a loan
- recommending a credit limit based on risk level

## 🏗️ Tech stack

- Python
- FastAPI
- Streamlit
- XGBoost
- scikit-learn
- DuckDB
- SHAP
- Plotly
- Pandas / NumPy

## 📁 Project structure

```text
credit-risk-engine/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── db_service.py
│       └── model_service.py
├── dashboard/
│   └── app.py
├── data/
│   ├── historical_credit_data.csv
│   ├── model_metrics.json
│   └── portfolio_audit.duckdb
├── models/
│   └── credit_model.joblib
├── train.py
├── requirements.txt
└── README.md
```

## 🚀 Getting started

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train the model

```bash
python train.py
```

This generates:

- `data/historical_credit_data.csv`
- `data/model_metrics.json`
- `models/credit_model.joblib`

### 4) Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000/docs`

### 5) Run the dashboard

```bash
streamlit run dashboard/app.py
```

## 📡 API endpoints

### Health check

```http
GET /health
```

### Predict credit risk

```http
POST /predict
```

Request body includes borrower and loan data such as:

- age
- annual_income
- loan_amount
- loan_term_months
- credit_score
- existing_debt
- derogatory_marks
- employment_length_years

### Portfolio metrics

```http
GET /portfolio-metrics
```

Returns aggregate metrics such as:

- total loans evaluated
- approval rate
- average default risk
- exposure by risk tier
- recent decisions

## 📊 Dashboard features

The Streamlit dashboard includes:

- ⚡ live underwriting simulator
- 📊 portfolio health monitoring
- 📈 model validation and calibration plots
- 🧾 audit log review

## 🧪 Notes

This project uses synthetic data for demonstration and educational purposes. In a production setting, you would replace it with real underwriting data, implement stronger governance, and integrate MLOps workflows such as versioning, monitoring, and model approval pipelines.

## ✅ Summary

This repository demonstrates how to build a practical ML-backed credit risk solution that combines:

- model training
- API deployment
- explainability
- portfolio monitoring
- business-facing decision support

