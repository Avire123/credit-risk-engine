import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Credit Risk & Portfolio Intelligence",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI Credit Risk Underwriting & Portfolio Intelligence")
st.markdown("Real-time credit scoring engine, explainable AI (SHAP), and model risk validation.")

# Sidebar Navigation
tab_selection = st.sidebar.radio(
    "Navigation", 
    ["Underwriter Simulator", "Portfolio Health Monitor", "Model Validation & Calibration", "API Audit Logs"]
)

# Check backend status
try:
    health_res = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("API Status: Connected 🟢")
    else:
        st.sidebar.error("API Status: Degraded 🟡")
except Exception:
    st.sidebar.error("API Status: Offline 🔴 (Run 'uvicorn app.main:app' in terminal)")

# -------------------------------------------------------------------
# TAB 1: Live Underwriter Simulator
# -------------------------------------------------------------------
if tab_selection == "Underwriter Simulator":
    st.header("⚡ Live Loan Application Scoring")
    st.caption("Adjust borrower parameters to perform real-time model inference.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Borrower & Loan Details")
        age = st.slider("Age", 18, 85, 35)
        annual_income = st.number_input("Annual Income ($)", value=75000.0, step=5000.0)
        loan_amount = st.number_input("Requested Loan Amount ($)", value=15000.0, step=1000.0)
        loan_term_months = st.selectbox("Loan Term (Months)", [24, 36, 48, 60], index=1)
        credit_score = st.slider("Credit Score (FICO)", 300, 850, 710)
        existing_debt = st.number_input("Existing Debt ($)", value=12000.0, step=1000.0)
        derogatory_marks = st.selectbox("Derogatory Marks / Delinquencies", [0, 1, 2, 3], index=0)
        employment_length_years = st.slider("Employment Length (Years)", 0, 30, 5)

        submit_btn = st.button("Evaluate Credit Risk", type="primary", use_container_width=True)

    with col2:
        st.subheader("Underwriting Decision & SHAP Attribution")
        
        if submit_btn:
            payload = {
                "age": age,
                "annual_income": annual_income,
                "loan_amount": loan_amount,
                "loan_term_months": loan_term_months,
                "credit_score": credit_score,
                "existing_debt": existing_debt,
                "derogatory_marks": derogatory_marks,
                "employment_length_years": employment_length_years
            }

            try:
                response = requests.post(f"{API_BASE_URL}/predict", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    decision = data["decision"]
                    prob = data["default_probability"]
                    
                    if decision == "Approved":
                        st.success(f"### Decision: APPROVED (Default Risk: {prob:.1%})")
                    elif decision == "High Risk Review":
                        st.warning(f"### Decision: HIGH RISK REVIEW (Default Risk: {prob:.1%})")
                    else:
                        st.error(f"### Decision: DENIED (Default Risk: {prob:.1%})")

                    st.metric("Recommended Credit Limit", f"${data['recommended_credit_limit']:,.2f}")

                    st.write("**Feature Impact Breakdown (SHAP Values)**")
                    shap_df = pd.DataFrame(
                        list(data["shap_explanations"].items()), 
                        columns=["Feature", "SHAP Impact"]
                    ).sort_values(by="SHAP Impact", ascending=True)

                    fig = px.bar(
                        shap_df, 
                        x="SHAP Impact", 
                        y="Feature", 
                        orientation="h",
                        color="SHAP Impact",
                        color_continuous_scale="RdYlGn_r",
                        title="Risk Contributors (Positive = Higher Default Risk)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("API error during prediction.")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server: {str(e)}")

# -------------------------------------------------------------------
# TAB 2: Portfolio Health Monitor
# -------------------------------------------------------------------
elif tab_selection == "Portfolio Health Monitor":
    st.header("📊 Macro-Level Portfolio Analytics")
    
    try:
        res = requests.get(f"{API_BASE_URL}/portfolio-metrics")
        if res.status_code == 200:
            metrics = res.json()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Loans Evaluated", metrics["total_loans_evaluated"])
            m2.metric("Overall Approval Rate", f"{metrics['approval_rate']:.1%}")
            m3.metric("Average Default Risk", f"{metrics['average_default_probability']:.1%}")

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Capital Exposure by Risk Tier")
                exposure = metrics["exposure_by_risk_tier"]
                fig_pie = px.pie(
                    values=list(exposure.values()), 
                    names=list(exposure.keys()), 
                    color=list(exposure.keys()),
                    color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                st.subheader("Recent Applications: Credit Score vs. Risk")
                if metrics["recent_logs"]:
                    df_logs = pd.DataFrame(metrics["recent_logs"])
                    fig_scatter = px.scatter(
                        df_logs, 
                        x="credit_score", 
                        y="default_probability", 
                        color="risk_tier",
                        size="loan_amount",
                        hover_data=["request_id", "decision"],
                        color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No applications logged yet.")

    except Exception as e:
        st.error(f"Error fetching portfolio metrics: {str(e)}")

# -------------------------------------------------------------------
# TAB 3: Model Validation & Calibration
# -------------------------------------------------------------------
elif tab_selection == "Model Validation & Calibration":
    st.header("📈 Model Validation & Risk Calibration Metrics")
    st.caption("Quantitative performance benchmarks evaluating discriminatory power and probability calibration.")

    metrics_file = "data/model_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            m_data = json.load(f)

        # High level score metrics
        k1, k2, k3 = st.columns(3)
        k1.metric("ROC-AUC Score", f"{m_data['roc_auc']:.4f}", help="Measures discrimination power (0.5 = random, 1.0 = perfect)")
        k2.metric("PR-AUC (Avg Precision)", f"{m_data['pr_auc']:.4f}", help="Precision-Recall Area Under Curve")
        k3.metric("Brier Score Loss", f"{m_data['brier_score']:.4f}", help="Measures probability calibration accuracy (lower is better)")

        st.markdown("---")

        col_c1, col_c2 = st.columns(2)

        # 1. Calibration Curve Plot
        with col_c1:
            st.subheader("Probability Calibration Curve")
            prob_true = m_data["calibration"]["prob_true"]
            prob_pred = m_data["calibration"]["prob_pred"]

            fig_cal = go.Figure()
            # Perfectly calibrated reference line
            fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="gray")))
            # Model calibration line
            fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="XGBoost Model", line=dict(color="#2980b9", width=3)))
            
            fig_cal.update_layout(
                xaxis_title="Predicted Probability",
                yaxis_title="True Proportion of Defaults",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(x=0.05, y=0.95)
            )
            st.plotly_chart(fig_cal, use_container_width=True)

        # 2. ROC & Precision-Recall Curves
        with col_c2:
            st.subheader("ROC Curve (Receiver Operating Characteristic)")
            fpr = m_data["roc_curve"]["fpr"]
            tpr = m_data["roc_curve"]["tpr"]

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(dash="dash", color="gray")))
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC = {m_data['roc_auc']:.2f})", line=dict(color="#27ae60", width=3)))

            fig_roc.update_layout(
                xaxis_title="False Positive Rate (FPR)",
                yaxis_title="True Positive Rate (TPR)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(x=0.6, y=0.1)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    else:
        st.warning("Model metrics file not found. Run 'python train.py' to generate performance benchmarks.")

# -------------------------------------------------------------------
# TAB 4: API Audit Logs
# -------------------------------------------------------------------
elif tab_selection == "API Audit Logs":
    st.header("📜 Live DuckDB Prediction Logs")
    
    try:
        res = requests.get(f"{API_BASE_URL}/portfolio-metrics")
        if res.status_code == 200:
            metrics = res.json()
            if metrics["recent_logs"]:
                st.dataframe(pd.DataFrame(metrics["recent_logs"]), use_container_width=True)
            else:
                st.info("No audit logs recorded in DuckDB yet.")
    except Exception as e:
        st.error(f"Failed to fetch logs: {str(e)}")