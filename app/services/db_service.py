import duckdb
import os
import json
from datetime import datetime

DB_PATH = "data/portfolio_audit.duckdb"

def init_db():
    """Initializes DuckDB table for logging prediction requests."""
    os.makedirs("data", exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loan_audit_log (
            request_id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            age INTEGER,
            annual_income DOUBLE,
            loan_amount DOUBLE,
            loan_term_months INTEGER,
            credit_score INTEGER,
            existing_debt DOUBLE,
            derogatory_marks INTEGER,
            employment_length_years INTEGER,
            default_probability DOUBLE,
            risk_tier VARCHAR,
            decision VARCHAR,
            recommended_credit_limit DOUBLE,
            shap_explanations VARCHAR
        )
    """)
    conn.close()

def log_prediction(request_id: str, payload: dict, result: dict):
    """Logs an incoming credit request and prediction output to DuckDB."""
    conn = duckdb.connect(DB_PATH)
    conn.execute("""
        INSERT INTO loan_audit_log VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        request_id,
        datetime.utcnow(),
        payload["age"],
        payload["annual_income"],
        payload["loan_amount"],
        payload["loan_term_months"],
        payload["credit_score"],
        payload["existing_debt"],
        payload["derogatory_marks"],
        payload["employment_length_years"],
        result["default_probability"],
        result["risk_tier"],
        result["decision"],
        result["recommended_credit_limit"],
        json.dumps(result["shap_explanations"])
    ))
    conn.close()

def get_portfolio_summary():
    """Fetches macro-level metrics for the Portfolio Monitor dashboard."""
    conn = duckdb.connect(DB_PATH)
    
    total = conn.execute("SELECT COUNT(*) FROM loan_audit_log").fetchone()[0]
    if total == 0:
        conn.close()
        return {
            "total_loans_evaluated": 0,
            "approval_rate": 0.0,
            "average_default_probability": 0.0,
            "exposure_by_risk_tier": {"Low": 0.0, "Medium": 0.0, "High": 0.0},
            "recent_logs": []
        }

    approved_count = conn.execute("SELECT COUNT(*) FROM loan_audit_log WHERE decision = 'Approved'").fetchone()[0]
    avg_default = conn.execute("SELECT AVG(default_probability) FROM loan_audit_log").fetchone()[0] or 0.0

    # Risk tier exposure aggregation
    exposure_df = conn.execute("""
        SELECT risk_tier, SUM(loan_amount) as total_exposure 
        FROM loan_audit_log 
        GROUP BY risk_tier
    """).fetchdf()
    
    exposure_map = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
    for _, row in exposure_df.iterrows():
        exposure_map[row["risk_tier"]] = float(row["total_exposure"])

    # Recent activity logs
    recent_logs = conn.execute("""
        SELECT request_id, timestamp, loan_amount, credit_score, default_probability, risk_tier, decision
        FROM loan_audit_log
        ORDER BY timestamp DESC
        LIMIT 100
    """).fetchdf().to_dict(orient="records")

    conn.close()

    return {
        "total_loans_evaluated": total,
        "approval_rate": round(approved_count / total, 4),
        "average_default_probability": round(avg_default, 4),
        "exposure_by_risk_tier": exposure_map,
        "recent_logs": recent_logs
    }