from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class LoanApplicationRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    annual_income: float = Field(..., gt=0, example=750000.0)
    loan_amount: float = Field(..., gt=0, example=150000.0)
    loan_term_months: int = Field(..., example=36)
    credit_score: int = Field(..., ge=300, le=850, example=710)
    existing_debt: float = Field(..., ge=0, example=120000.0)
    derogatory_marks: int = Field(..., ge=0, example=0)
    employment_length_years: int = Field(..., ge=0, example=5)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "annual_income": 850000.0,
                "loan_amount": 200000.0,
                "loan_term_months": 36,
                "credit_score": 720,
                "existing_debt": 100000.0,
                "derogatory_marks": 0,
                "employment_length_years": 6
            }
        }

class PredictionResponse(BaseModel):
    request_id: str
    default_probability: float
    risk_tier: str  # Low, Medium, High
    decision: str   # Approved, High Risk Review, Denied
    recommended_credit_limit: float
    shap_explanations: Dict[str, float]

class PortfolioMetricsResponse(BaseModel):
    total_loans_evaluated: int
    approval_rate: float
    average_default_probability: float
    exposure_by_risk_tier: Dict[str, float]
    recent_logs: List[Dict]