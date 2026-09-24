from fastapi import FastAPI, HTTPException
from app.schemas import LoanApplicationRequest, PredictionResponse, PortfolioMetricsResponse
from app.services.model_service import CreditModelService
from app.services.db_service import init_db, log_prediction, get_portfolio_summary

# THIS IS THE MISSING VARIABLE UVICORN IS LOOKING FOR:
app = FastAPI(
    title="AI Credit Risk & Underwriting API",
    version="1.0.0",
    description="Real-time ML inference engine for credit scoring, default risk prediction, and portfolio metrics."
)

# Initialize database and model service
init_db()
model_service = CreditModelService()

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "online", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Scoring Engine"])
def predict_credit_risk(application: LoanApplicationRequest):
    try:
        input_data = application.dict()
        result = model_service.predict(input_data)
        
        # Asynchronously log prediction request to DuckDB
        log_prediction(result["request_id"], result["processed_payload"], result)
        
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/portfolio-metrics", response_model=PortfolioMetricsResponse, tags=["Portfolio Analytics"])
def fetch_portfolio_metrics():
    try:
        metrics = get_portfolio_summary()
        return PortfolioMetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval error: {str(e)}")