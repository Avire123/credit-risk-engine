import joblib
import pandas as pd
import numpy as np
import shap
import uuid

class CreditModelService:
    def __init__(self, model_path: str = "models/credit_model.joblib"):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.feature_names = artifact["feature_names"]
        
        # Extract base transformer and XGBoost classifier for SHAP
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        
        # Initialize TreeExplainer on classifier
        self.explainer = shap.TreeExplainer(self.classifier)

    def predict(self, raw_input: dict) -> dict:
        request_id = str(uuid.uuid4())[:8]

        # Calculate dynamic engineering features
        dti_ratio = (raw_input["existing_debt"] + (raw_input["loan_amount"] / raw_input["loan_term_months"])) / (raw_input["annual_income"] / 12)
        loan_to_income = raw_input["loan_amount"] / raw_input["annual_income"]

        full_features = {
            "age": raw_input["age"],
            "annual_income": raw_input["annual_income"],
            "loan_amount": raw_input["loan_amount"],
            "loan_term_months": raw_input["loan_term_months"],
            "credit_score": raw_input["credit_score"],
            "existing_debt": raw_input["existing_debt"],
            "derogatory_marks": raw_input["derogatory_marks"],
            "employment_length_years": raw_input["employment_length_years"],
            "dti_ratio": round(dti_ratio, 4),
            "loan_to_income": round(loan_to_income, 4)
        }

        df_input = pd.DataFrame([full_features])[self.feature_names]

        # Model Inference
        prob_default = float(self.pipeline.predict_proba(df_input)[0, 1])

        # Risk Classification & Decision Logic
        if prob_default < 0.15:
            risk_tier = "Low"
            decision = "Approved"
            recommended_limit = round(raw_input["annual_income"] * 0.35, 2)
        elif prob_default < 0.35:
            risk_tier = "Medium"
            decision = "High Risk Review"
            recommended_limit = round(raw_input["annual_income"] * 0.15, 2)
        else:
            risk_tier = "High"
            decision = "Denied"
            recommended_limit = 0.0

        # SHAP Explanations
        scaled_input = self.preprocessor.transform(df_input)
        shap_values = self.explainer.shap_values(scaled_input)[0]
        
        shap_dict = {
            col: float(val) 
            for col, val in zip(self.feature_names, shap_values)
        }

        return {
            "request_id": request_id,
            "default_probability": round(prob_default, 4),
            "risk_tier": risk_tier,
            "decision": decision,
            "recommended_credit_limit": recommended_limit,
            "shap_explanations": shap_dict,
            "processed_payload": full_features
        }