"""
main.py - FastAPI application for serving the credit risk model

"""

import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import os
import pandas as pd
from src.api.pydantic_models import CustomerFeatures, PredictionResponse, HealthResponse

# SETUP LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model variable
_model = None

# MODEL LOADING
# ============================================================
def load_model():
    """Load the trained model from disk."""
    global _model
    # Path to model: project_root/models/credit_risk_best.pkl
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "credit_risk_best.pkl"
    )
    try:
        _model = joblib.load(model_path)
        logger.info(f" Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f" Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def predict(features_dict: dict) -> dict:
    """Run prediction using the loaded model."""
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Convert dict to DataFrame (1 row)
    input_df = pd.DataFrame([features_dict])
    
    # Use predict_proba — NOT predict — to get probabilities
    proba_array = _model.predict_proba(input_df)
    prob_high_risk = proba_array[0][1]  # P(is_high_risk = 1)
    is_high_risk = 1 if prob_high_risk >= 0.5 else 0
    
    return {
        "risk_probability": prob_high_risk,
        "is_high_risk": is_high_risk
    }



# APPLICATION STARTUP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; cleanup at shutdown."""
    logger.info(" Starting up the Credit Risk API...")
    load_model()
    yield
    logger.info(" Shutting down the Credit Risk API...")

# CREATE FASTAPI APP
# ============================================================
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Predicts BNPL default risk using RFM-based behavioral model",
    version="1.0.0",
    lifespan=lifespan
)

# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(features: CustomerFeatures):
    """
    Predict credit risk for a customer.
    
    Example request body:
    {
        "total_amount": 15000,
        "avg_amount": 1500,
        "n_transactions": 10,
        "total_value": 15000,
        "avg_value": 1500,
        "std_value": 200,
        "fraud_rate": 0.0
    }
    """
    try:
        features_dict = features.model_dump()  # or .dict() for Pydantic v1
        result = predict(features_dict)
        
        return PredictionResponse(
            risk_probability=result["risk_probability"],
            is_high_risk=result["is_high_risk"]
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# RUN FOR DEVELOPMENT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)