# src/api/main.py
from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bati Bank Credit Risk API", version="1.0")

# Load best model from MLflow Registry (Staging)
try:
    model_uri = "models:/CreditRisk_LogisticRegression/Staging"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✅ Loaded model from MLflow Registry: CreditRisk_LogisticRegression (Staging)")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

@app.get("/")
def read_root():
    return {"message": "Bati Bank Credit Risk Scoring API", "status": "OK"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame (1 row)
        input_df = pd.DataFrame([features.dict()])
        
        # Predict
        proba = model.predict_proba(input_df)[0][1]  # P(high-risk)
        prediction = 1 if proba >= 0.5 else 0
        
        return PredictionResponse(
            risk_probability=float(proba),
            is_high_risk=prediction
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))