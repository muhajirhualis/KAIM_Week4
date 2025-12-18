
from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    total_amount: float
    avg_amount: float
    n_transactions: float
    total_value: float
    avg_value: float
    std_value: float
    fraud_rate: float

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
    message: str = "Success"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
    