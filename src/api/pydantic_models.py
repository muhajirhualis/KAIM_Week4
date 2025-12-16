
from pydantic import BaseModel
from typing import List, Optional

class CustomerFeatures(BaseModel):
    n_transactions: float
    avg_value: float
    total_value: float
    std_value: float
    fraud_rate: float
    ProductCategory_financial_services: int
    ProductCategory_airtime: int
    ProductCategory_data_bundles: int
    ProductCategory_Other: int
    ChannelId_Web: int
    ChannelId_Android: int
    ChannelId_IOS: int
    ChannelId_Other: int
    # Add other one-hot columns as needed — match your final feature set

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
    message: str = "Success"
    