import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from src.feature_engineering import CustomerAggregator
from src.target_proxy import HighRiskLabeler

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'], 
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'Value': [100, 200, 50, 75, 300],
        'Amount': [100, 200, 50, 75, 300],
        'FraudResult': [0, 0, 1, 0, 0],
        'TransactionStartTime': pd.to_datetime([
            '2023-01-01', '2023-01-05', 
            '2023-01-10', '2023-01-12', 
            '2023-01-15'
        ])
    })

def test_customer_aggregator(sample_data):
    agg = CustomerAggregator()
    out = agg.transform(sample_data)
    
    assert 'CustomerId' in out.columns
    assert 'n_transactions' in out.columns
    assert 'total_value' in out.columns
    assert len(out) == 3  # 3 customers
    assert out.loc[out['CustomerId'] == 'C1', 'n_transactions'].iloc[0] == 2

def test_rfm_labeler(sample_data):
    labeler = HighRiskLabeler(random_state=42)
    out = labeler.fit_transform(sample_data)
    
    assert 'CustomerId' in out.columns
    assert 'is_high_risk' in out.columns
    assert set(out['is_high_risk'].unique()).issubset({0, 1})
    assert len(out) == 3