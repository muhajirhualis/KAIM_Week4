# tests/test_smoke.py
import pytest
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

def test_imports():
    """Smoke test: ensure core modules can be imported."""
    try:
        from eda_utils import eda_load_data
        assert True
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_data_load_smoke():
    """Smoke test: verify sample data loads without crashing."""
    from eda_utils import eda_load_data
    try:
        # Use relative path assuming tests run from repo root
        df = eda_load_data("data/raw/Xente_DataSet.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "CustomerId" in df.columns
    except FileNotFoundError:
        pytest.skip("Raw data not available in CI — skipping data load test")
    except Exception as e:
        pytest.fail(f"Data load failed: {e}")